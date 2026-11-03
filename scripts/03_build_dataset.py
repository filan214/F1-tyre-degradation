"""Build clean race laps (part A) and fixed past-only features (part B)."""

import argparse
from collections.abc import Sequence
import hashlib
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from f1pace import config, data, dataset, features, manifest, plots, reporting


def run_dataset(*, source: str = "auto") -> list[Path]:
    """Export the combined audit and publish parquet only for a complete build."""
    config.ensure_dirs()
    if source == "fastf1":
        data.enable_cache()
    frame = manifest.read_manifest(config.PROCESSED_DIR / "race_manifest.csv")
    result = dataset.assemble_clean_laps(frame, source=source)
    audit_path = config.TABLE_DIR / "audit_all_races.csv"
    failure_path = config.TABLE_DIR / "dataset_load_errors.csv"
    review_path = config.TABLE_DIR / "dataset_review.csv"
    result.audit.to_csv(audit_path, index=False)
    result.failures.to_csv(failure_path, index=False)
    result.review.to_csv(review_path, index=False)
    details = reporting.manifest_log_details(frame)
    if not result.failures.empty or result.laps.empty:
        reporting.append_phase2_log("clean dataset incomplete", details + "\nIncluded-race failures:\n\n" + reporting.markdown_table(result.failures) + "\n\nNo new complete parquet was published. Any existing parquet is from an earlier run.\n")
        raise RuntimeError(f"Dataset incomplete: {len(result.failures)} included-race failures; {len(result.laps)} successfully prepared laps. See {failure_path}")
    path = config.PROCESSED_DIR / "laps_clean.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    result.laps.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)
    totals = result.laps.groupby("split").size().rename("clean_laps").reset_index()
    years = result.laps.groupby(["year", "split"])["race_id"].agg(races="nunique", clean_laps="size").reset_index()
    outputs = [path, audit_path, failure_path, review_path]
    body = details + "\nTotal clean laps per split:\n\n" + reporting.markdown_table(totals) + "\n\nClean laps by year:\n\n" + reporting.markdown_table(years)
    body += "\n\nDataset diagnostics (unmatched weather retained as missing):\n\n" + reporting.markdown_table(result.review)
    body += f"\n\nAll {frame['include'].sum()} included races were assembled; {len(result.laps)} clean laps; {len(result.audit)} audit rows; zero included-race load errors.\n\nSaved:\n\n"
    body += "\n".join(f"- `{item.relative_to(config.ROOT).as_posix()}`" for item in outputs)
    body += "\n\nPart A complete: manifest copies, fig04, complete parquet and combined audit saved; decision/reason counts and manual directives logged. Feature generation is handled separately by part B.\n"
    reporting.append_phase2_log("clean laps complete", body)
    print(years.to_string(index=False))
    print(totals.to_string(index=False))
    print(f"Part A saved {len(result.laps)} clean laps and {len(result.audit)} audit rows.")
    return outputs


def run_features() -> list[Path]:
    """Use cached clean laps to save features, measured eligibility and training fig05."""
    config.ensure_dirs()
    input_path = config.PROCESSED_DIR / "laps_clean.parquet"
    clean = pd.read_parquet(input_path)
    frame, audit = features.build_features_with_audit(clean)
    audit_path = config.TABLE_DIR / "feature_eligibility_audit.csv"
    audit.to_csv(audit_path, index=False)
    counts = clean.groupby("year").size().rename("clean_laps").reset_index()
    counts["split"] = counts["year"].map(manifest.split_for_year)
    eligible = frame.groupby("year").size()
    counts["feature_rows"] = counts["year"].map(eligible).fillna(0).astype(int)
    counts["dropped_rows"] = counts["clean_laps"] - counts["feature_rows"]
    totals = counts.groupby("split")[["clean_laps", "feature_rows", "dropped_rows"]].sum().reset_index()
    body = (
        f"Input: `data/processed/laps_clean.parquet`; SHA-256 `{hashlib.sha256(input_path.read_bytes()).hexdigest()}`.\n\n"
        "Current lap cleanliness is guaranteed by the Phase 2 input: F1 records this precondition with zero additional removals. "
        "F2–F4 give ordered first-removal counts, so each ineligible clean lap is counted once. No cleaning threshold is recalculated during feature construction.\n\n"
        "Eligibility before/after each rule, per split:\n\n" + reporting.markdown_table(audit)
        + "\n\nRow counts per split:\n\n" + reporting.markdown_table(totals)
        + "\n\nRow counts per year:\n\n" + reporting.markdown_table(counts)
    )
    if frame.loc[frame["split"].eq("train")].empty:
        reporting.append_phase3_log("feature build incomplete", body + "\n\nNo eligible training rows survived; no new features parquet or figure was published.\n")
        raise RuntimeError("No eligible training rows for the Phase 3 feature build")
    source_note = "Data: FastF1/Ergast via TracingInsights archive" if "data_source" in frame and frame["data_source"].str.startswith("TracingInsights").any() else "Data: FastF1"
    figure = plots.plot_degradation_by_compound(frame, source_note=source_note)
    path = config.PROCESSED_DIR / "features.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)
    outputs = [path, audit_path, figure]
    body += "\n\nFeature contract: lag1–lag3 are clean, consecutive and in the same stint; optional lag4/lag5 stay missing across a gap or stint change and are excluded from model inputs. Stint slope/count use only strictly earlier clean laps in that stint, including earlier laps before a gap. Current lap time appears only on the target side.\n\n"
    body += "Fig05 uses configured training seasons only; median target delta by tyre age and compound is labelled associative, not causal. No test targets were plotted or analysed.\n\n"
    body += "The user authorized the fixed Phase 3 transformation and row counts for both splits. No model was trained; no prediction, baseline score, tuning or evaluation metric was produced. The final-evaluation marker remains untouched.\n\nSaved:\n\n"
    body += "\n".join(f"- `{item.relative_to(config.ROOT).as_posix()}`" for item in outputs)
    body += "\n\nPhase 3 outputs complete. Stop before Phase 4: no baselines or model training.\n"
    reporting.append_phase3_log("features and eligibility counts", body)
    print(audit.to_string(index=False))
    print(totals.to_string(index=False))
    print(f"Part B saved {len(frame)} feature rows and training-only fig05. No model trained.")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    """Choose cached features, clean laps, or both; never run model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "archive", "fastf1"], default="auto")
    parser.add_argument("--part", choices=["all", "clean", "features"], default="all", help="features reuses saved clean laps; clean runs only Phase 2 part A")
    args = parser.parse_args(argv)
    reporting.configure_logging()
    try:
        if args.part in {"all", "clean"}:
            run_dataset(source=args.source)
        if args.part in {"all", "features"}:
            run_features()
    except Exception as exc:
        logging.getLogger(__name__).error("Dataset build failed: %s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
