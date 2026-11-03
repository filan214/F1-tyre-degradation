"""Build the Phase 2 manifest for the configured training and test seasons."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from f1pace import config, data, manifest, plots, reporting


def run_manifest(*, source: str = "auto") -> list[Path]:
    """Save matching manifest copies, decision counts, review evidence and fig04."""
    config.ensure_dirs()
    if source == "fastf1":
        data.enable_cache()
    path = config.PROCESSED_DIR / "race_manifest.csv"
    previous = manifest.read_manifest(path) if path.exists() else None
    frame = manifest.build_manifest(source=source, previous=previous)
    copy = config.TABLE_DIR / path.name
    frame.to_csv(path, index=False)
    frame.to_csv(copy, index=False)
    summary = config.TABLE_DIR / "race_manifest_summary.csv"
    manifest.summarise_manifest(frame).to_csv(summary, index=False)
    review = config.TABLE_DIR / "race_manifest_review.csv"
    manifest.manual_review_table(frame).to_csv(review, index=False)
    note = "Data: FastF1" if source == "fastf1" else "Data: FastF1/Ergast via TracingInsights archive"
    figure = plots.plot_race_manifest(frame, source_note=note)
    outputs = [path, copy, summary, review, figure]
    reporting.append_phase2_log("race manifest", reporting.manifest_log_details(frame) + "\nSaved:\n\n" + "\n".join(f"- `{item.relative_to(config.ROOT).as_posix()}`" for item in outputs))
    print(manifest.summarise_manifest(frame).to_string(index=False))
    print(f"Manifest: {len(frame)} races, {frame['include'].sum()} included, {(~frame['include']).sum()} excluded; {frame['needs_manual_review'].sum()} flagged for review.")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    """Run the manifest build using the free verified archive by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "archive", "fastf1"], default="auto")
    args = parser.parse_args(argv)
    reporting.configure_logging()
    try:
        run_manifest(source=args.source)
    except Exception as exc:
        logging.getLogger(__name__).error("Manifest build failed: %s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
