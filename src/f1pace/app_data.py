"""Export saved 2025 results for the read-only app; never loads a model or predicts."""

from datetime import datetime
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path

import pandas as pd

from f1pace import config

CORE_COLUMNS = ("race_id", "event_name", "circuit", "driver", "team", "lap_number", "stint", "compound",
                "tyre_age", "lap_time_s", "pred_b1", "pred_rf")
OPTIONAL_PREDICTIONS = ("pred_b2", "pred_b3", "pred_b4", "pred_ridge", "pred_xgb")
ERROR_TABLES = ("error_by_tyre_age_bucket.csv", "error_by_compound.csv", "error_by_stint_phase.csv",
                "error_by_slope_group.csv", "error_by_circuit.csv")
TABLES = ("test_summary.csv", "test_results_by_race.csv", *ERROR_TABLES, "cv_summary.csv")
PREDICTIONS_FILE = "test_predictions.parquet"
MANIFEST_FILE = "manifest.json"
EXPECTED_ROWS = 13_718
EXPECTED_RACES = 18
MAE_ATOL = 1e-12
OPTIONAL_LIMIT_BYTES = 5_000_000
MAX_TOTAL_BYTES = 10_000_000

_logger = logging.getLogger(__name__)


def _check_source(source: pd.DataFrame, expected_rows: int, expected_races: int) -> None:
    """Reject anything but the complete test season with every prediction present."""
    required = {*CORE_COLUMNS, *OPTIONAL_PREDICTIONS, "year", "split", "source_commit"} - {"event_name"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Saved predictions are missing columns: {', '.join(missing)}")
    seasons = [str(year) for year in config.TEST_YEARS]
    held_out = (source["year"].isin(config.TEST_YEARS) & source["split"].eq("test")
                & source["race_id"].astype(str).str[:4].isin(seasons))
    if not held_out.all():
        raise ValueError(f"Export must hold only {seasons} test rows; found {int((~held_out).sum())} others")
    if len(source) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(source)}")
    if source["race_id"].nunique() != expected_races:
        raise ValueError(f"Expected {expected_races} races, found {source['race_id'].nunique()}")
    predictions = ["pred_b1", "pred_rf", *OPTIONAL_PREDICTIONS]
    absent = source[predictions].isna().sum()
    if absent.any():
        raise ValueError(f"Missing predictions: {absent[absent > 0].to_dict()}")


def _attach_event_names(source: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Add event names by race, checking that the circuit agrees with the race manifest."""
    known = races[["race_id", "event_name", "circuit", "include"]].rename(columns={"circuit": "manifest_circuit"})
    joined = source.merge(known, on="race_id", how="left", validate="many_to_one")
    unnamed = sorted(joined.loc[joined["event_name"].isna(), "race_id"].unique())
    if unnamed:
        raise ValueError(f"No event_name in the race manifest for: {', '.join(unnamed)}")
    if not joined["include"].astype(bool).all():
        raise ValueError("Saved predictions contain a race the manifest excludes")
    if not joined["circuit"].astype(object).eq(joined["manifest_circuit"]).all():
        raise ValueError("The saved circuit disagrees with the race manifest")
    return joined


def _source_commits(races: pd.DataFrame, source: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return the pinned archive repository and commit per season, one each."""
    pinned = races.dropna(subset=["source_commit"]).drop_duplicates(["year", "data_source", "source_commit"])
    if pinned["year"].duplicated().any():
        raise ValueError("A season has more than one source commit in the race manifest")
    commits = {str(row.year): {"repository": row.data_source, "commit": row.source_commit} for row in pinned.itertuples()}
    saved = set(source["source_commit"].astype(str))
    expected = {commits.get(str(year), {}).get("commit") for year in config.TEST_YEARS}
    if saved != expected:
        raise ValueError("Saved prediction source commits disagree with the race manifest")
    return commits


def _reconcile_mae(frame: pd.DataFrame, by_race: pd.DataFrame) -> float:
    """Recompute per-race MAE from the export and return the largest gap to the saved table."""
    saved = by_race.set_index("race_id")
    if set(saved.index) != set(frame["race_id"]):
        raise ValueError("MAE reconciliation failed: race sets differ")
    grouped = frame.groupby("race_id")
    if not grouped.size().reindex(saved.index).eq(saved["n_laps"]).all():
        raise ValueError("MAE reconciliation failed: per-race lap counts differ")
    largest = 0.0
    for column in frame.columns[frame.columns.str.startswith("pred_")]:
        recomputed = (frame[column] - frame["lap_time_s"]).abs().groupby(frame["race_id"]).mean()
        largest = max(largest, float((recomputed.reindex(saved.index) - saved[column]).abs().max()))
    if not largest <= MAE_ATOL:
        raise ValueError(f"MAE reconciliation failed: largest gap {largest:.3e} s exceeds {MAE_ATOL:.0e} s")
    return largest


def _chosen_predictors(table_dir: Path) -> tuple[str, str]:
    """Read the frozen model and baseline labels from the saved Phase 6-7 tables."""
    models, baselines = set(), set()
    for name in ERROR_TABLES:
        table = pd.read_csv(table_dir / name)
        models.update(table["model"])
        baselines.update(table["baseline"])
    baselines.update(pd.read_csv(table_dir / "test_summary.csv")["comparison_baseline"])
    if len(models) != 1 or models - set(CORE_COLUMNS):
        raise ValueError(f"Saved tables disagree on the chosen model: {sorted(models)}")
    if len(baselines) != 1 or baselines - set(CORE_COLUMNS):
        raise ValueError(f"Saved tables disagree on the baseline: {sorted(baselines)}")
    return models.pop(), baselines.pop()


def _to_parquet(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def export_app_data(*, predictions_path: Path | None = None, table_dir: Path | None = None,
                    race_manifest_path: Path | None = None, out_dir: Path | None = None,
                    expected_rows: int = EXPECTED_ROWS, expected_races: int = EXPECTED_RACES,
                    optional_limit_bytes: int = OPTIONAL_LIMIT_BYTES, max_total_bytes: int = MAX_TOTAL_BYTES) -> dict:
    """Validate saved results, then write the app files and return their manifest."""
    predictions_path = predictions_path or config.PROCESSED_DIR / PREDICTIONS_FILE
    table_dir = table_dir or config.TABLE_DIR
    race_manifest_path = race_manifest_path or config.TABLE_DIR / "race_manifest.csv"
    out_dir = out_dir or config.APP_DATA_DIR

    source = pd.read_parquet(predictions_path)
    _check_source(source, expected_rows, expected_races)
    races = pd.read_csv(race_manifest_path)
    commits = _source_commits(races, source)
    model, baseline = _chosen_predictors(table_dir)
    tables = {name: (table_dir / name).read_bytes() for name in TABLES}

    joined = _attach_event_names(source, races)
    text_columns = ["race_id", "event_name", "circuit", "driver", "team", "compound"]
    joined[text_columns] = joined[text_columns].astype(object)
    full = _to_parquet(joined[[*CORE_COLUMNS, *OPTIONAL_PREDICTIONS]])
    include_optional = len(full) + sum(map(len, tables.values())) < optional_limit_bytes
    if not include_optional:
        _logger.info("Optional predictions dropped: the fuller export reaches the %d byte limit", optional_limit_bytes)
    payload = full if include_optional else _to_parquet(joined[list(CORE_COLUMNS)])

    exported = pd.read_parquet(BytesIO(payload))
    by_race = pd.read_csv(BytesIO(tables["test_results_by_race.csv"]))
    largest_gap = _reconcile_mae(exported, by_race)

    contents = {PREDICTIONS_FILE: payload, **tables}
    files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                    "rows": len(exported) if name == PREDICTIONS_FILE else len(pd.read_csv(BytesIO(data)))}
             for name, data in contents.items()}
    manifest = {
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "exported_by": "scripts/08_export_app_data.py",
        "n_races": int(exported["race_id"].nunique()),
        "chosen_model": model,
        "baseline": baseline,
        "optional_predictions_included": include_optional,
        "source_commits": commits,
        "reconciliation": {"atol": MAE_ATOL, "max_abs_diff": largest_gap,
                           "reference": "test_results_by_race.csv"},
        "files": files,
    }
    document = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    total = sum(map(len, contents.values())) + len(document)
    if total >= max_total_bytes:
        raise ValueError(f"Export size {total} bytes reaches the {max_total_bytes} byte limit")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in {**contents, MANIFEST_FILE: document}.items():
        (out_dir / name).write_bytes(data)
    _logger.info("Exported %d files, %d bytes, to %s", len(contents) + 1, total, out_dir)
    return manifest
