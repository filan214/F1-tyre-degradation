"""Measured preparation summaries and append-only experiment entries."""

from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

import pandas as pd

from f1pace import config, data, manifest


def configure_logging() -> None:
    """Keep per-race progress visible without duplicate FastF1 or per-rule noise."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s", stream=sys.stdout)
    logging.getLogger("fastf1").propagate = False
    data.fastf1.set_log_level(logging.INFO)
    logging.getLogger("f1pace.cleaning").setLevel(logging.WARNING)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render measured table values without adding a tabulate dependency."""
    if frame.empty:
        return "None."
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def _append_log(phase: int, title: str, body: str) -> None:
    path = config.ROOT / "EXPERIMENT_LOG.md"
    document = path.read_text(encoding="utf-8")
    marker = "\n---\n\n## Key results summary"
    if marker not in document:
        raise ValueError("EXPERIMENT_LOG.md is missing the Entries section boundary")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    command = "& " + subprocess.list2cmdline([sys.executable, *sys.argv])
    entry = f"### {timestamp[:10]} — Phase {phase} — {title}\n\nTimestamp: {timestamp}\n\nCommand: `{command}`\n\n{body.rstrip()}\n"
    path.write_text(document.replace(marker, "\n\n" + entry + marker, 1), encoding="utf-8")


def append_phase2_log(title: str, body: str) -> None:
    """Append a measured Phase 2 entry without replacing earlier history."""
    _append_log(2, title, body)


def append_phase3_log(title: str, body: str) -> None:
    """Append a measured Phase 3 entry without replacing earlier history."""
    _append_log(3, title, body)


def append_phase4_log(title: str, body: str) -> None:
    """Append measured training baseline results without altering prior entries."""
    _append_log(4, title, body)


def append_phase5_log(title: str, body: str) -> None:
    """Append measured training CV results without altering prior history."""
    _append_log(5, title, body)


def append_phase6_log(title: str, body: str) -> None:
    """Append full held-out results or failed opening attempts to the experiment log."""
    _append_log(6, title, body)


def append_phase7_log(title: str, body: str) -> None:
    """Append descriptive analysis of the already saved final-test predictions."""
    _append_log(7, title, body)


def manifest_log_details(frame: pd.DataFrame) -> str:
    """Describe decisions, provenance, overrides and diagnostic review evidence."""
    counts = manifest.summarise_manifest(frame)
    sources = frame.loc[frame["data_source"].ne(""), ["year", "data_source", "source_commit"]].drop_duplicates()
    overrides = frame.loc[frame["manual_override"].ne(""), ["race_id", "manual_override", "manual_note", "include", "exclude_reason"]]
    reviews = manifest.manual_review_table(frame)
    overview = frame.groupby(["year", "split"])["include"].agg(included="sum", scheduled="size").reset_index()
    overview["excluded"] = overview["scheduled"] - overview["included"]
    return (
        "Fixed eligibility: wet-compound share >5% or any recorded rainfall excludes the race. "
        "When both apply, `wet_compound` is the primary reason; both measurements remain in the manifest. "
        "Load/schema errors are isolated per race.\n\nIncluded/excluded by year:\n\n"
        + markdown_table(overview) + "\n\nCounts by primary reason (each race counted once):\n\n"
        + markdown_table(counts) + "\n\nSource provenance:\n\n" + markdown_table(sources)
        + "\n\nManual overrides:\n\n" + ("No manual overrides were applied." if overrides.empty else markdown_table(overrides))
        + "\n\nRaces flagged for manual review (diagnostics only; no automatic override):\n\n"
        + markdown_table(reviews)
        + "\n\nThe user authorized 2025 ingestion for fixed Phase 2 preparation and quality counts only. "
        "No test prediction results, feature selection, tuning, or model evaluation were produced.\n"
    )
