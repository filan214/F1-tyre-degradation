"""Explore one dry training race and save its cleaning audit and figures."""

import argparse
from collections.abc import Sequence
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from f1pace import archive, cleaning, config, data, plots

_logger = logging.getLogger(__name__)


def _command() -> str:
    return "& " + subprocess.list2cmdline([sys.executable, *sys.argv])


def _append_log(entry: str, *, deviation: str | None = None) -> None:
    path = config.ROOT / "EXPERIMENT_LOG.md"
    document = path.read_text(encoding="utf-8")
    marker = "\n---\n\n## Key results summary"
    if marker not in document:
        raise ValueError("EXPERIMENT_LOG.md is missing the Entries section boundary")
    document = document.replace(marker, "\n\n" + entry.rstrip() + "\n" + marker, 1)
    if deviation is not None:
        marker = "\n---\n\n## Error analysis findings"
        if marker not in document:
            raise ValueError("EXPERIMENT_LOG.md is missing the Deviations section boundary")
        document = document.replace(marker, "\n\n" + deviation.rstrip() + "\n" + marker, 1)
    path.write_text(document, encoding="utf-8")


def _audit_markdown(audit: pd.DataFrame) -> str:
    lines = ["| Rule | Description | Removed | Remaining |", "|---|---|---|---|"]
    for row in audit.itertuples(index=False):
        lines.append(f"| {row.rule_id} | {row.rule_description} | {row.laps_removed} | {row.laps_remaining} |")
    return "\n".join(lines)


def run_exploration(
    year: int, round_number: int, *, synthetic: bool = False, source: str = "auto"
) -> list[Path]:
    """Coordinate one race's diagnostics, cleaning, exports, and measured log entry."""
    if year not in config.TRAIN_YEARS:
        raise ValueError("Phase 1 may explore only training seasons; the test season is locked")
    config.ensure_dirs()
    data.enable_cache()
    identifier = data.race_id(year, round_number)
    if source not in {"auto", "archive", "fastf1"}:
        raise ValueError(f"Unknown data source: {source}")
    use_archive = source == "archive" or (source == "auto" and (year, round_number) in archive.SUPPORTED_RACES)
    if use_archive:
        session = archive.load_archived_race(year, round_number)
        event_name = str(session.event["EventName"])
        source_note = "Data: FastF1/Ergast via TracingInsights archive"
        print(f"Source: free TracingInsights archive, commit {session.provenance['commit']}")
    else:
        schedule = data.get_race_schedule(year)
        selected = schedule.loc[schedule["round"].eq(round_number)]
        if len(selected) != 1:
            raise ValueError(f"Round {round_number} was not uniquely found in the {year} race schedule")
        event_name = str(selected.iloc[0]["event_name"])
        session = data.load_race(year, round_number)
        source_note = "Data: FastF1"
    print(f"Selected training race: {identifier} — {event_name}")
    print("Lap columns:", list(session.laps.columns))
    raw = data.laps_to_frame(session)
    data.validate_exploration_session(session)
    print("Rainfall counts:")
    print(session.weather_data["Rainfall"].value_counts(dropna=False).to_string())
    print("Raw lap TrackStatus counts:")
    print(raw["TrackStatus"].value_counts(dropna=False).to_string())
    raw = data.attach_weather(raw, session.weather_data)
    clean, audit = cleaning.clean_laps(raw)
    cleaning.validate_clean_laps(clean)
    print("Cleaning audit:")
    print(audit.to_string(index=False))

    audit_path = config.TABLE_DIR / f"audit_{identifier}.csv"
    audit.to_csv(audit_path, index=False)
    outputs = [
        audit_path,
        plots.plot_laptime_vs_tyre_age(clean, identifier, synthetic=synthetic, source_note=source_note),
        plots.plot_laptime_vs_lap(raw, clean, identifier, synthetic=synthetic, source_note=source_note),
        plots.plot_cleaning_audit(audit, identifier, synthetic=synthetic, source_note=source_note),
    ]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    title = "Synthetic verification only" if synthetic else "One-race exploration"
    data_source = "synthetic fixture only" if synthetic else "FastF1-derived TracingInsights archive (training split)" if use_archive else "FastF1 race timing (training split)"
    source_details = ""
    if use_archive:
        provenance = session.provenance
        source_details = (
            f"Source: [TracingInsights/2024]({archive.SOURCE_URL}), commit `{provenance['commit']}`.\n\n"
            "Raw inputs are verified by SHA-256 and cached under `data/raw/tracinginsights/2024_01/`.\n\n"
            "| Source file | SHA-256 |\n|---|---|\n"
            + "\n".join(f"| [{item['name']}]({item['url']}) | `{item['sha256']}` |" for item in provenance["files"])
            + "\n\nArchive notes: publisher lap times/positions include official Ergast/Jolpica corrections; durations have millisecond export precision. "
            "Per-lap archive weather and mini-sector additions are discarded; separate weather samples are joined backward at lap start. "
            "The status inventory is derived from lap codes and race-control red flags, not an original status timeline.\n\n"
        )
    counts = json.dumps({str(key): int(value) for key, value in raw["TrackStatus"].value_counts(dropna=False).items()})
    largest = audit.iloc[1:].loc[audit.iloc[1:]["laps_removed"].idxmax()]
    entry = (
        f"### {timestamp[:10]} — Phase 1 — {title}: {identifier}\n\n"
        f"Timestamp: {timestamp}\n\nCommand: `{_command()}`\n\n"
        f"Data: {data_source}; {event_name}; {len(raw)} raw laps, {len(clean)} clean laps.\n\n"
        + source_details
        +
        f"Results:\n\n- Dry/no-red-flag checks passed.\n- Raw lap TrackStatus counts: `{counts}`.\n\n"
        + _audit_markdown(audit)
        + "\n\nSaved outputs:\n\n"
        + "\n".join(f"- `{path.relative_to(config.ROOT).as_posix()}`" for path in outputs)
        + f"\n\nObservations:\n\n- {len(clean) / len(raw):.1%} of raw laps survived the ordered rules; each removed lap is counted once.\n"
        + f"- Largest first-removal count: {largest['rule_id']} ({largest['rule_description']}), {largest['laps_removed']} laps.\n"
        + "- Tyre-age curves are descriptive: fuel burn, traffic, and other effects remain mixed with tyre ageing. No model or prediction metric was produced.\n\n"
        + "Next:\n\n- Stop after Phase 1 and review the saved figures and audit before starting Phase 2.\n"
    )
    _append_log(entry)
    for path in outputs:
        print(f"Saved: {path}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the training race selection and run Phase 1 with failure logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=config.TRAIN_YEARS, required=True)
    parser.add_argument("--round", type=int, dest="round_number", required=True)
    parser.add_argument("--source", choices=["auto", "fastf1", "archive"], default="auto", help="auto uses the verified free archive for Bahrain 2024")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    logging.getLogger("fastf1").propagate = False
    data.fastf1.set_log_level(logging.INFO)
    config.ensure_dirs()
    try:
        run_exploration(args.year, args.round_number, source=args.source)
    except Exception as exc:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        message = f"{type(exc).__name__}: {exc}"
        entry = (
            f"### {timestamp[:10]} — Phase 1 — exploration attempt failed: {data.race_id(args.year, args.round_number)}\n\n"
            f"Timestamp: {timestamp}\n\nCommand: `{_command()}`\n\n"
            f"Results: {message}\n\n"
            "No complete real-data exploration result was produced. The Phase 1 real-data DoD remains unmet; no audit counts or figures are inferred from this failure.\n\n"
            "Next: resolve the reported error, then rerun the same command. Do not start Phase 2.\n"
        )
        deviation = None
        if isinstance(exc, data.MissingLapColumnsError):
            deviation = (
                f"### {timestamp[:10]} — Phase 1 — FastF1 schema mismatch\n\n"
                f"Installed FastF1: {data.fastf1.__version__}. {message}\n\n"
                "Loading stopped as required by CLAUDE.md. No column was silently renamed or substituted.\n"
            )
        _append_log(entry, deviation=deviation)
        _logger.error("Phase 1 failed: %s", message)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
