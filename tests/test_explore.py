"""Isolated exploration orchestration; no real race downloads or results."""

import importlib.util
import io
import logging
from pathlib import Path
from types import ModuleType

import fastf1
import pandas as pd
import pytest

from f1pace import archive, config


@pytest.fixture
def explore_module(isolated_config: Path) -> ModuleType:
    """Import the numbered CLI and prepare an isolated append-only log."""
    log = "# Synthetic verification log\n\n## Entries\n\n---\n\n## Key results summary\n\n---\n\n## Deviations from IMPLEMENTATION_GUIDE.md\n\n---\n\n## Error analysis findings (Phase 7)\n"
    (config.ROOT / "EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")
    path = Path(__file__).resolve().parents[1] / "scripts" / "01_explore_one_race.py"
    spec = importlib.util.spec_from_file_location("explore_one_race", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network_replacements(session, monkeypatch: pytest.MonkeyPatch) -> None:
    session.load = lambda **kwargs: None
    schedule = pd.DataFrame({"RoundNumber": [1], "EventName": ["Synthetic Grand Prix"], "Location": ["Synthetic circuit"], "EventFormat": ["conventional"]})
    monkeypatch.setattr(fastf1, "get_event_schedule", lambda *args, **kwargs: schedule)
    monkeypatch.setattr(fastf1, "get_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)


def test_locked_year_is_rejected_before_download(explore_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 1 must never open the 2025 test season."""
    def forbidden(*args, **kwargs):
        raise AssertionError("A locked year reached the network")
    monkeypatch.setattr(fastf1, "get_event_schedule", forbidden)
    with pytest.raises(SystemExit) as exc:
        explore_module.main(["--year", "2025", "--round", "1"])
    assert exc.value.code == 2


def test_run_exploration_exports_and_logs_synthetic_verification(
    explore_module, synthetic_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real transformations/exporters with a fake network source, marked synthetic."""
    _network_replacements(synthetic_session, monkeypatch)
    outputs = explore_module.run_exploration(2024, 1, synthetic=True, source="fastf1")
    assert len(outputs) == 4
    assert all(path.is_file() for path in outputs)
    audit = pd.read_csv(config.TABLE_DIR / "audit_2024_01.csv")
    assert audit.iloc[0]["laps_remaining"] == 14
    assert audit.iloc[-1]["laps_remaining"] == 4
    log = (config.ROOT / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert "Synthetic verification only" in log
    assert "synthetic fixture" in log
    assert "| C10 |" in log
    assert log.startswith("# Synthetic verification log")


def test_failed_download_is_logged_without_results(explore_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Record a failed real-data attempt without inventing audit counts or figures."""
    def unavailable(*args, **kwargs):
        raise RuntimeError("schedule unavailable")
    monkeypatch.setattr(fastf1, "get_event_schedule", unavailable)
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)
    with pytest.raises(SystemExit) as exc:
        explore_module.main(["--year", "2024", "--round", "1", "--source", "fastf1"])
    assert exc.value.code == 1
    log = (config.ROOT / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert "schedule unavailable" in log
    assert "No complete real-data exploration result" in log
    assert not list(config.TABLE_DIR.glob("*.csv"))
    assert not list(config.FIG_DIR.glob("*.png"))


def test_missing_source_column_is_logged_as_deviation(explore_module, synthetic_session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop and report a missing required FastF1 column in the deviation section."""
    synthetic_session.laps = synthetic_session.laps.drop(columns="Deleted")
    _network_replacements(synthetic_session, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        explore_module.main(["--year", "2024", "--round", "1", "--source", "fastf1"])
    assert exc.value.code == 1
    log = (config.ROOT / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    deviation = log.split("## Deviations from IMPLEMENTATION_GUIDE.md")[1]
    assert "Deleted" in deviation
    assert not list(config.TABLE_DIR.glob("*.csv"))


def test_fastf1_console_messages_are_not_duplicated_or_debug_spammed(
    explore_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep FastF1's INFO handler without propagating DEBUG tracebacks to root."""
    fastf1_logger = logging.getLogger("fastf1")
    monkeypatch.setattr(fastf1_logger, "propagate", True)
    console_output = io.StringIO()
    for handler in fastf1_logger.handlers:
        if type(handler) is logging.StreamHandler:
            monkeypatch.setattr(handler, "stream", console_output)
    root_output = io.StringIO()
    root_handler = logging.StreamHandler(root_output)
    logging.getLogger().addHandler(root_handler)

    def emit_messages(year: int, round_number: int, *, source: str = "auto") -> list[Path]:
        logger = fastf1_logger.getChild("phase1_logging_regression")
        logger.info("phase1_console_probe")
        logger.debug("phase1_hidden_debug")
        return []

    monkeypatch.setattr(explore_module, "run_exploration", emit_messages)
    try:
        explore_module.main(["--year", "2024", "--round", "1"])
        assert console_output.getvalue().count("phase1_console_probe") == 1
        assert "phase1_hidden_debug" not in console_output.getvalue()
        assert root_output.getvalue() == ""
    finally:
        logging.getLogger().removeHandler(root_handler)


def test_auto_source_uses_free_archive_without_protected_requests(
    explore_module, synthetic_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original Bahrain command must succeed without touching F1's gated APIs."""
    def forbidden(*args, **kwargs):
        pytest.fail("The verified free source must not call protected FastF1 endpoints")

    synthetic_session.provenance = {"source": "synthetic fixture", "commit": "synthetic-verification", "files": []}
    monkeypatch.setattr(archive, "load_archived_race", lambda *args: synthetic_session)
    monkeypatch.setattr(fastf1, "get_event_schedule", forbidden)
    monkeypatch.setattr(fastf1, "get_session", forbidden)
    outputs = explore_module.run_exploration(2024, 1, synthetic=True)
    assert all(path.is_file() for path in outputs)
    log = (config.ROOT / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert "synthetic-verification" in log
    assert "TracingInsights" in log
