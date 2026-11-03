"""Run every app page offline against the committed app_data/ files."""

import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="install the app extra: pip install -e .[dev,app]")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
APP_DATA = ROOT / "app_data"
ENTRY = APP_DIR / "streamlit_app.py"
PAGES = {"overview": "views/overview.py", "race_explorer": "views/race_explorer.py",
         "where_model_helps": "views/where_model_helps.py", "method_and_limits": "views/method_and_limits.py"}
SOURCE_NOTE = "Data: FastF1-derived TracingInsights archive"


@pytest.fixture(autouse=True)
def _app_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match streamlit run, which puts the entry point's folder on sys.path."""
    monkeypatch.syspath_prepend(str(APP_DIR))


def _open(page: str | None = None) -> AppTest:
    app = AppTest.from_file(str(ENTRY), default_timeout=60).run()
    assert not app.exception
    if page is not None:
        app.switch_page(PAGES[page]).run()
    return app


def _charts(app: AppTest) -> list[dict]:
    return [json.loads(element.proto.spec) for element in app.main if getattr(element, "type", "") == "vega_lite_chart"]


def _text(app: AppTest) -> str:
    return "\n".join(element.value for kind in ("title", "markdown", "caption", "subheader", "warning", "info")
                     for element in getattr(app, kind))


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(APP_DATA / name)


def _manifest() -> dict:
    return json.loads((APP_DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_without_exceptions(page: str) -> None:
    app = _open(page)
    assert not app.exception
    assert app.title


def test_default_page_is_overview() -> None:
    assert "F1 Race Pace Forecasting" in [title.value for title in _open().title]


def test_navigation_reaches_distinct_pages() -> None:
    titles = {_open(page).title[0].value for page in PAGES}
    assert len(titles) == len(PAGES)


def test_overview_cards_come_from_test_summary() -> None:
    app = _open("overview")
    summary = _table("test_summary.csv").set_index("predictor")
    manifest = _manifest()
    model, baseline = summary.loc[manifest["chosen_model"]], summary.loc[manifest["baseline"]]
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Model MAE"] == f"{model['mean_mae_s']:.3f} s"
    assert values["Baseline MAE"] == f"{baseline['mean_mae_s']:.3f} s"
    assert values["Improvement"] == f"{model['improvement_pct']:.2f}%"
    assert values["Races won / lost"] == f"{model['races_won_vs_baseline']} / {model['races_lost_vs_baseline']}"


def test_overview_has_full_predictor_table_summary_and_readme_link() -> None:
    app = _open("overview")
    assert len(app.dataframe[0].value) == 7
    summary_text = app.markdown[0].value
    assert summary_text.count(". ") + 1 == 3 and summary_text.endswith(".")
    assert "github.com/filan214/F1-tyre-degradation" in _text(app)


def test_race_explorer_covers_every_race() -> None:
    """Select all races and check the title, the driver list and the race MAE."""
    app = _open("race_explorer")
    results = _table("test_results_by_race.csv").set_index("race_id")
    frame = pd.read_parquet(APP_DATA / "test_predictions.parquet")
    manifest = _manifest()
    assert list(app.selectbox(key="race").options) == sorted(
        f"{event} ({circuit})" for event, circuit in frame[["event_name", "circuit"]].drop_duplicates().itertuples(index=False))
    names = frame.drop_duplicates("race_id").set_index("race_id")
    for race_id, row in names.iterrows():
        app.selectbox(key="race").select(f"{row['event_name']} ({row['circuit']})").run()
        assert not app.exception
        values = {metric.label: metric.value for metric in app.metric}
        assert values["Model race MAE"] == f"{results.loc[race_id, manifest['chosen_model']]:.3f} s"
        assert values["Baseline race MAE"] == f"{results.loc[race_id, manifest['baseline']]:.3f} s"
        expected = sorted(frame.loc[frame["race_id"].eq(race_id), "driver"].unique())
        assert [option.split(" ")[0] for option in app.selectbox(key="driver").options] == expected
        assert row["event_name"] in _charts(app)[0]["title"]["text"]


def test_race_explorer_handles_the_driver_with_fewest_laps() -> None:
    frame = pd.read_parquet(APP_DATA / "test_predictions.parquet")
    counts = frame.groupby(["race_id", "driver"]).size().sort_values()
    race_id, driver = counts.index[0]
    row = frame[frame["race_id"].eq(race_id)].iloc[0]
    app = _open("race_explorer")
    app.selectbox(key="race").select(f"{row['event_name']} ({row['circuit']})").run()
    option = next(option for option in app.selectbox(key="driver").options if option.startswith(driver))
    app.selectbox(key="driver").select(option).run()
    assert not app.exception


def test_race_explorer_notes_removed_laps_and_marks_stints() -> None:
    app = _open("race_explorer")
    text = _text(app).lower()
    assert "removed laps" in text and "not shown" in text
    assert "Stint" in json.dumps(_charts(app)[0])


def test_race_chart_breaks_lines_at_missing_laps() -> None:
    """Never draw a line across a removed lap, which would imply it was predicted."""
    import app_common

    laps = pd.DataFrame({"lap_number": [10, 11, 14, 15], "stint": [1, 1, 1, 2]})
    assert app_common.line_segments(laps).tolist() == [1, 1, 2, 3]


def test_where_model_helps_shows_every_table_and_all_losses() -> None:
    app = _open("where_model_helps")
    assert len(_charts(app)) == 5
    assert len(app.dataframe) == 5
    assert [len(table.value) for table in app.dataframe] == [5, 3, 2, 2, 18]
    text = _text(app)
    losers = _table("error_by_circuit.csv").query("improvement_s < 0")
    assert not losers.empty
    for circuit in losers["circuit"]:
        assert circuit in text
    for bucket in _table("error_by_tyre_age_bucket.csv").query("improvement_s < 0")["tyre_age_bucket"]:
        assert bucket in text


def test_where_model_helps_sorts_circuits_with_the_worst_loss_first() -> None:
    app = _open("where_model_helps")
    circuits = _table("error_by_circuit.csv").sort_values("improvement_s")
    assert app.dataframe[4].value.iloc[0, 0] == circuits.iloc[0]["circuit"]


@pytest.mark.parametrize("page", ["race_explorer", "where_model_helps"])
def test_every_chart_has_title_units_and_source_note(page: str) -> None:
    charts = _charts(_open(page))
    assert charts
    for chart in charts:
        assert chart["title"]["text"]
        assert chart["title"]["subtitle"] == SOURCE_NOTE
        assert "seconds" in json.dumps(chart)


@pytest.mark.parametrize("page", ["race_explorer", "where_model_helps"])
def test_source_note_has_an_explicit_colour(page: str) -> None:
    """The theme's default subtitle colour is unreadable on the dark background."""
    for chart in _charts(_open(page)):
        assert chart["title"]["subtitleColor"]


def test_gain_charts_leave_room_for_every_row() -> None:
    """A fixed height also has to hold the title, legend and axis, which collapsed short charts."""
    for chart in _charts(_open("where_model_helps")):
        assert chart["height"] == {"step": 28}


def test_method_page_has_framing_attribution_and_pinned_commits() -> None:
    app = _open("method_and_limits")
    text = _text(app)
    assert "held-out" in text and "not a live forecast" in text
    assert "not causal" in text.lower() or "descriptive" in text.lower()
    assert "TracingInsights" in text and "FastF1" in text
    assert "app_data/LICENSES" in text
    for details in _manifest()["source_commits"].values():
        assert f"https://github.com/{details['repository']}/tree/{details['commit']}" in text
    assert len(app.dataframe[0].value) == len(_table("cv_summary.csv"))


def _python_files() -> list[Path]:
    return sorted(APP_DIR.rglob("*.py"))


def test_app_has_no_forbidden_imports() -> None:
    forbidden = {"fastf1", "sklearn", "xgboost", "joblib", "pickle", "f1pace", "requests", "urllib", "httpx", "socket"}
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not {name.split(".")[0] for name in names} & forbidden, path


def test_app_source_has_no_hard_coded_metrics() -> None:
    """Fail on a long decimal, or on the row count, typed into app code."""
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\d\.\d{3,}", source), path
        assert not re.search(r"13[,_]?718", source), path


def test_app_uses_no_absolute_paths() -> None:
    for path in _python_files():
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/", path.read_text(encoding="utf-8")), path


def test_requirements_pin_only_streamlit_pandas_pyarrow_matching_pyproject() -> None:
    lines = (APP_DIR / "requirements.txt").read_text(encoding="utf-8").split()
    pins = dict(line.split("==") for line in lines)
    assert set(pins) == {"streamlit", "pandas", "pyarrow"}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = dict(item.split("==") for item in [*project["dependencies"], *project["optional-dependencies"]["app"]])
    assert all(declared[name] == version for name, version in pins.items())


def test_committed_exports_match_their_manifest() -> None:
    manifest = _manifest()
    assert sorted(path.name for path in APP_DATA.iterdir() if path.is_file()) == sorted([*manifest["files"], "manifest.json"])
    for name, details in manifest["files"].items():
        data = (APP_DATA / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == details["sha256"], name
        assert len(data) == details["bytes"], name
    assert sum(path.stat().st_size for path in APP_DATA.rglob("*") if path.is_file()) < 10_000_000


def test_licence_copies_match_the_pinned_archive_commits() -> None:
    """Keep each archive's licence byte for byte, and name its commit in the notice."""
    licences = APP_DATA / "LICENSES"
    expected = {
        "TracingInsights-2023-LICENSE.txt": "8eaea9798a53c0bb992e5f1a902f5725ca6e321ef7293f9a47e2dbdf79e91c13",
        "TracingInsights-2024-LICENSE.txt": "a350f1d49e6af623bcb0d8a18d4ed2862ca31c6c0b3808a266c31ecae477ee51",
        "TracingInsights-2025-LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((licences / name).read_bytes()).hexdigest() == digest, name
    notice = (licences / "NOTICE.md").read_text(encoding="utf-8")
    assert "Changes made to the source data" in notice
    for details in _manifest()["source_commits"].values():
        assert details["commit"] in notice


def test_committed_predictions_reconcile_to_saved_race_results() -> None:
    frame = pd.read_parquet(APP_DATA / "test_predictions.parquet")
    assert len(frame) == 13_718 and frame["race_id"].nunique() == 18
    assert frame["race_id"].str.startswith("2025_").all()
    predictions = [column for column in frame if column.startswith("pred_")]
    assert not frame[predictions].isna().any().any()
    saved = _table("test_results_by_race.csv").set_index("race_id")
    for column in predictions:
        recomputed = (frame[column] - frame["lap_time_s"]).abs().groupby(frame["race_id"]).mean()
        assert (recomputed.reindex(saved.index) - saved[column]).abs().max() <= 1e-12
