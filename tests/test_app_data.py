"""Export assertions for Phase 9, checked on a small synthetic fixture."""

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from f1pace import app_data

RACES = {"2025_01": ("First Grand Prix", "Alpha"), "2025_02": ("Second Grand Prix", "Beta")}
PREDICTORS = ["pred_b1", "pred_b2", "pred_b3", "pred_b4", "pred_ridge", "pred_rf", "pred_xgb"]


def _predictions() -> pd.DataFrame:
    rows = []
    for race_id, (_, circuit) in RACES.items():
        for driver in ("AAA", "BBB"):
            for lap in range(10, 13):
                rows.append({
                    "race_id": race_id, "year": 2025, "round": int(race_id[-2:]), "circuit": circuit,
                    "driver": driver, "team": "Team", "lap_number": lap, "stint": 1, "lag1": 90.0,
                    "compound": "HARD", "tyre_age": float(lap), "lap_time_s": 90.0 + lap / 100, "split": "test",
                    "data_source": "TracingInsights/2025", "source_commit": "c2025",
                    **{name: 90.0 + index / 10 for index, name in enumerate(PREDICTORS)},
                })
    return pd.DataFrame(rows)


def _by_race(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.groupby("race_id").size().rename("n_laps").reset_index()
    for name in PREDICTORS:
        table[name] = (frame[name] - frame["lap_time_s"]).abs().groupby(frame["race_id"]).mean().to_numpy()
    return table


def _race_manifest() -> pd.DataFrame:
    rows = [{"race_id": race_id, "year": 2025, "event_name": event, "circuit": circuit, "include": True,
             "data_source": "TracingInsights/2025", "source_commit": "c2025"}
            for race_id, (event, circuit) in RACES.items()]
    rows.append({"race_id": "2023_01", "year": 2023, "event_name": "Old", "circuit": "Old", "include": True,
                 "data_source": "TracingInsights/2023", "source_commit": "c2023"})
    rows.append({"race_id": "2024_01", "year": 2024, "event_name": "Old", "circuit": "Old", "include": True,
                 "data_source": "TracingInsights/2024", "source_commit": "c2024"})
    return pd.DataFrame(rows)


def _fixture(root: Path, frame: pd.DataFrame | None = None) -> dict:
    """Write synthetic inputs and return the keyword arguments for an export."""
    frame = _predictions() if frame is None else frame
    tables = root / "tables"
    tables.mkdir(parents=True)
    predictions = root / "test_predictions.parquet"
    frame.to_parquet(predictions, index=False)
    clean = frame[frame["split"].eq("test")]
    _by_race(clean).to_csv(tables / "test_results_by_race.csv", index=False)
    pd.DataFrame({"predictor": PREDICTORS, "mean_mae_s": 0.1, "comparison_baseline": "pred_b1"}).to_csv(tables / "test_summary.csv", index=False)
    pd.DataFrame({"predictor": ["pred_rf"], "mean_mae_s": [0.1]}).to_csv(tables / "cv_summary.csv", index=False)
    for name in ("tyre_age_bucket", "compound", "stint_phase", "slope_group", "circuit"):
        pd.DataFrame({name: ["x"], "model": ["pred_rf"], "baseline": ["pred_b1"], "improvement_s": [0.1]}).to_csv(tables / f"error_by_{name}.csv", index=False)
    race_manifest = tables / "race_manifest.csv"
    _race_manifest().to_csv(race_manifest, index=False)
    return {"predictions_path": predictions, "table_dir": tables, "race_manifest_path": race_manifest,
            "out_dir": root / "out", "expected_rows": 12, "expected_races": 2}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_writes_the_contract(tmp_path: Path) -> None:
    """Keep required columns and copy tables byte for byte, with a matching manifest."""
    arguments = _fixture(tmp_path)
    manifest = app_data.export_app_data(**arguments)
    out = arguments["out_dir"]
    frame = pd.read_parquet(out / "test_predictions.parquet")
    assert list(frame.columns) == [*app_data.CORE_COLUMNS, *app_data.OPTIONAL_PREDICTIONS]
    assert frame.loc[frame["race_id"].eq("2025_02"), "event_name"].eq("Second Grand Prix").all()
    assert "lag1" not in frame.columns and "source_commit" not in frame.columns
    for name in app_data.TABLES:
        assert (out / name).read_bytes() == (arguments["table_dir"] / name).read_bytes()
    assert (out / "manifest.json").is_file()
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert set(manifest["files"]) == {"test_predictions.parquet", *app_data.TABLES}
    for name, details in manifest["files"].items():
        assert details["sha256"] == _sha(out / name)
        assert details["bytes"] == (out / name).stat().st_size
    assert manifest["files"]["test_predictions.parquet"]["rows"] == 12
    assert manifest["n_races"] == 2
    assert manifest["chosen_model"] == "pred_rf" and manifest["baseline"] == "pred_b1"
    assert manifest["exported_at"]
    assert {key: value["commit"] for key, value in manifest["source_commits"].items()} == {"2023": "c2023", "2024": "c2024", "2025": "c2025"}
    assert manifest["reconciliation"]["max_abs_diff"] <= 1e-12


def test_export_leaves_inputs_unchanged(tmp_path: Path) -> None:
    """Treat every Phase 0-8 input as read-only."""
    arguments = _fixture(tmp_path)
    inputs = [arguments["predictions_path"], *sorted(arguments["table_dir"].iterdir())]
    before = {path: _sha(path) for path in inputs}
    app_data.export_app_data(**arguments)
    assert {path: _sha(path) for path in inputs} == before


def test_rejects_wrong_row_count(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    with pytest.raises(ValueError, match="rows"):
        app_data.export_app_data(**{**arguments, "expected_rows": 13})


def test_rejects_wrong_race_count(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    with pytest.raises(ValueError, match="races"):
        app_data.export_app_data(**{**arguments, "expected_races": 3})


@pytest.mark.parametrize("change", [{"year": 2024}, {"split": "train"}, {"race_id": "2024_01"}])
def test_rejects_training_rows(tmp_path: Path, change: dict) -> None:
    """Catch training-season rows through the year, the split or the race ID."""
    frame = _predictions()
    for column, value in change.items():
        frame.loc[frame.index[0], column] = value
    arguments = _fixture(tmp_path, frame)
    with pytest.raises(ValueError, match="2025|test"):
        app_data.export_app_data(**arguments)


@pytest.mark.parametrize("column", ["pred_rf", "pred_b1", "pred_xgb"])
def test_rejects_missing_prediction(tmp_path: Path, column: str) -> None:
    frame = _predictions()
    frame.loc[frame.index[3], column] = float("nan")
    arguments = _fixture(tmp_path)
    frame.to_parquet(arguments["predictions_path"], index=False)
    with pytest.raises(ValueError, match="[Mm]issing prediction"):
        app_data.export_app_data(**arguments)


def test_rejects_mae_mismatch_above_tolerance(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    table = pd.read_csv(arguments["table_dir"] / "test_results_by_race.csv")
    table.loc[0, "pred_rf"] += 1e-9
    table.to_csv(arguments["table_dir"] / "test_results_by_race.csv", index=False)
    with pytest.raises(ValueError, match="MAE"):
        app_data.export_app_data(**arguments)


def test_accepts_mae_difference_inside_tolerance(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    table = pd.read_csv(arguments["table_dir"] / "test_results_by_race.csv")
    table.loc[0, "pred_rf"] += 5e-13
    table.to_csv(arguments["table_dir"] / "test_results_by_race.csv", index=False)
    assert app_data.export_app_data(**arguments)["reconciliation"]["atol"] == 1e-12


def test_rejects_missing_source_column(tmp_path: Path) -> None:
    frame = _predictions().drop(columns="compound")
    arguments = _fixture(tmp_path, frame)
    with pytest.raises(ValueError, match="compound"):
        app_data.export_app_data(**arguments)


def test_rejects_race_without_event_name(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    table = _race_manifest().iloc[1:]
    table.to_csv(arguments["race_manifest_path"], index=False)
    with pytest.raises(ValueError, match="event_name"):
        app_data.export_app_data(**arguments)


def test_rejects_circuit_disagreement(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    table = _race_manifest()
    table.loc[0, "circuit"] = "Elsewhere"
    table.to_csv(arguments["race_manifest_path"], index=False)
    with pytest.raises(ValueError, match="circuit"):
        app_data.export_app_data(**arguments)


def test_rejects_missing_table(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    (arguments["table_dir"] / "error_by_circuit.csv").unlink()
    with pytest.raises(FileNotFoundError):
        app_data.export_app_data(**arguments)


def test_rejects_inconsistent_model_labels(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    pd.DataFrame({"compound": ["x"], "model": ["pred_xgb"], "baseline": ["pred_b1"]}).to_csv(arguments["table_dir"] / "error_by_compound.csv", index=False)
    with pytest.raises(ValueError, match="model"):
        app_data.export_app_data(**arguments)


def test_rejects_total_size_over_limit(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    with pytest.raises(ValueError, match="size|bytes"):
        app_data.export_app_data(**arguments, max_total_bytes=10)


def test_drops_optional_predictions_when_over_soft_limit(tmp_path: Path) -> None:
    """Keep only the core columns when the fuller export would pass the soft limit."""
    arguments = _fixture(tmp_path)
    manifest = app_data.export_app_data(**arguments, optional_limit_bytes=10)
    frame = pd.read_parquet(arguments["out_dir"] / "test_predictions.parquet")
    assert list(frame.columns) == list(app_data.CORE_COLUMNS)
    assert manifest["optional_predictions_included"] is False


def test_failed_validation_writes_nothing(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    with pytest.raises(ValueError):
        app_data.export_app_data(**{**arguments, "expected_rows": 13})
    assert not arguments["out_dir"].exists()


def test_export_module_cannot_load_models() -> None:
    """Guard the Phase 9 exception: this module never loads or fits a model."""
    tree = ast.parse(Path(app_data.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"joblib", "pickle", "sklearn", "xgboost", "fastf1"}
