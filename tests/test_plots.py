"""Render synthetic figures only under temporary output directories."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f1pace import cleaning, config, plots


def test_manifest_figure_renders_all_decision_reasons(isolated_config, phase2_sessions) -> None:
    """Render a labelled synthetic fig04 while preserving the input and closing it."""
    from f1pace import manifest
    frame = manifest.build_manifest([2024, 2025])
    frame.loc[0, "manual_override"] = "exclude"
    frame = manifest.validate_manifest(frame)
    original = frame.copy(deep=True)
    existing = plt.get_fignums()
    path = plots.plot_race_manifest(frame, synthetic=True)
    assert path.name == "fig04_race_manifest_summary.png"
    assert mpimg.imread(path).shape[0] >= 600
    assert_frame_equal(frame, original)
    assert plt.get_fignums() == existing


def test_degradation_figure_uses_training_targets_only(isolated_config, clean_stint, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even extreme test targets must not enter the plotted compound medians."""
    from f1pace import features
    training = features.build_features(clean_stint)
    test = training.assign(year=2025, race_id="2025_01", split="test", target_delta=9999.)
    combined = pd.concat([training, test], ignore_index=True)
    before = combined.copy(deep=True)
    captured = {}

    def inspect_figure(fig, filename, **kwargs):
        captured["x"] = fig.axes[0].lines[0].get_xdata().tolist()
        captured["y"] = fig.axes[0].lines[0].get_ydata().tolist()
        captured["title"] = fig.axes[0].get_title()
        captured["note"] = kwargs["note"]
        plt.close(fig)
        return config.FIG_DIR / filename

    monkeypatch.setattr(plots, "_save", inspect_figure)
    plots.plot_degradation_by_compound(combined, synthetic=True)
    assert captured["x"] == [4, 5, 6, 7, 8, 9, 10]
    assert captured["y"] == [4.] * 7
    assert "Synthetic" in captured["title"]
    assert "associative" in (captured["title"] + captured["note"]).lower()
    assert "not causal" in captured["note"].lower()
    assert_frame_equal(combined, before)


def test_degradation_figure_exports_png(isolated_config, clean_stint) -> None:
    """Verify the real renderer saves fig05 and closes its figure."""
    from f1pace import features
    existing = plt.get_fignums()
    path = plots.plot_degradation_by_compound(features.build_features(clean_stint), synthetic=True)
    assert path == config.FIG_DIR / "fig05_degradation_by_compound.png"
    assert mpimg.imread(path).shape[0] >= 600
    assert plt.get_fignums() == existing


def test_degradation_figure_rejects_test_season_mislabeled_train(isolated_config, clean_stint) -> None:
    """A wrong split label must not make test targets visible in fig05."""
    from f1pace import features
    wrong = features.build_features(clean_stint).assign(year=2025)
    with pytest.raises(ValueError, match="split"):
        plots.plot_degradation_by_compound(wrong, synthetic=True)


@pytest.mark.parametrize("number", [1, 2, 3])
def test_figure_exports_have_expected_names_and_render(
    isolated_config: Path, synthetic_laps: pd.DataFrame, number: int
) -> None:
    """Create readable PNGs without changing inputs or leaving figures open."""
    original = synthetic_laps.copy(deep=True)
    clean, audit = cleaning.clean_laps(synthetic_laps)
    existing = plt.get_fignums()
    if number == 1:
        result = plots.plot_laptime_vs_tyre_age(clean, "synthetic", synthetic=True)
        expected = "fig01_laptime_vs_tyre_age_synthetic.png"
    elif number == 2:
        result = plots.plot_laptime_vs_lap(synthetic_laps, clean, "synthetic", synthetic=True)
        expected = "fig02_laptime_vs_lap_synthetic.png"
    else:
        result = plots.plot_cleaning_audit(audit, "synthetic", synthetic=True)
        expected = "fig03_cleaning_audit_synthetic.png"
    assert result == config.FIG_DIR / expected
    pixels = mpimg.imread(result)
    assert pixels.shape[0] > 100 and pixels.shape[1] > 100
    assert plt.get_fignums() == existing
    assert_frame_equal(synthetic_laps, original)


@pytest.mark.parametrize("number", [1, 2])
def test_pace_figures_reject_empty_clean_data(isolated_config, synthetic_laps, number: int) -> None:
    """An empty cleaning result cannot silently become an exploration figure."""
    with pytest.raises(ValueError, match="No clean laps"):
        if number == 1:
            plots.plot_laptime_vs_tyre_age(synthetic_laps.iloc[:0], "synthetic", synthetic=True)
        else:
            plots.plot_laptime_vs_lap(synthetic_laps, synthetic_laps.iloc[:0], "synthetic", synthetic=True)
