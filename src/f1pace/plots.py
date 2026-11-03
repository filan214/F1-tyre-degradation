"""Matplotlib exports for race exploration and manifest diagnostics."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from f1pace import config
from f1pace.cleaning import removal_reason

_logger = logging.getLogger(__name__)
_COMPOUND_COLOURS = {"SOFT": "#e53935", "MEDIUM": "#d4aa00", "HARD": "#808080"}


def plot_error_breakdown(table: pd.DataFrame, by: str, model: str, baseline: str, *, synthetic: bool = False) -> Path:
    """Plot equal-race subgroup MAE with explicit support and frozen predictors."""
    if by not in ('tyre_age_bucket', 'compound') or table.empty:
        raise ValueError('Expected nonempty tyre-age or compound breakdown')
    ordered = table.copy()
    if by == 'compound':
        ordered = ordered.set_index(by).reindex([c for c in _COMPOUND_COLOURS if c in set(table[by])]).reset_index()
    positions = list(range(len(ordered)))
    colours = ordered[by].map(_COMPOUND_COLOURS).tolist() if by == 'compound' else '#35785c'
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar([x - .2 for x in positions], ordered['baseline_mean_race_mae_s'], width=.4, color='#999999', label=baseline.removeprefix('pred_').upper() + ' baseline')
    ax.bar([x + .2 for x in positions], ordered['model_mean_race_mae_s'], width=.4, color=colours, label=model.removeprefix('pred_').upper() + ' model')
    labels = [f'{row[by]}\n{row["n_laps"]:,} laps / {row["n_races"]} races' for row in ordered.to_dict('records')]
    ax.set_xticks(positions, labels, fontsize=9)
    dimension = 'tyre age' if by == 'tyre_age_bucket' else 'compound'
    prefix = 'Synthetic test data — ' if synthetic else ''
    ax.set(title=prefix + f'2025 saved test errors by {dimension}', xlabel='Tyre age (laps, including prior sessions)' if by == 'tyre_age_bucket' else 'Dry compound', ylabel='Mean per-race absolute lap-time error (s)')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    filename = 'fig10_error_by_tyre_age.png' if by == 'tyre_age_bucket' else 'fig11_error_by_compound.png'
    return _save(fig, filename, synthetic=synthetic, source_note='Data: FastF1/Ergast via TracingInsights archive', note='Descriptive, not causal. Equal weight per represented race within each group; race composition can differ.')


def plot_test_comparison(by_race: pd.DataFrame, chosen: str, baseline: str, *, synthetic: bool = False) -> Path:
    """Compare the training-selected model and baseline on each held-out race."""
    ordered = by_race.sort_values('race_id')
    positions = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar([x - 0.2 for x in positions], ordered[baseline], width=0.4, color='#999999', label=f'{baseline.removeprefix("pred_").upper()}: training-selected baseline')
    ax.bar([x + 0.2 for x in positions], ordered[f'pred_{chosen}'], width=0.4, color='#35785c', label=f'{chosen.upper()}: training-selected model')
    ax.set_xticks(positions, ordered['race_id'], rotation=45, ha='right')
    prefix = 'Synthetic test data — ' if synthetic else ''
    ax.set(title=prefix + 'Final 2025 test: model versus baseline by race', xlabel='Held-out race (season_round)', ylabel='Mean absolute lap-time error (s)')
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return _save(fig, 'fig09_test_model_vs_baseline_by_race.png', synthetic=synthetic, source_note='Data: FastF1/Ergast via TracingInsights archive', note='Same eligible laps for both predictors; model and baseline selected before opening 2025.')


def plot_cv_comparison(by_race: pd.DataFrame, *, synthetic: bool = False) -> Path:
    """Compare baseline and model OOF MAEs across equally weighted training races."""
    columns = [name for name in by_race if name.startswith('pred_')]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([by_race[c].to_numpy() for c in columns], tick_labels=[c.removeprefix('pred_').upper() for c in columns], showmeans=True)
    prefix = 'Synthetic test data — ' if synthetic else ''
    ax.set(title=prefix + 'Training race holdout CV: models versus baseline', xlabel='Predictor', ylabel='Per-race mean absolute error (s)')
    ax.grid(axis='y', alpha=0.2)
    return _save(fig, 'fig07_cv_model_vs_baseline.png', synthetic=synthetic, source_note='Data: FastF1/Ergast via TracingInsights archive', note='Model predictions are out of fold. Parameters selected on these training folds; non-nested CV.')


def plot_cv_importance(table: pd.DataFrame, *, synthetic: bool = False) -> Path:
    """Show validation-fold permutation effects on MAE in seconds."""
    ordered = table.sort_values('mean_importance_s')
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(ordered['feature'], ordered['mean_importance_s'], xerr=ordered['std_importance_s'], color='#446a8a', capsize=2)
    ax.axvline(0, color='#777777', linewidth=0.8)
    prefix = 'Synthetic test data — ' if synthetic else ''
    ax.set(title=prefix + 'Selected model: validation-fold permutation importance', xlabel='Increase in MAE after permutation (s; mean ± SD)', ylabel='Original model input')
    ax.grid(axis='x', alpha=0.2)
    return _save(fig, 'fig08_feature_importance_cv.png', synthetic=synthetic, source_note='Data: FastF1/Ergast via TracingInsights archive', note='Held-out training folds only; five repeats per fold. SD spans folds/repeats, not a confidence interval.')


def plot_baseline_mae(by_race: pd.DataFrame, *, synthetic: bool = False) -> Path:
    """Save a box plot of training-race MAEs for the four baselines."""
    columns = ['pred_b1', 'pred_b2', 'pred_b3', 'pred_b4']
    if by_race.empty:
        raise ValueError('No baseline race errors to plot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([by_race[c].to_numpy() for c in columns], tick_labels=['B1: last lap', 'B2: mean 3', 'B3: mean 5', 'B4: stint trend'], showmeans=True)
    prefix = 'Synthetic test data — ' if synthetic else ''
    ax.set(title=prefix + 'Training race baseline errors (2023–2024)', xlabel='Baseline', ylabel='Per-race mean absolute error (s)')
    ax.grid(axis='y', alpha=0.2)
    return _save(fig, 'fig06_baseline_mae_train.png', synthetic=synthetic, source_note='Data: FastF1/Ergast via TracingInsights archive', note='Each race has equal weight. Identical eligible laps for all baselines.')


def _save(fig: Figure, filename: str, *, synthetic: bool, note: str = "", source_note: str = "Data: FastF1") -> Path:
    source = "Data: synthetic test fixture (FastF1 schema)" if synthetic else source_note
    fig.text(0.015, 0.015, source + (". " + note if note else ""), fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / filename
    try:
        fig.savefig(path, dpi=150, facecolor="white", metadata={"Description": source})
    finally:
        plt.close(fig)
    _logger.info("Saved %s", path)
    return path


def plot_laptime_vs_tyre_age(
    clean: pd.DataFrame, race_id: str, *, synthetic: bool = False, source_note: str = "Data: FastF1"
) -> Path:
    """Plot driver-stint traces and compound medians against tyre age."""
    if clean.empty:
        raise ValueError("No clean laps to plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (_, _, compound), stint in clean.groupby(["Driver", "Stint", "Compound"]):
        ordered = stint.sort_values("TyreLife")
        ax.plot(ordered["TyreLife"], ordered["lap_time_s"], color=_COMPOUND_COLOURS[compound], alpha=0.22, linewidth=0.7)
    for compound, colour in _COMPOUND_COLOURS.items():
        median = clean.loc[clean["Compound"].eq(compound)].groupby("TyreLife")["lap_time_s"].median()
        if not median.empty:
            ax.plot(median.index, median.to_numpy(), color=colour, linewidth=2.6, marker=".", label=f"{compound} median")
    prefix = "Synthetic test data — " if synthetic else ""
    ax.set(title=f"{prefix}{race_id}: clean lap time vs tyre age", xlabel="Tyre age (laps, including prior sessions)", ylabel="Lap time (s)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _save(fig, f"fig01_laptime_vs_tyre_age_{race_id}.png", synthetic=synthetic, source_note=source_note, note="Descriptive association; not a causal degradation estimate.")


def plot_laptime_vs_lap(
    raw: pd.DataFrame, clean: pd.DataFrame, race_id: str, *, synthetic: bool = False, source_note: str = "Data: FastF1"
) -> Path:
    """Show up to six drivers by stint, with labelled hollow removal markers."""
    if clean.empty:
        raise ValueError("No clean laps to plot")
    drivers = sorted(clean["Driver"].dropna().unique())[:6]
    if not drivers:
        raise ValueError("No clean laps with driver identifiers to plot")
    labelled = raw.assign(removal_reason=removal_reason(raw))
    fig, axes = plt.subplots((len(drivers) + 1) // 2, 2, figsize=(12, 3.6 * ((len(drivers) + 1) // 2)), squeeze=False)
    palette = plt.get_cmap("tab10")
    for ax, driver in zip(axes.flat, drivers):
        driver_clean = clean.loc[clean["Driver"].eq(driver)]
        for stint_number, stint in driver_clean.groupby("Stint"):
            ordered = stint.sort_values("LapNumber")
            colour = palette((int(stint_number) - 1) % 10)
            ax.plot(ordered["LapNumber"], ordered["lap_time_s"], color=colour, linewidth=1, marker=".", label=f"Stint {int(stint_number)}")
        removed = labelled.loc[labelled["Driver"].eq(driver) & labelled["removal_reason"].ne("kept")].dropna(subset=["LapNumber", "lap_time_s"])
        for _, lap in removed.iterrows():
            colour = palette((int(lap["Stint"]) - 1) % 10) if pd.notna(lap["Stint"]) else "#777777"
            ax.scatter(lap["LapNumber"], lap["lap_time_s"], facecolors="none", edgecolors=[colour], s=35, linewidths=1)
            ax.annotate(lap["removal_reason"], (lap["LapNumber"], lap["lap_time_s"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set(title=driver, xlabel="Race lap number (lap)", ylabel="Lap time (s)")
        ax.margins(x=0.06, y=0.15)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(drivers):]:
        ax.set_visible(False)
    prefix = "Synthetic test data — " if synthetic else ""
    fig.suptitle(f"{prefix}{race_id}: lap times, stints, and removed laps", fontsize=13)
    return _save(fig, f"fig02_laptime_vs_lap_{race_id}.png", synthetic=synthetic, source_note=source_note, note="Hollow markers: first removal rule. Missing lap times cannot be positioned.")


def plot_cleaning_audit(
    audit: pd.DataFrame, race_id: str, *, synthetic: bool = False, source_note: str = "Data: FastF1"
) -> Path:
    """Export the number of laps first removed by each cleaning rule."""
    rules = audit.loc[audit["rule_id"].ne("raw")]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = rules["rule_id"] + " — " + rules["rule_description"]
    bars = ax.barh(labels, rules["laps_removed"], color="#446a8a")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(rules["laps_removed"].max()) * 1.15, 1))
    prefix = "Synthetic test data — " if synthetic else ""
    ax.set(title=f"{prefix}{race_id}: cleaning audit", xlabel="Laps removed (count)", ylabel="First matching cleaning rule")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    return _save(fig, f"fig03_cleaning_audit_{race_id}.png", synthetic=synthetic, source_note=source_note)


def plot_race_manifest(
    frame: pd.DataFrame, *, synthetic: bool = False, source_note: str = "Data: FastF1/Ergast via TracingInsights archive"
) -> Path:
    """Plot included and excluded race counts per year with primary reasons."""
    from f1pace.manifest import validate_manifest

    checked = validate_manifest(frame)
    if checked.empty:
        raise ValueError("No races in manifest")
    reason = checked["exclude_reason"].where(~checked["include"], "included")
    counts = pd.crosstab(checked["year"], reason)
    colours = {"included": "#35785c", "wet_compound": "#487fa8", "rainfall": "#91bbd6", "load_error": "#c96049", "manual_override": "#a686b5"}
    labels = {"included": "Included", "wet_compound": "Excluded: wet tyres >5%", "rainfall": "Excluded: rainfall", "load_error": "Excluded: load error", "manual_override": "Excluded: manual override"}
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = pd.Series(0, index=counts.index)
    for category in colours:
        if category not in counts:
            continue
        values = counts[category]
        bars = ax.bar(counts.index.astype(str), values, bottom=bottom, label=labels[category], color=colours[category], width=0.55)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in values], label_type="center", color="white", fontsize=11)
        bottom += values
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.4, f"{total} scheduled", ha="center", fontsize=10)
    prefix = "Synthetic test data — " if synthetic else ""
    ax.set(title=f"{prefix}Race manifest: inclusion by season", xlabel="Season (year)", ylabel="Grand Prix races (count)", ylim=(0, bottom.max() * 1.25))
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "fig04_race_manifest_summary.png", synthetic=synthetic, source_note=source_note, note="Primary reason per excluded race; fixed preprocessing only.")


def plot_degradation_by_compound(
    frame: pd.DataFrame, *, synthetic: bool = False, source_note: str = "Data: FastF1/Ergast via TracingInsights archive"
) -> Path:
    """Plot training-only median target delta by tyre age as an association."""
    from f1pace.manifest import split_for_year

    required = {"year", "split", "compound", "tyre_age", "target_delta"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Missing degradation figure columns: " + ", ".join(missing))
    if not frame["split"].eq(frame["year"].map(split_for_year)).all():
        raise ValueError("Figure split does not match configured seasons")
    training = frame.loc[frame["split"].eq("train") & frame["year"].isin(config.TRAIN_YEARS)]
    if training.empty:
        raise ValueError("No eligible training rows for fig05")
    fig, ax = plt.subplots(figsize=(10, 6))
    for compound, colour in _COMPOUND_COLOURS.items():
        median = training.loc[training["compound"].eq(compound)].groupby("tyre_age")["target_delta"].median()
        if not median.empty:
            ax.plot(median.index, median.to_numpy(), color=colour, linewidth=1.8, marker=".", label=compound)
    ax.axhline(0, linewidth=0.8, color="#777777", linestyle="--")
    prefix = "Synthetic test data — " if synthetic else ""
    years = ", ".join(str(year) for year in sorted(training["year"].unique()))
    ax.set(title=f"{prefix}Tyre age and pace change: associative view\nTraining seasons {years}", xlabel="Tyre age (laps, including prior sessions)", ylabel="Median lap time minus prior-three-lap mean (s)")
    ax.grid(alpha=0.2)
    ax.legend(title="Dry compound", frameon=False)
    return _save(fig, "fig05_degradation_by_compound.png", synthetic=synthetic, source_note=source_note, note="Training rows only. Associative, not causal; fuel, traffic and track evolution remain mixed in.")
