"""Error breakdown page, with the model's losses stated before its wins."""

import streamlit as st

from app_common import gain_chart, label, load_manifest, load_table

SECTIONS = [
    ("error_by_tyre_age_bucket.csv", "tyre_age_bucket", "Tyre age (laps)",
     "These groups describe performance by age, not the separate effect of the tyre-age feature."),
    ("error_by_compound.csv", "compound", "Compound",
     "Compound groups contain different races and drivers, so these differences do not establish tyre performance differences."),
    ("error_by_stint_phase.csv", "stint_phase", "Stint phase",
     "Early means fewer than 10 earlier clean laps in the stint, and late means at least 10. Removed laps are not counted, so this is not physical stint age or closeness to a pit stop."),
    ("error_by_slope_group.csv", "slope_group", "Prior stint trend",
     "Low means a stint slope of 0 s/lap or less, and high means above 0 s/lap. High describes rising earlier lap times, not proven degradation."),
    ("error_by_circuit.csv", "circuit", "Circuit",
     "Each circuit is one race, so a single result here carries no uncertainty estimate. Sorted with the worst result first."),
]

manifest = load_manifest()
model_name, baseline_name = label(manifest["chosen_model"]), label(manifest["baseline"])
tables = {name: load_table(name) for name, *_ in SECTIONS}
tables["error_by_circuit.csv"] = tables["error_by_circuit.csv"].sort_values("improvement_s", kind="stable")

st.title("Where the model helps")
st.markdown(f"Gain is {baseline_name} error minus {model_name} error, so a positive gain means the model is better. Errors are mean per-race MAE in seconds, with each race in a group counted equally.")


def _describe_losses(table, group: str, unit: str = "") -> list[str]:
    losses = table[table["improvement_s"] < 0]
    return [f"- **{row[group]}{unit}**: model {row['model_mean_race_mae_s']:.3f} s, baseline {row['baseline_mean_race_mae_s']:.3f} s (gain {row['improvement_s']:+.3f} s), "
            f"{row['races_lost']} of {row['n_races']} races lost"
            for _, row in losses.iterrows()]


circuits = tables["error_by_circuit.csv"]
young = tables["error_by_tyre_age_bucket.csv"]
st.subheader("Where the model loses")
st.markdown("Tyre-age groups the model loses:\n\n" + ("\n".join(_describe_losses(young, "tyre_age_bucket", " laps old")) or "- None"))
st.markdown(f"Circuits the model loses ({(circuits['improvement_s'] < 0).sum()} of {len(circuits)}):\n\n" + ("\n".join(_describe_losses(circuits, "circuit")) or "- None"))

columns = {
    "n_laps": st.column_config.NumberColumn("Laps"),
    "n_races": st.column_config.NumberColumn("Races"),
    "model_mean_race_mae_s": st.column_config.NumberColumn("Model MAE (s)", format="%.4f"),
    "baseline_mean_race_mae_s": st.column_config.NumberColumn("Baseline MAE (s)", format="%.4f"),
    "improvement_s": st.column_config.NumberColumn("Gain (s)", format="%+.4f"),
    "improvement_pct": st.column_config.NumberColumn("Gain (%)", format="%+.2f"),
    "races_won": st.column_config.NumberColumn("Won"),
    "races_lost": st.column_config.NumberColumn("Lost"),
    "races_tied": st.column_config.NumberColumn("Tied"),
    "model_pooled_mae_s": st.column_config.NumberColumn("Model pooled (s)", format="%.4f"),
    "baseline_pooled_mae_s": st.column_config.NumberColumn("Baseline pooled (s)", format="%.4f"),
}
for name, group, heading, note in SECTIONS:
    table = tables[name]
    st.subheader(heading)
    st.altair_chart(gain_chart(table, group, f"Model gain by {heading.lower()}"), width="stretch")
    st.dataframe(table, hide_index=True, column_order=[group, *columns], column_config={group: heading, **columns})
    st.caption(note)

st.caption("Won, lost and tied count races against the baseline. Pooled columns weight every lap equally instead of every race. These results are descriptive, not causal. They come from an already opened test set and no change to the model follows from them. Groups differ in race mix and size, and no confidence intervals are given.")
