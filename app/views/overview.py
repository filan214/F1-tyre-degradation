"""Overview page: summary, headline numbers and the full predictor table."""

import streamlit as st

from app_common import README_URL, label, load_manifest, load_table

manifest = load_manifest()
summary = load_table("test_summary.csv")
rows = summary.set_index("predictor")
model = rows.loc[manifest["chosen_model"]]
baseline = rows.loc[manifest["baseline"]]
n_laps = manifest["files"]["test_predictions.parquet"]["rows"]
n_races = manifest["n_races"]
verdict = "beat" if model["improvement_s"] > 0 else "did not beat"

st.title("F1 Race Pace Forecasting")
st.markdown(
    "This project predicts a driver's next lap time in dry, green-flag race conditions using only what is known before that lap starts. "
    f"Models were tuned and selected on 2023–2024 races, then scored on {n_laps:,} eligible laps from {n_races} dry 2025 races held out from all of that work. "
    f"The selected model, {label(manifest['chosen_model'])}, {verdict} the {label(manifest['baseline'])} baseline by "
    f"{model['improvement_pct']:.2f}% on mean per-race error, with wins and losses both shown here."
)

model_column, baseline_column, improvement_column, races_column = st.columns(4)
model_column.metric("Model MAE", f"{model['mean_mae_s']:.3f} s", help=f"{label(manifest['chosen_model'])}: mean of the per-race mean absolute errors.")
baseline_column.metric("Baseline MAE", f"{baseline['mean_mae_s']:.3f} s", help=f"{label(manifest['baseline'])}: the best baseline on the training races.")
improvement_column.metric("Improvement", f"{model['improvement_pct']:.2f}%", help=f"{model['improvement_s']:.4f} seconds per lap on mean per-race MAE.")
races_column.metric("Races won / lost", f"{model['races_won_vs_baseline']} / {model['races_lost_vs_baseline']}", help=f"Out of {n_races} held-out races, against the baseline.")

st.subheader("All seven predictors on the same 2025 laps")
table = summary.assign(predictor=summary["predictor"].map(label))
columns = {
    "predictor": st.column_config.TextColumn("Predictor"),
    "mean_mae_s": st.column_config.NumberColumn("Mean MAE (s)", format="%.4f"),
    "median_mae_s": st.column_config.NumberColumn("Median MAE (s)", format="%.4f"),
    "std_mae_s": st.column_config.NumberColumn("SD (s)", format="%.4f"),
    "improvement_s": st.column_config.NumberColumn("Gain (s)", format="%+.4f"),
    "improvement_pct": st.column_config.NumberColumn("Gain (%)", format="%+.2f"),
    "races_won_vs_baseline": st.column_config.NumberColumn("Races won"),
    "races_lost_vs_baseline": st.column_config.NumberColumn("Races lost"),
    "races_tied_vs_baseline": st.column_config.NumberColumn("Races tied"),
}
st.dataframe(table, hide_index=True, column_order=list(columns), column_config=columns)
st.caption(f"Mean, median and SD are taken over per-race MAE, so each race counts equally. Gain is measured against the baseline, and a negative gain means worse. The {label(manifest['baseline'])} row compares the baseline with itself, so every race is a tie.")
st.markdown(f"Full method and results: [project README on GitHub]({README_URL}).")
