"""Method and limits page: framing, data attribution and caveats."""

import streamlit as st

from app_common import label, load_manifest, load_table

manifest = load_manifest()
n_laps = manifest["files"]["test_predictions.parquet"]["rows"]

st.title("Method and limits")
st.info(f"This is a 2025 held-out evaluation, not a live forecast. It shows saved predictions for {n_laps:,} eligible laps from {manifest['n_races']} dry races. The app runs no model.")

st.subheader("What this can and cannot say")
st.markdown("""
- **Tyre degradation** means declining tyre performance. **Tyre wear** means physical rubber loss. They are related but not the same.
- Lap time is the net of several effects. Tyre ageing makes the car slower, and fuel burn makes it lighter and faster. Track evolution, traffic, driver pace management and temperature also matter.
- Slower laps do not automatically mean tyre degradation. Traffic, safety cars, pit laps and weather can also cause them.
- Performance does not always decline smoothly.
- Public timing data lets this project model pace associated with tyre age. It does not measure remaining rubber or grip, and it does not isolate the causal effect of degradation. Results here are descriptive, not causal.
""")

st.subheader("How predictions were made")
st.markdown("""
- **Inputs:** only information available before a lap starts. The previous three laps must be clean, consecutive and in the same stint, otherwise the lap is dropped and never imputed.
- **Target:** each model predicts how much a lap differs from the mean of the previous three laps. Errors are reported in seconds of lap time.
- **Selection:** tuning and model choice used cross-validation on 2023–2024, with whole races held out together. The 2025 races played no part in those choices.
- **Scope:** dry Grand Prix races and clean, green-flag laps only. This is next-lap prediction, not pit-strategy advice.
""")

st.subheader("Model selection on training races")
cv_summary = load_table("cv_summary.csv")
st.dataframe(cv_summary.assign(predictor=cv_summary["predictor"].map(label)), hide_index=True, column_config={
    "predictor": "Predictor", "n_races": "Races",
    "mean_mae_s": st.column_config.NumberColumn("Mean race MAE (s)", format="%.4f"),
    "median_mae_s": st.column_config.NumberColumn("Median race MAE (s)", format="%.4f"),
    "std_mae_s": st.column_config.NumberColumn("SD of race MAE (s)", format="%.4f"),
    "improvement_s": st.column_config.NumberColumn("Gain vs baseline (s)", format="%+.4f"),
    "improvement_pct": st.column_config.NumberColumn("Gain vs baseline (%)", format="%+.2f"),
    "races_won_vs_baseline": "Races won",
})
st.caption(f"Out-of-fold results on 2023–2024 races. These are development estimates, because parameters were selected on the same folds. {label(manifest['chosen_model'])} had the lowest mean race MAE, which is why it is the model shown throughout.")

st.subheader("Data sources")
commits = "\n".join(f"- {season}: [{source['repository']} at {source['commit'][:7]}](https://github.com/{source['repository']}/tree/{source['commit']})"
                    for season, source in sorted(manifest["source_commits"].items()))
st.markdown(
    "Lap data comes from the public TracingInsights archive, which is derived from "
    "[FastF1](https://github.com/theOehrly/Fast-F1), at these pinned commits:\n\n" + commits
)
st.markdown(
    "The publisher applies official Ergast/Jolpica lap-time corrections and exports durations to the millisecond. "
    "The archive repositories carry their own licences: MIT for 2023 and 2024, Apache-2.0 for 2025. "
    "Copies of those licences and a notice of the changes made to the data are in `app_data/LICENSES` in the project repository."
)
st.caption(f"Exported {manifest['exported_at']}. This is an unofficial project, not associated with the Formula 1 companies or the FIA. F1, FORMULA 1 and GRAND PRIX are trade marks of Formula One Licensing B.V.")
