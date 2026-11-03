"""Race explorer page: one driver's laps, with actual and predicted times."""

import streamlit as st

from app_common import ACTUAL_LABEL, label, load_manifest, load_predictions, load_table, race_chart

manifest = load_manifest()
model, baseline = manifest["chosen_model"], manifest["baseline"]
predictions = load_predictions()
results = load_table("test_results_by_race.csv").set_index("race_id")

st.title("Race explorer")
st.markdown("Pick a 2025 race and a driver to compare actual lap times with the two predictions made for each lap.")

races = predictions[["race_id", "event_name", "circuit"]].drop_duplicates()
names = {f"{race.event_name} ({race.circuit})": race for race in races.itertuples(index=False)}
race_column, driver_column = st.columns(2)
race_name = race_column.selectbox("Race", sorted(names), key="race")
race = names[race_name]

race_laps = predictions[predictions["race_id"].eq(race.race_id)]
teams = race_laps.drop_duplicates("driver").set_index("driver")["team"].sort_index()
drivers = {f"{driver} ({team})": driver for driver, team in teams.items()}
driver = drivers[driver_column.selectbox("Driver", list(drivers), key="driver")]

laps = race_laps[race_laps["driver"].eq(driver)]
st.altair_chart(race_chart(laps, model, baseline, f"{driver}: actual and predicted lap times, {race.event_name}"), width="stretch")
st.caption(
    "Removed laps (pit in and out laps, safety car and yellow-flag laps, deleted laps, and laps without three clean laps before them in the same stint) "
    "are not shown, so lines break where they were. Shading separates stints. "
    f"{label(baseline)} repeats the previous lap, so its line follows the actual line one lap later."
)

row = results.loc[race.race_id]
st.subheader("This race, all drivers")
model_column, baseline_column, gain_column, laps_column = st.columns(4)
model_column.metric("Model race MAE", f"{row[model]:.3f} s", help=f"{label(model)}, across all eligible laps of every driver in this race.")
baseline_column.metric("Baseline race MAE", f"{row[baseline]:.3f} s", help=f"{label(baseline)}, across all eligible laps of every driver in this race.")
gain_column.metric("Model gain", f"{row[baseline] - row[model]:+.3f} s", help="Baseline MAE minus model MAE. Negative means the model lost this race.")
laps_column.metric("Eligible laps", f"{int(row['n_laps']):,}")

with st.expander("Lap values for this driver"):
    table = laps.sort_values("lap_number")[["lap_number", "stint", "compound", "tyre_age", "lap_time_s", model, baseline]]
    seconds = st.column_config.NumberColumn
    st.dataframe(table, hide_index=True, column_config={
        "lap_number": "Lap", "stint": "Stint", "compound": "Compound",
        "tyre_age": seconds("Tyre age (laps)", format="%d"),
        "lap_time_s": seconds(f"{ACTUAL_LABEL} (s)", format="%.3f"),
        model: seconds(f"{label(model)} (s)", format="%.3f"),
        baseline: seconds(f"{label(baseline)} (s)", format="%.3f"),
    })
