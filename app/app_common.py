"""File loading, labels and charts shared by the app pages."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP_DATA_DIR = Path(__file__).resolve().parent.parent / "app_data"
MANIFEST_FILE = "manifest.json"
SOURCE_NOTE = "Data: FastF1-derived TracingInsights archive"
README_URL = "https://github.com/filan214/F1-tyre-degradation#readme"
PREDICTOR_LABELS = {
    "pred_rf": "Random Forest", "pred_xgb": "XGBoost", "pred_ridge": "Ridge", "pred_b1": "B1 Last lap",
    "pred_b2": "B2 Mean of 3", "pred_b3": "B3 Mean of 5", "pred_b4": "B4 Stint trend",
}
ACTUAL_LABEL = "Actual lap time"
BETTER_LABEL = "Model better"
WORSE_LABEL = "Model worse"

# One set, readable on both Streamlit surfaces, so a chart never depends on detecting the viewer's theme.
COLOURS = {"actual": "#6f6e69", "model": "#2a78d6", "baseline": "#d95926", "worse": "#e34948", "muted": "#898781"}


@st.cache_data
def load_manifest() -> dict:
    """Read provenance, chosen predictors and row counts."""
    return json.loads((APP_DATA_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    """Read one exported result table."""
    return pd.read_csv(APP_DATA_DIR / name)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Read the exported lap-level 2025 predictions."""
    return pd.read_parquet(APP_DATA_DIR / "test_predictions.parquet")


def label(predictor: str) -> str:
    """Return the display name of a predictor column."""
    return PREDICTOR_LABELS.get(predictor, predictor)


def _title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, subtitle=SOURCE_NOTE, subtitleColor=COLOURS["muted"], anchor="start")


def line_segments(laps: pd.DataFrame) -> pd.Series:
    """Number runs of consecutive same-stint laps, so lines break where laps were removed."""
    continues = laps["lap_number"].diff().eq(1) & laps["stint"].diff().eq(0)
    return (~continues).cumsum()


def race_chart(laps: pd.DataFrame, model: str, baseline: str, title: str) -> alt.LayerChart:
    """Plot actual, model and baseline lap times by lap, with shaded stints and hover values."""
    colours = COLOURS
    laps = laps.sort_values("lap_number").reset_index(drop=True)
    laps = laps.assign(segment=line_segments(laps), actual=laps["lap_time_s"], model=laps[model], baseline=laps[baseline])
    series = {"actual": ACTUAL_LABEL, "model": label(model), "baseline": label(baseline)}
    long = laps.melt(id_vars=["lap_number", "segment"], value_vars=list(series), var_name="key", value_name="seconds")
    long["series"] = long["key"].map(series)

    lap_scale = alt.Scale(domain=[int(laps["lap_number"].min()) - 1, int(laps["lap_number"].max()) + 1], nice=False)
    lap_axis = alt.X("lap_number:Q", title="Lap number", scale=lap_scale, axis=alt.Axis(format="d", tickMinStep=1, grid=False))
    stints = (laps.groupby("stint").agg(first=("lap_number", "min"), last=("lap_number", "max"), compound=("compound", "first")).reset_index())
    stints = stints.assign(start=stints["first"] - 0.5, end=stints["last"] + 0.5, middle=(stints["first"] + stints["last"]) / 2,
                           name="Stint " + stints["stint"].astype(str) + " " + stints["compound"].str.title(),
                           shaded=stints.index % 2 == 1)
    bands = (alt.Chart(stints.loc[stints["shaded"]]).mark_rect(color=colours["muted"], opacity=0.12)
             .encode(x=alt.X("start:Q", scale=lap_scale), x2="end:Q"))
    names = (alt.Chart(stints).mark_text(color=colours["muted"], fontSize=12, baseline="top")
             .encode(x=alt.X("middle:Q", scale=lap_scale), y=alt.value(4), text="name:N"))

    order = [series["actual"], series["model"], series["baseline"]]
    legend = alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=2, symbolSize=400, labelLimit=240)
    lines = (alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=28))
             .encode(x=lap_axis,
                     y=alt.Y("seconds:Q", title="Lap time (seconds)", scale=alt.Scale(zero=False)),
                     color=alt.Color("series:N", sort=order, legend=legend,
                                     scale=alt.Scale(domain=order, range=[colours["actual"], colours["model"], colours["baseline"]])),
                     strokeDash=alt.StrokeDash("series:N", sort=order, legend=legend,
                                               scale=alt.Scale(domain=order, range=[[1, 0], [1, 0], [5, 3]])),
                     detail="segment:N"))

    hover = alt.selection_point(name="lap_hover", fields=["lap_number"], on="pointerover", clear="pointerout", empty=False)
    tooltip = [alt.Tooltip("lap_number:Q", title="Lap"), alt.Tooltip("stint:Q", title="Stint"),
               alt.Tooltip("compound:N", title="Compound"), alt.Tooltip("tyre_age:Q", title="Tyre age (laps)"),
               alt.Tooltip("actual:Q", title=f"{series['actual']} (s)", format=".3f"),
               alt.Tooltip("model:Q", title=f"{series['model']} (s)", format=".3f"),
               alt.Tooltip("baseline:Q", title=f"{series['baseline']} (s)", format=".3f")]
    columns = ["lap_number", "stint", "compound", "tyre_age", "actual", "model", "baseline"]
    hit_area = alt.Chart(laps[columns]).mark_rule(strokeWidth=16, opacity=0).encode(x=lap_axis, tooltip=tooltip).add_params(hover)
    marker = alt.Chart(laps[columns]).mark_rule(color=colours["muted"], strokeWidth=1).encode(x=lap_axis).transform_filter(hover)
    return alt.layer(bands, marker, lines, names, hit_area).resolve_scale(x="shared").properties(title=_title(title), height=430)


def gain_chart(table: pd.DataFrame, group: str, title: str) -> alt.LayerChart:
    """Plot model gain per group in table order, in red with a value label where the model loses."""
    colours = COLOURS
    table = table.assign(outcome=[BETTER_LABEL if gain >= 0 else WORSE_LABEL for gain in table["improvement_s"]])
    groups = alt.Y(f"{group}:N", sort=table[group].tolist(), title=None, axis=alt.Axis(labelLimit=260, ticks=False, grid=False))
    gain = alt.X("improvement_s:Q", title="Model gain versus baseline (seconds; positive means the model is better)")
    tooltip = [alt.Tooltip(f"{group}:N", title="Group"), alt.Tooltip("improvement_s:Q", title="Gain (s)", format="+.4f"),
               alt.Tooltip("model_mean_race_mae_s:Q", title="Model MAE (s)", format=".4f"),
               alt.Tooltip("baseline_mean_race_mae_s:Q", title="Baseline MAE (s)", format=".4f"),
               alt.Tooltip("races_won:Q", title="Races won"), alt.Tooltip("races_lost:Q", title="Races lost")]
    outcomes = alt.Color("outcome:N", legend=alt.Legend(orient="top", title=None),
                         scale=alt.Scale(domain=[BETTER_LABEL, WORSE_LABEL], range=[colours["model"], colours["worse"]]))
    bars = alt.Chart(table).mark_bar(size=14, cornerRadiusEnd=4).encode(x=gain, y=groups, color=outcomes, tooltip=tooltip)
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=colours["muted"]).encode(x="zero:Q")
    losses = (alt.Chart(table[table["improvement_s"] < 0]).mark_text(align="right", dx=-5, color=colours["worse"], fontWeight="bold")
              .encode(x=gain, y=groups, text=alt.Text("improvement_s:Q", format="+.3f")))
    # A step per row, because a fixed height also has to hold the title, legend and axis.
    return alt.layer(bars, zero, losses).properties(title=_title(title), height=alt.Step(28))
