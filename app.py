import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_assets():
    meta = json.loads((ART / "metadata.json").read_text())
    models = {name: joblib.load(ART / f"{name}.joblib")
              for name in ["reg_early", "reg_midterm", "clf_early", "clf_midterm"]}
    return meta, models


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["parent_edu"] = (d["Medu"] + d["Fedu"]) / 2
    d["alc_total"] = d["Dalc"] + d["Walc"]
    d["log_absences"] = np.log1p(d["absences"])
    if {"G1", "G2"} <= set(d.columns):
        d["grade_trend"] = d["G2"] - d["G1"]
        d["grade_mean"] = (d["G1"] + d["G2"]) / 2
    return d


meta, models = load_assets()

st.title("🎓 Academic Performance Forecasting & Student Success Predictor")
st.caption("Decision-support prototype trained on 395 students (two Portuguese schools, Mathematics). "
           "Predictions are estimates – never use them as the only basis for a decision about a student.")

mode_label = st.radio("Prediction mode",
                      ["Early warning (no grades yet)", "Mid-term forecast (G1 & G2 known)"], horizontal=True)
mode = "early" if mode_label.startswith("Early") else "midterm"
key = "B_early" if mode == "early" else "A_midterm"

if mode == "early":
    st.info("Early-warning mode is a screening aid only: on test data it catches about half of the students "
            "who go on to fail. Use the mid-term mode once G1 and G2 are available.")

inputs = [c for c in meta["raw_inputs"] if mode == "midterm" or c not in ("G1", "G2")]
values = {}
cols = st.columns(4)
for i, c in enumerate(inputs):
    with cols[i % 4]:
        if c in meta["categorical_options"]:
            values[c] = st.selectbox(c, meta["categorical_options"][c], key=f"{mode}_{c}")
        else:
            lo, hi, med = meta["numeric_ranges"][c]
            hi = 100 if c == "absences" else max(hi, lo + 1)
            values[c] = st.number_input(c, min_value=lo, max_value=hi, value=med, step=1, key=f"{mode}_{c}")

if st.button("Predict", type="primary"):
    X = add_features(pd.DataFrame([values]))[meta["features"][key]]
    grade = float(np.clip(models[f"reg_{mode}"].predict(X)[0], 0, 20))
    p = float(models[f"clf_{mode}"].predict_proba(X)[0, 1])
    thr = meta["thresholds"][key]
    band = "HIGH" if p >= thr else ("MEDIUM" if p >= thr * 0.6 else "LOW")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted final grade (0-20)", f"{grade:.1f}")
    c2.metric("Probability of failing (G3 < 10)", f"{p:.0%}")
    c3.metric("Risk level", band)
    if band == "HIGH":
        st.error("High risk: consider early academic support, a mentor check-in and a talk about attendance and study habits.")
    elif band == "MEDIUM":
        st.warning("Moderate risk: monitor progress and offer optional support.")
    else:
        st.success("Low risk: on track based on the available information.")
