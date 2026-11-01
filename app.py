from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RC Seismic Vulnerability — GU MIT-AI", layout="centered")
st.title("Rapid Seismic Vulnerability Assessment of RC Buildings in Pokhara")
st.caption("MIT-AI thesis research demonstration • Gandaki University")
st.warning("Exploratory simulated drift only. Not validated for structural safety or real damage decisions.")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "frozen_rf_model.joblib")

pkg = load_model()
meta = json.loads((ROOT / "model_input_bounds.json").read_text(encoding="utf-8"))
features = meta["feature_order"]
limits = meta["training_feature_ranges"]

with st.form("prediction"):
    pga = st.number_input("Peak ground acceleration, PGA (g)", min_value=0.01, max_value=3.0, value=0.55, step=0.05)
    direction = st.selectbox("Earthquake direction", ["X", "Y"])
    cw = st.number_input("Column width (mm)", min_value=1.0, value=356.0, step=10.0)
    cd = st.number_input("Column depth (mm)", min_value=1.0, value=356.0, step=10.0)
    bw = st.number_input("Beam width (mm)", min_value=1.0, value=225.0, step=10.0)
    bd = st.number_input("Beam depth (mm)", min_value=1.0, value=425.0, step=10.0)
    storeys = st.number_input("Number of storeys", min_value=1, value=3, step=1)
    height = st.number_input("Frame height (m)", min_value=0.1, value=9.0, step=0.5)
    area = st.number_input("Plan bounding-box area (m²)", min_value=0.1, value=135.0, step=5.0)
    submitted = st.form_submit_button("Predict simulated MIDR", type="primary")

if submitted:
    vals = dict(target_PGA_g=pga, direction=pkg["direction_encoding"][direction],
                column_width_mm=cw, column_depth_mm=cd,
                beam_width_mm=bw, beam_depth_mm=bd,
                storeys_from_frame=storeys, height_from_frame_m=height,
                plan_bbox_area_m2=area)
    if any(not np.isfinite(v) or v <= 0 for k,v in vals.items() if k != "direction"):
        st.error("Please enter finite, positive physical dimensions and PGA.")
        st.stop()
    # The interface intentionally does not display an individual out-of-range red alert.
    # Training domain is NOT expanded: the scientific caution below applies to every prediction.
    outside = [k for k,v in vals.items() if v < limits[k]["min"]-1e-9 or v > limits[k]["max"]+1e-9]
    y = float(pkg["pipeline"].predict(pd.DataFrame([vals], columns=features))[0])
    states = ["None", "Slight", "Moderate", "Extensive", "Complete"]
    i = int(np.searchsorted([0.5, 1.0, 3.0, 8.0], y, side="right"))
    st.metric("Predicted simulated maximum inter-storey drift ratio", f"{y:.3f}%")
    st.markdown(f"**Provisional damage state:** {states[i]}")
    if outside:
        st.caption("Research caution: inputs may fall outside the six-building development domain; prediction reliability is not established for unseen configurations.")
    st.caption("Generated from one scaled El Centro 1940 NS component and unverified numerical labels, not observed building damage.")
