import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Pokhara RC | Seismic ML", page_icon="🏢", layout="centered")
meta = json.loads((BASE / "model_metadata.json").read_text(encoding="utf-8"))
model = joblib.load(BASE / "midr_geometry_RF.joblib")

st.title("Pokhara RC — Seismic Vulnerability Estimator")
st.caption("Phase 66 | Six modeled buildings | 144 exploratory nonlinear cases")
st.warning("Research prototype, not a structural safety certification. One ground-motion record and conditional Hazus High-Code C1L criteria; engineering validation is incomplete.")

st.subheader("Building and earthquake inputs")
pga = st.selectbox("Earthquake PGA (g)", meta["levels_PGA_g"])
direction = st.selectbox("Direction", ["X", "Y"])
scenarios = {
 "Lower detailing": "lower_detailing",
 "Nominal detailing": "nominal_assumption",
 "Upper detailing": "upper_detailing"
}
scenario_label = st.selectbox("Reinforcement / confinement scenario", list(scenarios))
scenario = scenarios[scenario_label]

st.subheader("Geometry")
col1, col2 = st.columns(2)
with col1:
 st.number_input("Column width (mm) — fixed in training", value=356, disabled=True)
 st.number_input("Column depth (mm) — fixed in training", value=356, disabled=True)
 st.number_input("Beam width (mm) — fixed in training", value=225, disabled=True)
 st.number_input("Beam depth (mm) — fixed in training", value=425, disabled=True)
with col2:
 area = st.number_input("ETABS node-plan area proxy (m²)", min_value=96.6242, max_value=180.6838, value=120.0, step=1.0)
 st.number_input("Floors — scope-limited", value=3, disabled=True)
 st.number_input("Frame height (m) — scope-limited", value=9.0, disabled=True)
 st.selectbox("Rebar grade (simulation-supported)", ["HYSD500 / Fy 500 MPa"], disabled=True)

st.info("Column and beam dimensions were identical in all six modeled buildings: changing them cannot yield a learned prediction. Likewise, Fy was fixed at 500 MPa. This version only predicts within the 3-storey / 9 m scope; 1–2 floor models were not trained.")
pars = meta["scenario_values"][scenario]
st.caption(f"Scenario parameters: column steel ratio {pars['assumed_column_rho']:.1%}, beam ratio {pars['assumed_beam_rho']:.1%}, ties {pars['assumed_tie_spacing_mm']:.0f} mm. Detailing parameter effects are exploratory.")
values = {
 "target_PGA_g": float(pga), "dir_Y": int(direction=="Y"),
 "column_width_mm": 356., "column_depth_mm": 356.,
 "beam_width_mm": 225., "beam_depth_mm": 425.,
 "plan_bbox_area_m2": float(area), "storeys_from_frame": 3,
 "height_from_frame_m": 9.0, **pars
}
outside=[k for k,v in values.items() if not (meta["range_by_feature"][k][0]-1e-6<=v<=meta["range_by_feature"][k][1]+1e-6)]
if outside:
 st.error("Inputs outside training ranges: "+", ".join(outside))
elif st.button("Estimate seismic damage", type="primary"):
 x=pd.DataFrame([{k:values[k] for k in meta["feature_order"]}])
 midr=max(0.,float(model.predict(x)[0]))
 thresholds=meta["hazus_threshold_ratio_assumed_C1L"]
 idx=int(np.searchsorted(thresholds,midr/100,side="right"))
 labels=["No damage","Slight","Moderate","Extensive","Complete threshold exceeded"]
 a,b=st.columns(2)
 a.metric("Estimated MIDR",f"{midr:.3f}%")
 b.metric("Conditional Hazus state", labels[idx])
 st.caption("The state is assigned from predicted MIDR and assumed C1L High-Code drift thresholds. Complete threshold exceedance is not verified physical collapse. No damage is inferred only by these thresholds.")
 st.info("Area is a node-coordinate bounding-box proxy, not measured floor area. Within-range input combinations may still differ from those actually simulated.")

st.subheader("Held-out building validation")
st.dataframe(pd.read_csv(BASE/"02_geometry_RF_LOBO_metrics.csv"), hide_index=True)
st.caption("Reported LOBO metrics cover all six modeled buildings, including B15 (4 storeys, 12 m). They are NOT independently recalculated for this 3-storey-only interface.")
st.markdown("[FEMA Hazus Earthquake Model technical manual]("+meta["hazus_source"]+")")
