
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Pokhara RC Seismic ML", page_icon="🏢",layout="wide")
st.title("Rapid Seismic Assessment of RC Buildings")
st.caption("Phase 77 · Pokhara RC · Experimental machine-learning research tool")

MODEL=Path(__file__).resolve().parent/"selected_model.joblib"
if not MODEL.exists():
    st.error("Missing selected_model.joblib. Run the Phase77 packaging notebook or copy Phase75 model alongside app.py.")
    st.stop()
try:
    pack=joblib.load(MODEL)
    model=pack["pipeline"]
    FEATURES=pack["features"]
    DIRECTIONS=pack.get("direction_encoding",{"X":0,"Y":1})
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

st.info("Outputs reflect exploratory OpenSees scenarios and the selected Phase75 ML model. "
        "They are not design-code checks, engineering certification, or calibrated earthquake damage probabilities.")

with st.sidebar:
    st.header("Building and ground motion")
    pga=st.slider("PGA (g)",0.35,0.80,0.45,0.01)
    direction=st.radio("Direction",["X","Y"],horizontal=True)
    col_w=st.selectbox("Column width (mm)",[300,356,400],index=1)
    col_d=st.selectbox("Column depth (mm)",[300,356,400],index=1)
    bw=st.number_input("Beam width (mm)",min_value=150.0,max_value=500.0,value=225.0,step=5.0)
    bd=st.number_input("Beam depth (mm)",min_value=250.0,max_value=850.0,value=425.0,step=5.0)
    storeys=st.selectbox("Storeys",[1,2,3],index=2)
    height=st.number_input("Frame height (m)",min_value=3.0,max_value=12.0,value=float(storeys*3),step=0.5)
    area=st.number_input("Plan XY bounding-box area proxy (m²)",min_value=40.0,max_value=400.0,value=180.68376144,step=1.0,
                         help="ETABS XY-node bounding rectangle, NOT verified architectural floor area.")
    if col_w!=col_d: st.warning("Rectangular columns are outside the original size-variation experiment.")
    if storeys!=3 or col_w!=356 or col_d!=356:
        st.warning("Input includes B1-parametric variations. Phase75 final model was trained on original 144 cases; prediction may be out-of-domain.")

def make_input(pga_value):
    all_values={
        "target_PGA_g":float(pga_value),
        "direction":int(DIRECTIONS[direction]),
        "column_width_mm":float(col_w),
        "column_depth_mm":float(col_d),
        "beam_width_mm":float(bw),
        "beam_depth_mm":float(bd),
        "storeys_from_frame":int(storeys),
        "height_from_frame_m":float(height),
        "plan_bbox_area_m2":float(area),
    }
    missing=[c for c in FEATURES if c not in all_values]
    if missing: raise ValueError(f"Unknown model features: {missing}")
    return pd.DataFrame([{k:all_values[k] for k in FEATURES}],columns=FEATURES)

def damage_state(midr):
    if midr<0.5: return "Below Slight threshold"
    if midr<1.0: return "Slight"
    if midr<3.0: return "Moderate"
    return "Extensive or higher"

try:
    prediction=max(0.0,float(model.predict(make_input(pga))[0]))
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

left,right=st.columns(2)
left.metric("Predicted MIDR",f"{prediction:.3f}%")
right.metric("Damage category",damage_state(prediction))
st.caption("Damage categories use illustrative Phase64 C1L high-code drift thresholds. "
           "The threshold-based category is not a verified observed damage state.")

st.subheader("PGA response of the ML model")
pgas=np.linspace(0.35,0.8,46)
mids=np.array([max(0.0,float(model.predict(make_input(x))[0])) for x in pgas])
fig=go.Figure()
fig.add_trace(go.Scatter(x=pgas,y=mids,mode="lines+markers",name="Predicted MIDR (%)"))
for threshold,name in [(0.5,"Slight"),(1.0,"Moderate"),(3.0,"Extensive or higher")]:
    fig.add_hline(y=threshold,line_dash="dash",annotation_text=f"{name} threshold")
fig.update_layout(xaxis_title="PGA (g)",yaxis_title="Predicted MIDR (%)",height=420)
st.plotly_chart(fig,use_container_width=True)
st.caption("This is a predicted MIDR–PGA response, not a probabilistic earthquake fragility curve.")

st.subheader("Phase 76 earthquake fragility analysis (scenario groups)")
CURVES=Path(__file__).resolve().parent/"phase76_curves.csv"
if CURVES.exists():
    c=pd.read_csv(CURVES)
    required={"research_group","PGA_g","damage_category","percentage"}
    if required.issubset(c):
        group=st.selectbox("Simulation group",sorted(c.research_group.unique()))
        filtered=c[c.research_group.eq(group)].copy()
        fig2=go.Figure()
        for state in ["Slight","Moderate","Extensive or higher"]:
            d=filtered[filtered.damage_category.eq(state)].sort_values("PGA_g")
            if len(d):fig2.add_trace(go.Scatter(x=d.PGA_g,y=d.percentage,mode="lines+markers",name=state))
        fig2.update_layout(title="Earthquake Fragility Curves — simulation scenario fractions",
                           xaxis_title="PGA (g)",yaxis_title="Percentage of simulation cases (%)",yaxis_range=[0,100])
        st.plotly_chart(fig2,use_container_width=True)
        st.caption("Slight and Moderate are mutually exclusive states. Extensive or higher is cumulative. "
                   "Fractions across structural scenarios are not statistically calibrated earthquake probabilities.")
else:
    st.caption("Phase76 curves are optional. Run the Phase77 packaging notebook to copy their CSV if available.")

with st.expander("Model and research scope"):
    st.json({"Phase75_training_regime":pack.get("regime","unknown"),
             "features":FEATURES,"model_status":pack.get("status","exploratory"),
             "independent_original_buildings":6,"ground_motion":"single El Centro earthquake record",
             "area_is_proxy":True,"B1_72_variants_are_independent_buildings":False})
