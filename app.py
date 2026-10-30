import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE=Path(__file__).resolve().parent
meta=json.loads((BASE/'model_metadata.json').read_text())
model=joblib.load(BASE/'midr_geometry_RF.joblib')
st.set_page_config(page_title='Pokhara RC Seismic ML',page_icon='🏢',layout='wide')
st.title('Pokhara RC — Seismic Vulnerability Estimator')
st.caption('20 ETABS buildings collected · 6 nonlinear models analyzed · 144 Phase 60 runs')
st.warning('Exploratory numerical assessment only. Hazus High-Code C1L damage levels are conditional; not verified physical damage or a structural safety certificate. One earthquake accelerogram.')

st.header('1. Building and earthquake inputs')
a,b,c=st.columns(3)
with a:
 pga=st.selectbox('Earthquake PGA (g)',meta['levels_PGA_g'])
 direction=st.selectbox('Earthquake direction',['X','Y'])
 area=st.number_input('Plan area proxy (m²)',min_value=0.1,value=120.0,step=1.0)
with b:
 cw=st.number_input('Column width (mm)',min_value=100.0,value=356.0,step=5.0)
 cd=st.number_input('Column depth (mm)',min_value=100.0,value=356.0,step=5.0)
 bw=st.number_input('Beam width (mm)',min_value=100.0,value=225.0,step=5.0)
 bd=st.number_input('Beam depth (mm)',min_value=100.0,value=425.0,step=5.0)
with c:
 floors=st.selectbox('Number of floors',[1,2,3],index=2)
 height=st.number_input('Building height (m)',min_value=2.0,max_value=9.0,value=9.0,step=0.5)
 rebar=st.selectbox('Longitudinal rebar grade',['HYSD500 / Fy 500 MPa','Fy 415 MPa (not simulated)'])
 scen=st.selectbox('Rebar reinforcement/confinement detailing',list(meta['scenario_values']),format_func=lambda x:x.replace('_',' ').title())
pars=meta['scenario_values'][scen]
st.caption(f"Detailing used in Phase 60: column steel {pars['assumed_column_rho']*100:.1f}%, beam steel {pars['assumed_beam_rho']*100:.1f}%, stirrup spacing {pars['assumed_tie_spacing_mm']:.0f} mm. These are scenario inputs, not an automatic assessment of actual installed reinforcement.")

vals={'target_PGA_g':float(pga),'dir_Y':int(direction=='Y'),'column_width_mm':float(cw),'column_depth_mm':float(cd),'beam_width_mm':float(bw),'beam_depth_mm':float(bd),'plan_bbox_area_m2':float(area),'storeys_from_frame':int(floors),'height_from_frame_m':float(height),**pars}
features=meta['range_by_feature']
outside=[k for k,v in vals.items() if v < features[k][0]-1e-5 or v > features[k][1]+1e-5]
unsupported=[]
if rebar != 'HYSD500 / Fy 500 MPa':unsupported.append('Fy 415 MPa was not simulated')
if floors != 3 or abs(height-9.0)>1e-5:unsupported.append('Current thesis-scope prediction supports only 3 floors / 9 m (1–2-floor models absent)')
for k in ['column_width_mm','column_depth_mm','beam_width_mm','beam_depth_mm']:
 if features[k][0] == features[k][1] and abs(vals[k]-features[k][0])>1e-4:
  unsupported.append(f'{k.replace("_"," ")} did not vary in training')
if outside:unsupported.append('Out of training range: '+', '.join(outside))
st.info('All geometric fields are editable so you can enter the building dimensions. If a value was not represented in training, the model withholds prediction rather than inventing a damage effect. Plan area can vary across its training range.')
if unsupported:
 st.warning('Prediction is unavailable for these inputs: '+'; '.join(dict.fromkeys(unsupported)))
if st.button('Estimate MIDR and damage state',type='primary',disabled=bool(unsupported)):
 X=pd.DataFrame([{k:vals[k] for k in meta['feature_order']}])
 pred=max(0.0,float(model.predict(X)[0]))
 idx=int(np.searchsorted(meta['hazus_threshold_ratio_assumed_C1L'],pred/100,side='right'))
 labels=['No damage','Slight','Moderate','Extensive','Complete threshold exceeded']
 l,r=st.columns(2)
 l.metric('Predicted MIDR',f'{pred:.3f}%')
 r.metric('Conditional Hazus damage state',labels[idx])
 st.caption('Prediction conditioned on assumed C1L High-Code thresholds. Complete-threshold exceedance does not establish physical collapse.')

st.header('2. Observed PGA–damage exceedance curves')
st.caption('From the 144 existing nonlinear cases; 36 cases per PGA. Not validated fragility curves and not a prediction for a user-entered geometry.')
obs=pd.read_csv(BASE/'hazus_observed_exceedance.csv')
pivot=obs.pivot(index='PGA_g',columns='threshold',values='observed_fraction').sort_index()
st.line_chart(pivot, y_label='Observed exceedance fraction (0–1)',x_label='PGA (g)')
with st.expander('See exceedance data'):
 st.dataframe(obs,hide_index=True)
st.header('3. Leave-one-building-out (LOBO) evaluation')
st.dataframe(pd.read_csv(BASE/'02_geometry_RF_LOBO_metrics.csv'),hide_index=True)
st.caption('LOBO evaluation includes the six analyzed buildings, including B15 which was extracted as 4 storeys / 12 m; this app restricts prediction to the requested 3-storey / 9 m scope. Metrics are not newly re-computed for this restricted interface.')
st.markdown('[FEMA Hazus earthquake technical manual]('+meta['hazus_source']+')')
