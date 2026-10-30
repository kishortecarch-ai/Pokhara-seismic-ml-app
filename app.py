import streamlit as st, pandas as pd, numpy as np, joblib, json
from pathlib import Path
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
meta=json.loads((P/'model_metadata.json').read_text())
reg=joblib.load(P/'midr_best_exploratory_model.joblib')
clf=joblib.load(P/'hazus_conditional_classifier.joblib')
st.set_page_config(page_title='Pokhara RC — Exploratory Seismic ML',layout='wide')
st.title('Pokhara RC — Exploratory Seismic Vulnerability Tool')
st.warning('Research prototype only. Hazus High-Code C1 design-level is an unverified conditional assumption; nonlinear runs used one ground-motion record. Not for structural safety certification or physical-collapse claims.')
st.caption('Predicted MIDR is an ML estimate. Damage group uses published Hazus drift thresholds only under the selected conditional building category; not observed actual damage.')
scens={'Lower detailing':(.010,.007,150,1.08),'Nominal assumption':(.015,.010,100,1.20),'Upper detailing':(.025,.015,75,1.30)}
st.subheader('Model inputs')
a,b,c=st.columns(3)
with a: pga=st.selectbox('PGA (g) — simulated levels',meta['training_PGA_levels_g'],index=0)
with b: direction=st.selectbox('Earthquake direction',['X','Y'])
with c: scen=st.selectbox('Detailing sensitivity scenario',list(scens))
stories=st.selectbox('Assumed RC C1 height group / number of stories',[3,4,5,6,7,8,9,10],index=0)
st.caption('Story count and C1 structural type must be checked against actual building evidence. Scenario detailing is assumed and does not establish Hazus design level.')
rho_col,rho_beam,tie,core=scens[scen]
vals=[pga,int(direction=='Y'),rho_col,rho_beam,tie,core,stories]
X=pd.DataFrame([vals],columns=meta['feature_order'])
# Warn about unseen covariates / combinations
outside=[col for col,v in zip(meta['feature_order'],vals) if not(meta['range_by_feature'][col][0]<=v<=meta['range_by_feature'][col][1])]
if outside:st.error('Out-of-training-domain inputs: '+', '.join(outside)+'. Prediction withheld.')
else:
 pred=float(reg.predict(X)[0]);pred=max(0,pred)
 hazus='C1L' if stories<=3 else ('C1M' if stories<=7 else 'C1H')
 thresholds=meta['hazus_high_code_c1_threshold_ratio'][hazus]
 five=int(np.searchsorted(thresholds,pred/100,side='right'))
 four={0:0,1:0,2:1,3:2,4:3}[five]
 names=['No / Slight','Moderate','Extensive','Complete threshold exceeded']
 class_id=int(np.asarray(clf.predict(X),dtype=int)[0]); class_id=meta['class_decoder'][class_id]
 l,r=st.columns(2)
 l.metric('Predicted MIDR',f'{pred:.3f}%')
 r.metric('MIDR-derived conditional Hazus group',f'{four} — {names[four]}')
 st.write(f'Independent ML classifier: **{class_id} — {names[class_id]}** (may disagree with MIDR-derived group).')
 st.info('Hazus uses five original states (No, Slight, Moderate, Extensive, Complete). This tool combines No/Slight into one analytical group; threshold exceedance is not proof of collapse.')
st.subheader('Observed simulation exceedance — not a validated fragility curve')
obs=pd.read_csv(P/'06_observed_exceedance_NOT_VALIDATED_FRAGILITY.csv')
fig,ax=plt.subplots(figsize=(7,3.7))
for k,g in obs.groupby('exceedance_four_group_at_least'):
 ax.plot(g.PGA_g,g.observed_fraction,marker='o',label=f'Group ≥ {k}')
ax.set(xlabel='PGA (g)',ylabel='Observed exceedance fraction',ylim=(0,1.05))
ax.legend();ax.grid(alpha=.2);st.pyplot(fig)
st.caption('Each PGA pools six buildings, X/Y and three detailing scenarios for a single earthquake accelerogram; these are not independent records.')
st.subheader('LOBO validation metrics')
st.dataframe(pd.read_csv(P/'01_regression_LOBO_metrics.csv'),hide_index=True)
st.dataframe(pd.read_csv(P/'02_hazus_classification_LOBO_metrics.csv'),hide_index=True)
st.markdown('[FEMA Hazus technical manual]('+meta['hazus_source']+')')
