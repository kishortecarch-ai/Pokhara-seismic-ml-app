"""Verify frozen-model reproduction. Run: python verify_model.py /path/to/THESIS_FINAL_08_unseen_building_predictions.csv"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
pkg=joblib.load(ROOT/'frozen_rf_model.joblib')
assert pkg['direction_encoding']=={'X':0,'Y':1}
print('Model loaded, direction encoding:',pkg['direction_encoding'])
if len(sys.argv)<2:
    print('For 112-case test, provide external prediction CSV path.')
    sys.exit(0)
df=pd.read_csv(sys.argv[1])
features=list(pkg['features'])
vals=df[features].copy()
if vals['direction'].dtype==object:
    vals['direction']=vals['direction'].map(pkg['direction_encoding'])
pred=pkg['pipeline'].predict(vals)
reference=df['pred_MIDR_pct'].to_numpy(float)
diff=np.abs(pred-reference)
print('Rows:',len(df),'maximum absolute error:',float(diff.max()),'matched:',int(np.count_nonzero(diff<1e-10)))
assert len(df)==112 and np.allclose(pred,reference,atol=1e-10,rtol=0)
print('PASS: 112/112 predictions reproduced')
