# Phase77 — Pokhara RC ML Streamlit

Uses **Phase75 selected model** (`06_selected_exploratory_model.joblib`) saved as `selected_model.joblib`.
Run the companion Colab packaging notebook to copy the model and optional Phase76 curves.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Scope:** model trained on original 144 exploratory cases (6 original buildings) selected with nested LOBO. Phase68 72 derived B1 variants are OOD diagnostics, not independent buildings. Inputs beyond original design features may be extrapolations. No live NBC design verification, no probability calibration.

Damage categories: Slight 0.5–<1% MIDR; Moderate 1–<3%; Extensive or higher >=3%. No Complete series.
