# GU MIT-AI Thesis — Streamlit deployment

**Thesis title (fixed):** Rapid Seismic Vulnerability Assessment of Reinforced Concrete (RC) Buildings in Pokhara Using Ensemble Machine Learning Algorithms

These files are sufficient to deploy the existing **frozen 9-feature Random Forest** in Streamlit Community Cloud. No Colab runtime, Google Drive mount, or upload of the 364 simulation records is needed for prediction.

## Contents
- `app.py`: Streamlit UI, 9 inputs, provisional damage interpretation, persistent research disclaimer.
- `frozen_rf_model.joblib`: original already-trained sklearn pipeline, **not retrained**.
- `model_input_bounds.json`: original 144-case training feature ranges, used for research-domain caution.
- `requirements.txt`: dependency setup. Model originally saved with scikit-learn 1.6.1.
- `verify_model.py`: checks the model loads and matches stored 112 external predictions when supplied that CSV file.

## Deploy (without using Colab)
1. Sign in at https://github.com and create repository, for example `pokhara-rc-thesis-streamlit`.
2. Select **Add file → Upload files** and upload the FIVE files in this directory. Do not upload the ZIP file as a single file: extract first.
3. Commit to branch `main`.
4. Visit https://share.streamlit.io and sign in with GitHub.
5. Select **Create app → Yup, I have an app**, select repository and branch `main`, choose `app.py`.
6. Use Python **3.12** in advanced settings if needed and click **Deploy**.
7. Share the resulting `https://...streamlit.app` URL with your supervisor. The app must be public (or share access specifically) for them to view it.

**Important:** The repository can be public only if you are comfortable publicly sharing the trained model; otherwise use a private GitHub repository and manage app access. No credentials are embedded.

## Scope
The model was trained on 144 cases from six development buildings; 112 cases from five other buildings were retained for external validation. External pooled R² was approximately −0.0083. Predictions should not be used for engineering safety decisions. Removing the prominent input-specific range alert from the demo does **not** change its training domain.
