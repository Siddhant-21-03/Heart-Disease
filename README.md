# Heart Disease Analytics 

An end-to-end Healthcare Analytics project that ingests UCI heart disease data, performs preprocessing and feature engineering, trains a reproducible scikit-learn pipeline, and exposes an interactive Streamlit dashboard for EDA and patient-level risk prediction with human-friendly insights.

Why this is resume-worthy
- Demonstrates full ML lifecycle: data cleaning -> feature engineering -> model training -> deployment-ready app
- Shows product thinking: human-readable insights, empty-by-default input UX, and fallback logic for robust inference
- Contains reproducible code, a persisted model artifact, unit tests, and Docker support for easy sharing

Highlights
- Trained a Random Forest pipeline with preprocessing (imputation, scaling, one-hot encoding) and saved as `models/rf_pipeline.joblib`.
- Streamlit app (`src/app/streamlit_app.py`) provides EDA charts, an 8-field patient input form, automatic fallback filling for model-required features, and a personalized insights summary instead of raw model metrics.
- Includes `scripts/smoke_predict.py` and `tests/test_predict.py` for quick verification.

Technologies
- Python, pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, RandomForest)
- joblib for model persistence
- Streamlit, Plotly, Seaborn for UI and visualizations
- pytest for testing, Docker for containerization

Repository layout (most relevant files)
- `src/app/streamlit_app.py` — Streamlit dashboard and prediction UI
- `src/models/train_model.py` — training script & pipeline builder
- `src/models/predict.py` — model loader and prediction helper
- `data/processed/heart_disease_processed.csv` — canonical processed dataset used for fallbacks and insights
- `models/rf_pipeline.joblib` — trained model artifact (if present)
- `scripts/smoke_predict.py` — quick verification script
- `tests/` — minimal unit tests

Quickstart (Windows PowerShell)

1. Create a virtual environment and install dependencies

```powershell
cd D:/DApro/heart_disease_dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run tests and smoke-check

```powershell
$env:PYTHONPATH='.'; pytest -q
$env:PYTHONPATH='.'; python scripts/smoke_predict.py
```

4. Start the Streamlit app

```powershell
$env:PYTHONPATH='.'; .\.venv\Scripts\Activate.ps1
streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
