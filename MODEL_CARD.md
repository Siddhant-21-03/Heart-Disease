# Model Card — Heart Disease Random Forest Pipeline

## Model Details
- **Name:** Random Forest classifier pipeline
- **Version:** 1.0
- **Author:** (Your Name) — project scaffold
- **Date:** 2025-11-20
- **Model artifact:** `models/rf_pipeline.joblib`
- **Frameworks/Libraries:** scikit-learn, joblib

## Intended Use
This model predicts the likelihood of heart disease for individual patients using clinical measurements (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, exercise-induced angina, ST depression, and additional engineered / processed features). It is intended for educational/demo purposes and not for clinical decision-making.

Primary use cases:
- Demonstration and training of ML model lifecycle
- Interactive dashboards showing how input features affect risk
- Rapid prototyping for healthcare analytics

## Not intended use
- Medical diagnosis or clinical decision support without validation and regulatory approval.

## Data
- Source: UCI Heart Disease dataset (combined from Cleveland, Hungarian, etc.)
- Processed data used for training is available at `data/processed/heart_disease_processed.csv`.
- Missing value strategy: median for numeric, most-frequent for categorical (see `src/models/train_model.py`).

## Model architecture and training
- Preprocessing: ColumnTransformer with numeric SimpleImputer(median) + StandardScaler, categorical SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore').
- Estimator: RandomForestClassifier (n_estimators=150, random_state=42)
- Trained with an 80/20 train/test split (see `src/models/train_model.py`).

## Evaluation Metrics (approximate)
See `METRICS.md` for the latest training metrics. Example values from the most recent run:
- Accuracy: 0.85
- Precision, Recall, F1: See METRICS.md

## Ethical considerations & caveats
- Data bias: The original dataset is not a demographically representative sample of a global population. Performance may degrade on other populations.
- Clinical risk: This model is not a substitute for medical expertise.
- Input validation: The app fills missing model-required features using medians/modes from the processed dataset; this is a pragmatic default, not a clinical imputation.

## How to reproduce
1. Ensure dependencies installed: `pip install -r requirements.txt`
2. Run training: `python src/models/train_model.py --input data/processed/heart_disease_processed.csv --output models`
3. Run tests: `pytest -q`

## Contact
For questions, open an issue or contact the repository owner.
