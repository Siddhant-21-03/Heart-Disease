"""Evaluate the saved pipeline with cross-validation and produce plots.
Saves outputs to `reports/` and prints summary metrics.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'heart_disease_processed.csv'
REPORT_DIR = PROJECT_ROOT / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise SystemExit(f"Processed data not found at {DATA_PATH}. Run preprocessing first.")

print('Loading data...')
df = pd.read_csv(DATA_PATH)

# detect target
TARGETS = ['target', 'num']
target_col = None
for t in TARGETS:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    raise SystemExit('No target column found in processed CSV')

# Ensure binary target called 'target'
if target_col == 'num':
    df['target'] = (df['num'].astype(float) > 0).astype(int)
else:
    df['target'] = df[target_col].astype(int)

# drop original targets and ids
drop = [c for c in ['id','dataset'] if c in df.columns]
df = df.drop(columns=drop)
for c in ['num']:
    if c in df.columns and c != 'target':
        df = df.drop(columns=[c])

X = df.drop(columns=['target'])
y = df['target']

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Running cross-validated predictions...')
# predicted labels
y_pred = cross_val_predict(clf, X, y, cv=cv, method='predict')
# predicted probabilities for positive class
try:
    y_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
except Exception:
    y_proba = None

acc = accuracy_score(y, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
conf = confusion_matrix(y, y_pred)
roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None

metrics = {
    'accuracy': float(acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'roc_auc': float(roc_auc) if roc_auc is not None else None,
    'confusion_matrix': conf.tolist()
}

print('Metrics:')
print(json.dumps(metrics, indent=2))

# save metrics json
(REPORT_DIR / 'cv_metrics.json').write_text(json.dumps(metrics, indent=2))

# ROC curve
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (5-fold CV)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'roc_curve.png')
    plt.close()

# Calibration plot
if y_proba is not None:
    prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration plot (5-fold CV)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'calibration.png')
    plt.close()

# Confusion matrix heatmap
try:
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (5-fold CV)')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'confusion_matrix.png')
    plt.close()
except Exception:
    # fallback: save numeric matrix
    (REPORT_DIR / 'confusion_matrix.txt').write_text(str(conf))

# Update METRICS.md with CV results (append)
metrics_md = Path(PROJECT_ROOT / 'METRICS.md')
old = ''
if metrics_md.exists():
    old = metrics_md.read_text()

new_section = '\n\n## Cross-validated metrics (5-fold)\n' + json.dumps(metrics, indent=2)
metrics_md.write_text(old + new_section)

print('Saved reports to', REPORT_DIR)
print('Updated METRICS.md')
