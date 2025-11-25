import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

from src.data.preprocess import basic_clean


TARGET_COLUMN_CANDIDATES = ['target', 'num']


def _detect_target(df: pd.DataFrame) -> str | None:
    for c in TARGET_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    return None


def train(input_csv: str | Path,
          output_dir: str | Path = 'models',
          test_size: float = 0.2,
          random_state: int = 42,
          n_estimators: int = 150) -> dict:
    """Train a RandomForest pipeline on the heart disease dataset.

    Parameters
    ----------
    input_csv : str | Path
        Path to raw or preprocessed CSV.
    output_dir : str | Path, default 'models'
        Directory where model and metrics are saved.
    test_size : float, default 0.2
        Fraction of data for test split.
    random_state : int, default 42
        Random seed for reproducibility.
    n_estimators : int, default 150
        Number of trees for RandomForestClassifier.

    Returns
    -------
    dict
        Metrics dictionary including accuracy, roc_auc and classification report.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Loading data from %s', input_csv)
    df = pd.read_csv(input_csv)

    # Basic cleaning from our helpers
    df = basic_clean(df)

    target_col = _detect_target(df)
    if target_col is None:
        raise ValueError('No target column found. Expected one of: ' + ','.join(TARGET_COLUMN_CANDIDATES))

    # For UCI 'num' target convert to binary: 0 -> 0, >0 -> 1
    if target_col == 'num':
        df['target'] = (df['num'].astype(float) > 0).astype(int)
    else:
        df['target'] = df[target_col].astype(int)

    # Remove original raw target columns from features to avoid leakage
    # (e.g., remove 'num' if present so the pipeline doesn't expect it as an input)
    for c in TARGET_COLUMN_CANDIDATES:
        if c in df.columns and c != 'target':
            df = df.drop(columns=[c])

    # Drop identifier columns if present
    drop_cols = [c for c in ['id', 'dataset'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Transformers
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
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info('Training model (n_estimators=%d, random_state=%d)', n_estimators, random_state)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # For binary RF, predict_proba available; ensure we handle cases where only 2 classes
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        roc_auc = None
    report = classification_report(y_test, preds, output_dict=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / 'rf_pipeline.joblib'
    joblib.dump(clf, model_path)

    # Persist metrics as JSON for easier downstream parsing
    metrics = {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict(),
        'report': report,
        'features': X.columns.tolist()
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    # Also store plain text for backward compatibility (optional)
    (out_dir / 'metrics.txt').write_text(f"accuracy={acc}\nroc_auc={roc_auc}\n")
    # Save feature list separately (helps inference services)
    (out_dir / 'features.txt').write_text('\n'.join(X.columns))

    # Try to extract feature importances with corresponding transformed feature names
    try:
        import numpy as np

        clf_steps = clf.named_steps
        classifier = clf_steps.get('classifier')
        preprocessor = clf_steps.get('preprocessor')
        importances = None
        feature_names = []
        if classifier is not None and hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            # Attempt to get transformed feature names (sklearn >=1.0)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                # Fallback: numeric columns + onehot names
                feature_names = list(numeric_cols)
                for cat in categorical_cols:
                    try:
                        ohe = preprocessor.named_transformers_['cat'].named_steps.get('onehot')
                        names = list(ohe.get_feature_names_out([cat]))
                        feature_names.extend(names)
                    except Exception:
                        # Last resort: use categorical column name
                        feature_names.append(cat)

        if importances is not None and len(importances) == len(feature_names):
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=False)
            fi_df.to_csv(out_dir / 'feature_importances.csv', index=False)
            logging.info('Wrote feature importances to %s', out_dir / 'feature_importances.csv')
    except Exception as e:
        logging.warning('Could not write feature importances: %s', e)

    # Persist model metadata (params, timestamp, git commit)
    metadata = {
        'model_path': str(model_path),
        'n_estimators': n_estimators,
        'random_state': random_state,
        'test_size': test_size,
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'metrics': metrics
    }
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(PROJECT_ROOT)).decode().strip()
        metadata['git_commit'] = commit
    except Exception:
        metadata['git_commit'] = None

    (out_dir / 'model_metadata.json').write_text(json.dumps(metadata, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {acc:.4f}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='CSV file with data (raw or processed)')
    parser.add_argument('--output', dest='output', default='models', help='Directory to save model')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2, help='Test split size (default=0.2)')
    parser.add_argument('--random-state', dest='random_state', type=int, default=42, help='Random seed (default=42)')
    parser.add_argument('--n-estimators', dest='n_estimators', type=int, default=150, help='Number of trees (default=150)')
    args = parser.parse_args()
    train(args.input, args.output, test_size=args.test_size, random_state=args.random_state, n_estimators=args.n_estimators)
