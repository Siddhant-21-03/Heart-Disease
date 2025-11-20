import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values.

    This is intentionally generic — adapt to your dataset.
    """
    df = df.copy()
    df = df.drop_duplicates()

    # Replace common placeholders for missing values
    df.replace(['?', 'NA', 'na', ''], pd.NA, inplace=True)

    # Impute numeric columns with median, categorical with mode
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    for c in obj_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else '')

    return df


def train_test_split_save(df: pd.DataFrame, target_col: str, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(out_dir / 'train.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)

    return train, test


def scale_numeric(df: pd.DataFrame, numeric_cols: list[str]):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler
