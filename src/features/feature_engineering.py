import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Example feature engineering — adapt to your dataset.

    For Cleveland UCI heart disease dataset you might create interaction
    features or bucketize age, cholesterol, etc.
    """
    df = df.copy()
    if 'age' in df.columns:
        df['age_bucket'] = pd.cut(df['age'], bins=[0,40,55,70,120], labels=['<40','40-55','55-70','70+'])
    return df
