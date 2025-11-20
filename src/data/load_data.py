import pandas as pd
from pathlib import Path


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame.

    Args:
        path: Path to CSV file.
    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df
