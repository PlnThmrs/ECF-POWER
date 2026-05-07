from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_df_preprocessed(path: Path | None = None) -> pd.DataFrame:
    """Charge le dataset prétraité depuis df_preprocessed.csv ou un chemin fourni."""
    if path is None:
        path = PROJECT_ROOT / "data" / "df_preprocessed.csv"
    return pd.read_csv(path)
