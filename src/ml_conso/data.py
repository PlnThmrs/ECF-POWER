import pandas as pd
from pathlib import Path


def load_df_preprocessed() -> pd.DataFrame:
    """Charge le dataset prétraité depuis df_preprocessed.csv."""
    data_path = Path(__file__).parent.parent.parent / "data" / "df_preprocessed.csv"
    df = pd.read_csv(data_path)
    return df
