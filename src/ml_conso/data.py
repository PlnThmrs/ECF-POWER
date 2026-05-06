from pathlib import Path

import pandas as pd


def load_df_preprocessed() -> pd.DataFrame:
    """Charge le dataset prétraité depuis le dataset principal ou un dataset sample.

    Le dataset principal est df_preprocessed.csv. En CI, si ce fichier n'est pas présent,
    on utilise df_preprocessed_sample.csv pour permettre l'exécution des tests.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    primary_path = data_dir / "df_preprocessed.csv"
    sample_path = data_dir / "df_preprocessed_sample.csv"

    if primary_path.exists():
        return pd.read_csv(primary_path)

    if sample_path.exists():
        return pd.read_csv(sample_path)

    raise FileNotFoundError(
        f"Aucun dataset trouvé, attendu {primary_path} ou {sample_path}"
    )
