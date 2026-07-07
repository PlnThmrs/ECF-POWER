# 1-Import de la data
# Import de path pour manipuler les chemins de fichiers
from pathlib import Path

import pandas as pd

# On récupère par défaut la racine du projet (niveau parent +2)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILENAME = "Merge_conso_meteo_soleil_090426.csv"


# Charge les données depuis le fichier d'origine
def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Charge le dataset local.
    """
    # Chemin du fichier non fourni: on prend le chemin par défaut
    if csv_path is None:
        csv_path = PROJECT_ROOT / DATA_FILENAME
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        # levée d'erreur si le fichier n'existe pas
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    # transformation du fichier en dataframe
    df = pd.read_csv(csv_path)

    return df
