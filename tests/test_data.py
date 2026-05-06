import sys
from pathlib import Path

from ml_conso.data import load_df_preprocessed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_load_conso_data_not_empty():
    df = load_df_preprocessed()
    assert not df.empty


def test_target_column_exists():
    df = load_df_preprocessed()
    assert "evo_conso_scaled" in df.columns
