from pathlib import Path

from ml_conso.data import load_df_preprocessed

TEST_DATA = Path("tests/data/df_preprocessed_sample.csv")


def test_load_conso_data_not_empty():
    df = load_df_preprocessed(path=TEST_DATA)
    assert not df.empty


def test_target_column_exists():
    df = load_df_preprocessed(path=TEST_DATA)
    assert "evo_conso_scaled" in df.columns