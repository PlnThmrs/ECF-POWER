import sys
from pathlib import Path

from ml_conso.data import load_df_preprocessed
from ml_conso.features import split_features_target, split_train_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))



def test_split_features_target():
    df = load_df_preprocessed()
    X, y = split_features_target(df)

    assert "evo_conso_scaled" not in X.columns
    assert len(X) == len(y)


def test_split_train_test():
    df = load_df_preprocessed()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)