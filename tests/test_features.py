from pathlib import Path

from common.features import split_features_target, split_train_test
from training.data import load_df_preprocessed

TEST_DATA = Path("tests/data/df_preprocessed_sample.csv")


def test_split_features_target():
    df = load_df_preprocessed(path=TEST_DATA)
    X, y = split_features_target(df)

    assert "evo_conso_scaled" not in X.columns
    assert len(X) == len(y)


def test_split_train_test():
    df = load_df_preprocessed(path=TEST_DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
