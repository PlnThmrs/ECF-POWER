from pathlib import Path

from common.features import split_features_target, split_train_test
from training.data import load_df_preprocessed
from training.train import train_model

TEST_DATA = Path("tests/data/df_preprocessed_sample.csv")


def test_train_model():
    df = load_df_preprocessed(path=TEST_DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = train_model(X_train, y_train)

    # Check that a model is returned
    assert model is not None
    # Check that it has predict method
    assert hasattr(model, "predict")
