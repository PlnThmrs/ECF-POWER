from pathlib import Path

from ml_conso.data import load_df_preprocessed
from ml_conso.evaluate import evaluate_model
from ml_conso.features import split_features_target, split_train_test
from ml_conso.train import train_model

TEST_DATA = Path("tests/data/df_preprocessed_sample.csv")


def test_evaluate_model():
    df = load_df_preprocessed(path=TEST_DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Check that metrics are returned
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["r2"], float)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1
