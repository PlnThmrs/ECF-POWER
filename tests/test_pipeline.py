from pathlib import Path

from ml_conso.pipeline import run_pipeline

TEST_DATA = Path("tests/data/df_preprocessed_sample.csv")


def test_pipeline_returns_metrics(tmp_path):
    metrics = run_pipeline(artifacts_dir=str(tmp_path), data_path=TEST_DATA)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert -1 <= metrics["r2"] <= 1
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
