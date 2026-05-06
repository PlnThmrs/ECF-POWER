import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ml_conso.pipeline import run_pipeline

if __name__ == "__main__":
    metrics = run_pipeline()
    print("Pipeline terminé avec succès.")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R2   : {metrics['r2']:.4f}")
