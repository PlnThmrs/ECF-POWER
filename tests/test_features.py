import pandas as pd
from ml_housing.preprocessing import validate_columns

def test_validate_columns_returns_bool():
    df = pd.DataFrame({"age": [50], "bmi": [0.06], "bp": [0.02]})
    assert validate_columns(df) is True