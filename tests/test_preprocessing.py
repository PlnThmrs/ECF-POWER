import pandas as pd

from ml_conso.preprocessing import remove_missing_values


def test_remove_missing_values():
    df = pd.DataFrame({"A": [1, 2, None, 4], "B": [None, 2, 3, 4], "C": [1, 2, 3, 4]})

    cleaned_df = remove_missing_values(df)

    assert len(cleaned_df) == 2  # Rows 1 and 3 have no NaN
    assert not cleaned_df.isnull().any().any()
