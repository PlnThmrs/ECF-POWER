from src.predict import predict_one
def test_predict_one_returns_number():
    sample = {
        "age": 0.05,
        "sex": 0.05,
        "bmi": 0.06,
    }
    result = predict_one(sample)
    assert isinstance(result, (int, float))