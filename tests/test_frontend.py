import pytest

from frontend.streamlit_app import call_api


def test_call_api_success(requests_mock):
    payload = {
        "MoyenneConso": 3.5,
        "AgeBat": 20,
        "SurfBat": 500,
        "NmbPieces": 4,
        "TypeChauffage": "Électricité",
        "NbOccupants": 3,
        "ZoneClimatique": "H1",
        "EtiquetteDPE": "C",
    }

    # Mock de la réponse API
    requests_mock.post(
        "http://127.0.0.1:8000/predict", json={"prediction": 42.0}, status_code=200
    )

    prediction = call_api(payload)

    assert prediction == 42.0


def test_call_api_error(requests_mock):
    payload = {"MoyenneConso": 3.5}

    requests_mock.post("http://127.0.0.1:8000/predict", status_code=500)

    with pytest.raises(Exception):
        call_api(payload)
