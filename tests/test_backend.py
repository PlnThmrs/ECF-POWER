from fastapi.testclient import TestClient

import backend.app as backend_app


# On remplace le modèle AVANT de créer le client
class FakeModel:
    def predict(self, df):
        return [42.0]


# On remplace le modèle AVANT d'initialiser l'app
backend_app.model = FakeModel()
app = backend_app.app
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch):
    # Fake modèle
    class FakeModel:
        def predict(self, df):
            return [42.0]

    # On remplace le vrai modèle par le fake
    monkeypatch.setattr("backend.app.model", FakeModel())

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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 42.0}
