from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "type": "L",
        "air_temperature": 300,
        "process_temperature": 310,
        "rotational_speed": 1400,
        "torque": 55,
        "tool_wear": 200
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert "prediction" in data
    assert "failure_probability" in data
    assert "risk_label" in data
    assert data["risk_label"] in ["Low Risk", "Medium Risk", "High Risk"]
