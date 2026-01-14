from fastapi.testclient import TestClient
from ecg_classifier.api import app
import pytest
from importlib.resources import files

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}



@pytest.mark.parametrize(
    "fmt, model_type",
    [
        ("csv",  "logreg"),
        ("csv",  "gru"),
        ("wfdb", "logreg"),
        ("wfdb", "gru"),
    ],
)
def test_demo_endpoint(fmt, model_type):
    response = client.get(
        "/demo",
        params={"format": fmt, "model_type": model_type},
    )

    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_csv():
    csv_path = files("ecg_classifier") / "demo" / "test_ecg_12lead.csv"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/predict",
            params={"format": "csv", "model_type": "logreg"},
            files={"files": ("test.csv", f, "text/csv")},
        )

    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_wfdb():
    wfdb_root = files("ecg_classifier") / "demo" / "wfdb"

    hea = wfdb_root / "demo_wfdb.hea"
    dat = wfdb_root / "demo_wfdb.dat"

    with open(hea, "rb") as f1, open(dat, "rb") as f2:
        response = client.post(
            "/predict",
            params={"format": "wfdb", "model_type": "logreg"},
            files=[
                ("files", ("demo_wfdb.hea", f1, "application/octet-stream")),
                ("files", ("demo_wfdb.dat", f2, "application/octet-stream")),
            ],
        )

    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert 0.0 <= data["confidence"] <= 1.0
