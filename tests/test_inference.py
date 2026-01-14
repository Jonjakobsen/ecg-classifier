import pytest
from importlib.resources import files
from ecg_classifier.inference import run_inference


@pytest.mark.parametrize(
    "fmt, model_type, file_path",
    [
        ("csv",  "logreg", "test_ecg_12lead.csv"),
        ("csv",  "gru",    "test_ecg_12lead.csv"),
        ("wfdb", "logreg", "wfdb/demo_wfdb"),
        ("wfdb", "gru",    "wfdb/demo_wfdb"),
    ],
)
def test_inference_runs(fmt, model_type, file_path):
    file_root = files("ecg_classifier") / "demo" / file_path

    result = run_inference(
        path=str(file_root),
        fmt=fmt,
        model_type=model_type,
    )

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

