from ecg_classifier.io import load_ecg
from ecg_classifier.models import ECGLogReg
from importlib.resources import files




MODEL_PATH = files("ecg_classifier") / "artifacts" / "logreg.joblib"


def run_inference(path: str, fmt: str = "wfdb") -> dict:
    signal = load_ecg(path, fmt)

    model = ECGLogReg()
    model.load(MODEL_PATH)

    pred = model.predict(signal)

    label_map = {0: "NORMAL", 1: "NOT NORMAL"}

    return {
        "label": label_map[pred["class"]],
        "confidence": pred["confidence"],
    }


