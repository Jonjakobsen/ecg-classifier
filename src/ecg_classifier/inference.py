from ecg_classifier.data import load_ecg_csv
from ecg_classifier.models import ECGLogReg
from importlib.resources import files



MODEL_PATH = files("ecg_classifier") / "artifacts" / "logreg.joblib"


def run_inference(input_path: str) -> dict:
    signal = load_ecg_csv(input_path)

    model = ECGLogReg()
    model.load(MODEL_PATH)

    return model.predict(signal)


