from ecg_classifier.io import load_ecg
from ecg_classifier.models import ECGLogReg
from ecg_classifier.models import ECGGRU
from importlib.resources import files




LOGREG_PATH = files("ecg_classifier") / "artifacts" / "logreg.joblib"
GRU_PATH = files("ecg_classifier") / "artifacts" / "gru.pt"

# Load modeller én gang
_LOGREG_MODEL = ECGLogReg()
_LOGREG_MODEL.load(LOGREG_PATH)

_GRU_MODEL = ECGGRU()
_GRU_MODEL.load(GRU_PATH)


label_map = {0: "NORMAL", 1: "NOT NORMAL"}


def run_inference(path: str, fmt: str = "wfdb", model_type: str ="logreg") -> dict:
    signal = load_ecg(path, fmt)


    if model_type == "logreg":
        pred = _LOGREG_MODEL.predict(signal)

    elif model_type == "gru":
        pred = _GRU_MODEL.predict(signal)

        

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose 'logreg' or 'gru'."
        )



    return {
        "label": label_map[pred["class"]],
        "confidence": pred["confidence"],
    }


