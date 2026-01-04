from ecg_classifier.io import load_ecg
from ecg_classifier.models import ECGLogReg
from ecg_classifier.models import ECGGRU
from importlib.resources import files




LOGREG_PATH = files("ecg_classifier") / "artifacts" / "logreg.joblib"
GRU_PATH = files("ecg_classifier") / "artifacts" / "gru.pt"


def run_inference(path: str, fmt: str = "wfdb", model_type: str ="logreg") -> dict:
    signal = load_ecg(path, fmt)


    if model_type == "logreg":

        model = ECGLogReg()
        model.load(LOGREG_PATH)

        pred = model.predict(signal)

        label_map = {0: "NORMAL", 1: "NOT NORMAL"}
    
    elif model_type == "gru":
        
        model = ECGGRU()
        model.load(GRU_PATH)

        pred = model.predict(signal)

        label_map = {0: "NORMAL", 1: "NOT NORMAL"}

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose 'logreg' or 'gru'."
        )



    return {
        "label": label_map[pred["class"]],
        "confidence": pred["confidence"],
    }


