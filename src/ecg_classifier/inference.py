from ecg_classifier.data import load_ecg_csv


def run_inference(input_path: str) -> dict:
    signal = load_ecg_csv(input_path)

    return {
        "status": "ok",
        "n_samples": int(signal.shape[0]),
        "message": "Data loading successful, model not yet implemented"
    }
