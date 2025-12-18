from ecg_classifier.models import ECGLogReg
import numpy as np

def train_dummy_model():
    model = ECGLogReg()

    # Fake data for now
    signals = [
        np.array([0.1, 0.2, 0.1, 0.2]),
        np.array([1.0, 1.1, 0.9, 1.2]),
    ]
    labels = np.array([0, 1])

    model.train(signals, labels)
    model.save("logreg.joblib")


if __name__ == "__main__":
    train_dummy_model()
