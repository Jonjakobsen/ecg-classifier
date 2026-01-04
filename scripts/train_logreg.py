from ecg_classifier.models import ECGLogReg
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score


def train_model():
    model = ECGLogReg()

    ROOT = Path(__file__).resolve().parents[1]

    # Load data
    X_train = np.load(ROOT / "x_train.npy")
    y_train = np.load(ROOT / "y_train_bin.npy")

    X_test = np.load(ROOT / "x_test.npy")
    y_test = np.load(ROOT / "y_test_bin.npy")

    


    # Train
    model.train(X_train, y_train)

    # Evaluate
    y_pred = []
    y_prob = []

    for x in X_test:
        out = model.predict(x)
        y_pred.append(out["class"])
        y_prob.append(out["prob_not_normal"])

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Test accuracy: {acc:.3f}")
    print(f"Test ROC-AUC:  {auc:.3f}")

    # Save model
    model.save(ROOT / "src" / "ecg_classifier" / "artifacts" / "logreg.joblib")


if __name__ == "__main__":
    train_model()
