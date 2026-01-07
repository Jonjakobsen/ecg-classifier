from ecg_classifier.models import ECGLogReg
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score


def train_model():
    ROOT = Path(__file__).resolve().parents[1]

    # ---------------- Data ----------------
    X_train = np.load(ROOT / "data" / "x_train.npy")
    y_train = np.load(ROOT / "data" / "y_train_bin.npy")

    X_val = np.load(ROOT / "data" / "x_val.npy")
    y_val = np.load(ROOT / "data" / "y_val_bin.npy")

    X_test = np.load(ROOT / "data" / "x_test.npy")
    y_test = np.load(ROOT / "data" / "y_test_bin.npy")

    # ---------------- Model ----------------
    model = ECGLogReg()

    # ---------------- Train ----------------
    model.train(X_train, y_train)

    # ---------------- Validation ----------------
    y_val_pred = []
    y_val_prob = []

    for x in X_val:
        out = model.predict(x)
        y_val_pred.append(out["class"])
        y_val_prob.append(out["prob_not_normal"])

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)

    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Validation ROC-AUC:  {val_auc:.3f}")

    # ---------------- Save model ----------------
    artifact_path = (
        ROOT / "src" / "ecg_classifier" / "artifacts" / "logreg.joblib"
    )
    model.save(artifact_path)

    # ---------------- Test evaluation ----------------
    y_test_pred = []
    y_test_prob = []

    for x in X_test:
        out = model.predict(x)
        y_test_pred.append(out["class"])
        y_test_prob.append(out["prob_not_normal"])

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)

    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test ROC-AUC:  {test_auc:.3f}")


if __name__ == "__main__":
    train_model()
