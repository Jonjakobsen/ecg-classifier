import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

from ecg_classifier.models.gru import ECGGRU


def train_model():
    ROOT = Path(__file__).resolve().parents[1]

    # Load data
    X_train = np.load(ROOT / "data" / "x_train.npy")
    y_train = np.load(ROOT / "data" / "y_train_bin.npy")

    X_val = np.load(ROOT / "data" / "x_val.npy")
    y_val = np.load(ROOT / "data" / "y_val_bin.npy")


    X_test = np.load(ROOT / "data" / "x_test.npy")
    y_test = np.load(ROOT / "data" / "y_test_bin.npy")

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Dataset / loader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Model
    model = ECGGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # -------- Early stopping setup --------
    max_epochs = 30
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0

    artifact_path = ROOT / "src" / "ecg_classifier" / "artifacts" / "gru.pt"

    # -------- Training --------
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).squeeze()  # probabilities
            val_loss = loss_fn(val_probs, y_val_t).item()
            val_preds = (val_probs >= 0.5).float()

        val_acc = (val_preds == y_val_t).float().mean().item()

        print(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.3f}"
        )

        # -------- Early stopping logic --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save(artifact_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    # -------- Final evaluation (best model) --------
    model.load(artifact_path)
    model.eval()

    y_pred = []
    y_prob = []

    with torch.no_grad():
        for x in X_test_t:
            out = model.predict(x.numpy())
            y_pred.append(out["class"])
            y_prob.append(out["prob_not_normal"])

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Test accuracy: {acc:.3f}")
    print(f"Test ROC-AUC:  {auc:.3f}")


if __name__ == "__main__":
    train_model()
