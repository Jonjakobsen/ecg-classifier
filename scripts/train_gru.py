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
    X_train = np.load(ROOT / "x_train.npy")
    y_train = np.load(ROOT / "y_train_bin.npy")

    X_test = np.load(ROOT / "x_test.npy")
    y_test = np.load(ROOT / "y_test_bin.npy")

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Dataset / loader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Model
    model = ECGGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # -------- Training --------
    model.train()
    for epoch in range(10):
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/10 - loss: {loss.item():.4f}")

    # -------- Evaluation --------
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

    # Save
    model.save(
        ROOT / "src" / "ecg_classifier" / "artifacts" / "gru.pt"
    )


if __name__ == "__main__":
    train_model()
