# src/ecg_classifier/models/gru.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ECGGRU(nn.Module):
    def __init__(self, n_leads=12, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_leads,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

    # ---------- API ----------

    def predict(self, signal: np.ndarray) -> dict:
        """
        signal: (n_samples, 12)
        """
        self.eval()
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prob = self(x).item()

        cls = int(prob >= 0.5)
        
        confidence = prob if cls == 1 else 1 - prob

        return {
            "class": cls,
            "confidence": float(confidence),
            "prob_not_normal": float(prob),
        }

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
