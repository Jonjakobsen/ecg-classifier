
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class ECGLogReg:
    def __init__(self):
        self.model = LogisticRegression(max_iter=100)

    def featurize(self, signal: np.ndarray) -> np.ndarray:
        """
        Very simple handcrafted features.
        """
        return np.array([
            signal.mean(),
            signal.std(),
            signal.min(),
            signal.max(),
        ])

    def train(self, signals: list[np.ndarray], labels: list[int]) -> None:
        X = np.vstack([self.featurize(s) for s in signals])
        y = np.array(labels)
        self.model.fit(X, y)

    def predict(self, signal: np.ndarray) -> dict:
        x = self.featurize(signal).reshape(1, -1)
        prob = self.model.predict_proba(x)[0]
        cls = int(prob.argmax())

        return {
            "class": cls,
            "confidence": float(prob[cls]),
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)

