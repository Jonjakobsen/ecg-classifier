from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ECGLogReg:
    def __init__(self, fs: int = 100):
        self.fs = fs
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced"))
                ])
    # ---------- Feature engineering ----------

    def _time_features(self, x: np.ndarray) -> list[float]:
        return [
            x.mean(),
            x.std(),
            x.min(),
            x.max(),
            np.ptp(x),                    # peak-to-peak
            np.sqrt(np.mean(x ** 2)),     # RMS
        ]

    def _freq_features(self, x: np.ndarray) -> list[float]:
        freqs, psd = welch(x, fs=self.fs, nperseg=256)

        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            return psd[mask].sum()

        return [
            psd.sum(),                   # total power
            band_power(0.5, 4),
            band_power(4, 8),
            band_power(8, 20),
            band_power(20, 40),
        ]

    def featurize(self, signal: np.ndarray) -> np.ndarray:
        """
        signal shape: (n_samples, 12)
        returns: (n_features,)
        """
        features = []

        for lead in range(signal.shape[1]):
            x = signal[:, lead]

            features.extend(self._time_features(x))
            features.extend(self._freq_features(x))

        return np.array(features, dtype=np.float32)

    # ---------- Training / inference ----------

    def train(self, signals: np.ndarray, labels: np.ndarray) -> None:
        X = np.vstack([self.featurize(s) for s in signals])
        y = labels.astype(int)

        self.model.fit(X, y)

    def predict(self, signal: np.ndarray) -> dict:
        x = self.featurize(signal).reshape(1, -1)
        prob = self.model.predict_proba(x)[0]
        cls = int(prob.argmax())

        return {
            "class": cls,
            "confidence": float(prob[cls]),
            "prob_not_normal": float(prob[1]),
        }

    # ---------- Persistence ----------

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "fs": self.fs,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.fs = data["fs"]
