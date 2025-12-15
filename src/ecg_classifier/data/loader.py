from pathlib import Path
import numpy as np


def load_ecg_csv(path: str | Path) -> np.ndarray:
    """
    Load a single-lead ECG signal from a CSV file.

    Expected format:
    One column, no header, raw signal values.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")

    signal = np.loadtxt(path, delimiter=",")

    if signal.ndim != 1:
        raise ValueError("ECG signal must be 1D")

    return signal
