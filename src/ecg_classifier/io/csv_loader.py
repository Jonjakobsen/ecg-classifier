from pathlib import Path
import numpy as np

EXPECTED_SAMPLES = 1000
EXPECTED_LEADS = 12


def load_csv(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    signal = np.loadtxt(path, delimiter=",")

    if signal.ndim != 2:
        raise ValueError("CSV must be 2D (samples, leads)")

    if signal.shape != (EXPECTED_SAMPLES, EXPECTED_LEADS):
        raise ValueError(
            f"Expected shape {(EXPECTED_SAMPLES, EXPECTED_LEADS)}, "
            f"got {signal.shape}"
        )

    return signal.astype(np.float32)
