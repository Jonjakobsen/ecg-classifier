# ecg_classifier/io/load.py
from pathlib import Path
import numpy as np

from .csv_loader import load_csv
from .wfdb_loader import load_wfdb


def load_ecg(path: str | Path, fmt: str) -> np.ndarray:
    fmt = fmt.lower()

    if fmt == "csv":
        return load_csv(path)

    if fmt == "wfdb":
        return load_wfdb(path)

    raise ValueError(f"Unknown ECG format: {fmt}")
