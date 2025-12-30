from pathlib import Path
import wfdb
import numpy as np


def load_wfdb(path: str | Path) -> np.ndarray:
    path = Path(path)

    hea = path.with_suffix(".hea")
    dat = path.with_suffix(".dat")

    if not hea.exists() or not dat.exists():
        raise FileNotFoundError(
            f"WFDB record not found (.hea/.dat missing): {path}"
        )

    signal, meta = wfdb.rdsamp(str(path))
    return signal
