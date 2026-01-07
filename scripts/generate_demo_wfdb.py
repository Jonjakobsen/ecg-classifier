from pathlib import Path
import numpy as np
import wfdb


def main():
    out_dir = Path("src/ecg_classifier/demo/wfdb")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 100
    duration = 10
    n_samples = fs * duration
    n_leads = 12

    t = np.linspace(0, duration, n_samples)

    signal = np.stack(
        [
            0.5 * np.sin(2 * np.pi * 1.2 * t)
            + 0.05 * np.random.randn(n_samples)
            for _ in range(n_leads)
        ],
        axis=1,
    )

    wfdb.wrsamp(
        record_name="demo_wfdb",
        fs=fs,
        units=["mV"] * n_leads,
        sig_name=[f"lead_{i+1}" for i in range(n_leads)],
        p_signal=signal,
        fmt=["16"] * n_leads,
    )


if __name__ == "__main__":
    main()
