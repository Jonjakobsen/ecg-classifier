from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
from tempfile import NamedTemporaryFile, TemporaryDirectory
import shutil
from importlib.resources import files
import os

from ecg_classifier.inference import run_inference


app = FastAPI(
    title="ECG Classifier API",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    format: str = "wfdb",
    model_type: str = "logreg",
    files: list[UploadFile] = File(...),
):
    """
    - format=csv  → upload 1 CSV file
    - format=wfdb → upload both .hea and .dat WFDB files for a single record
    """
    try:
        if format == "csv":
            if len(files) != 1:
                raise ValueError("CSV inference requires exactly one file")

            with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                shutil.copyfileobj(files[0].file, tmp)
                tmp_path = tmp.name

            return run_inference(
                path=tmp_path,
                fmt="csv",
                model_type=model_type,
            )

        elif format == "wfdb":
            with TemporaryDirectory() as tmpdir:
                for f in files:
                    dst = os.path.join(tmpdir, f.filename)
                    with open(dst, "wb") as out:
                        shutil.copyfileobj(f.file, out)

                # Udled record name (fra .hea)
                hea_files = [f for f in os.listdir(tmpdir) if f.endswith(".hea")]
                if len(hea_files) != 1:
                    raise ValueError("Exactly one .hea file must be uploaded")

                record_name = hea_files[0].replace(".hea", "")
                record_path = os.path.join(tmpdir, record_name)

                return run_inference(
                    path=record_path,
                    fmt="wfdb",
                    model_type=model_type,
                )

        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/demo")
def demo(
    model_type: str = "logreg",
    format: str = "wfdb",
):
    """
    Run inference on bundled demo ECG data.
    """
    try:
        demo_root = files("ecg_classifier") / "demo"

        if format == "wfdb":
            demo_path = demo_root / "wfdb" / "demo_wfdb"
        elif format == "csv":
            demo_path = demo_root / "test_ecg_12lead.csv"
        else:
            raise ValueError(f"Unsupported demo format: {format}")

        return run_inference(
            path=str(demo_path),
            fmt=format,
            model_type=model_type,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
