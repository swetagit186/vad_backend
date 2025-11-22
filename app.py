# app.py
import os
import shutil
import tempfile
import zipfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model, run_patient_inference

app = FastAPI(
    title="MRI VAD / Alzheimer / Normal Classifier (Slice-level ResNet18)"
)

# Enable CORS (frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during development; tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL = load_model()


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Upload a ZIP containing a single patient's DICOM images.
    Returns:
      - prediction (class label)
      - confidence
      - per-class probabilities
      - number of slices used
      - patient metadata
      - base64 PNG preview of middle slice
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    tmp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded ZIP
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Extract ZIP
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")

        # Find the first folder containing DICOM files
        patient_root = None
        for root, _, files in os.walk(tmp_dir):
            if any(fname.lower().endswith(".dcm") for fname in files):
                patient_root = root
                break

        if patient_root is None:
            raise HTTPException(
                status_code=400,
                detail="No DICOM files found in uploaded ZIP.",
            )

        result = run_patient_inference(patient_root, MODEL)

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "num_slices_used": result["num_slices_used"],
            "patient_metadata": result["patient_metadata"],
            "preview_base64": result["preview_base64"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
