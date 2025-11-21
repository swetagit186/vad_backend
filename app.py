# app.py
import os
import shutil
import tempfile
import zipfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model, run_model_on_folder

app = FastAPI(title="MRI VAD/Alzheimer/Normal Classifier (ResNet18 2.5D)")

# CORS (later restrict origins to your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL = load_model(model_dir="models", filename="resnet18_25d_vad.pth")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}



@app.get("/")
def root() -> Dict:
    return {"message": "MRI Dementia Classifier API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Accept a ZIP file containing ONE patient's DICOM images.
    Returns:
      - prediction: Normal / Alzheimer / Vascular Dementia
      - confidence: probability of predicted class
      - patient_metadata: basic DICOM fields
      - middle_slice_base64: PNG base64 of 1 slice for display
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "upload.zip")

    try:
        # Save uploaded zip
        with open(zip_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Find first folder that contains DICOM files
        dicom_folder = None
        for root, dirs, files in os.walk(tmp_dir):
            dcm_files = [f for f in files if f.lower().endswith(".dcm")]
            if dcm_files:
                dicom_folder = root
                break

        if dicom_folder is None:
            raise HTTPException(status_code=400, detail="No DICOM files found in the uploaded ZIP.")

        result = run_model_on_folder(dicom_folder, MODEL)

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "patient_metadata": result["patient_metadata"],
            "middle_slice_base64": result["middle_slice_base64"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
