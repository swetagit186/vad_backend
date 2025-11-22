# preprocessing.py
import os
import io
import base64
from typing import List, Dict, Tuple

import numpy as np
import pydicom
from PIL import Image

IMG_SIZE = 224  # must match your training


def collect_dicom_paths_and_metadata(root: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Walk 'root' and collect all .dcm files.
    Sort by InstanceNumber if available.
    Also extract basic patient metadata from the first readable DICOM.
    """
    dicom_paths: List[str] = []
    first_ds = None

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                full = os.path.join(dirpath, fname)
                dicom_paths.append(full)

    if not dicom_paths:
        raise ValueError("No DICOM files found in uploaded folder.")

    # sort by InstanceNumber (read header only)
    def sort_key(path: str) -> int:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    dicom_paths.sort(key=sort_key)

    # read first dataset for metadata
    for p in dicom_paths:
        try:
            first_ds = pydicom.dcmread(p, force=True)
            break
        except Exception:
            continue

    if first_ds is None:
        raise ValueError("Could not read any DICOM for metadata.")

    metadata = {
        "patient_name": str(getattr(first_ds, "PatientName", "Unknown")),
        "patient_id": str(getattr(first_ds, "PatientID", "Unknown")),
        "patient_sex": str(getattr(first_ds, "PatientSex", "Unknown")),
        "patient_age": str(getattr(first_ds, "PatientAge", "Unknown")),
        "study_date": str(getattr(first_ds, "StudyDate", "Unknown")),
        "modality": str(getattr(first_ds, "Modality", "Unknown")),
    }

    return dicom_paths, metadata


def load_and_normalize_slice(path: str) -> np.ndarray:
    """
    Load a single DICOM slice and return a normalized uint8 image (H, W).
    Uses percentile-based windowing for robustness.
    """
    ds = pydicom.dcmread(path, force=True)

    if not hasattr(ds, "PixelData"):
        raise ValueError(f"No pixel data in DICOM: {path}")

    arr = ds.pixel_array.astype(np.float32)

    # robust windowing
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 <= p1:
        p99 = p1 + 1.0

    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1 + 1e-6)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    return arr


def resize_to_model(img_arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize 2D uint8 slice to (size, size).
    """
    img = Image.fromarray(img_arr)
    img = img.resize((size, size))
    return np.array(img, dtype=np.uint8)


def middle_slice_base64(dicom_paths: List[str]) -> str:
    """
    Return base64 PNG of the middle slice, for frontend preview.
    """
    mid_idx = len(dicom_paths) // 2
    mid_path = dicom_paths[mid_idx]
    arr = load_and_normalize_slice(mid_path)
    arr = resize_to_model(arr)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
