# preprocessing.py
import os
import io
import base64
from typing import List, Tuple, Dict

import numpy as np
import pydicom
from PIL import Image


IMG_SIZE = 224


def load_dicom_slices(folder: str) -> Tuple[List[np.ndarray], Dict[str, str]]:
    """
    Load valid MRI images from a patient folder.

    Accepts shapes:
      - (H, W)
      - (1, H, W) -> squeeze
      - (N, H, W) -> multi-frame, use each frame
      - (H, W, 3) -> RGB, convert to grayscale

    Returns:
      slices: list of 2D float32 arrays
      metadata: dict of basic patient info
    """
    slices: List[np.ndarray] = []
    ds_list = []

    for f in os.listdir(folder):
        if not f.lower().endswith(".dcm"):
            continue

        fpath = os.path.join(folder, f)
        try:
            ds = pydicom.dcmread(fpath, force=True)

            if "PixelData" not in ds:
                continue

            arr = ds.pixel_array

            # (H, W)
            if arr.ndim == 2:
                slices.append(arr.astype(np.float32))
                ds_list.append(ds)

            # (1, H, W) -> squeeze
            elif arr.ndim == 3 and arr.shape[0] == 1:
                slices.append(arr[0].astype(np.float32))
                ds_list.append(ds)

            # (N, H, W) multi-frame
            elif arr.ndim == 3 and arr.shape[0] > 1:
                for i in range(arr.shape[0]):
                    slices.append(arr[i].astype(np.float32))
                    ds_list.append(ds)

            # (H, W, 3) RGB -> grayscale
            elif arr.ndim == 3 and arr.shape[2] == 3:
                gray = arr.mean(axis=2).astype(np.float32)
                slices.append(gray)
                ds_list.append(ds)

            else:
                continue

        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError(f"No valid MRI images found in folder: {folder}")

    first = ds_list[0]
    metadata = {
        "patient_name": str(getattr(first, "PatientName", "Unknown")),
        "patient_id": str(getattr(first, "PatientID", "Unknown")),
        "patient_sex": str(getattr(first, "PatientSex", "Unknown")),
        "patient_age": str(getattr(first, "PatientAge", "Unknown")),
        "study_date": str(getattr(first, "StudyDate", "Unknown")),
        "modality": str(getattr(first, "Modality", "Unknown")),
    }

    return slices, metadata


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize, then scale to 0–255 uint8."""
    mean = arr.mean()
    std = arr.std() or 1e-6
    norm = (arr - mean) / std

    minv, maxv = norm.min(), norm.max()
    if maxv - minv < 1e-6:
        maxv = minv + 1e-6

    norm = (norm - minv) / (maxv - minv)
    norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def resize_slice(np_img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    img = Image.fromarray(np_img)
    img = img.resize((size, size))
    return np.array(img, dtype=np.uint8)


def create_25d_tensors(slices: List[np.ndarray]) -> List["np.ndarray"]:
    """
    Build 2.5D stacks: [i-1, i, i+1] -> (3, H, W), later converted to torch.Tensor.
    Returns list of np arrays (3, IMG_SIZE, IMG_SIZE).
    """
    if len(slices) < 3:
        return []

    stacks: List[np.ndarray] = []

    for i in range(1, len(slices) - 1):
        s_prev = resize_slice(normalize_slice(slices[i - 1]))
        s_mid = resize_slice(normalize_slice(slices[i]))
        s_next = resize_slice(normalize_slice(slices[i + 1]))

        stack = np.stack([s_prev, s_mid, s_next], axis=0)  # (3, H, W)
        stacks.append(stack)

    return stacks


def get_middle_slice_base64(slices: List[np.ndarray]) -> str:
    """Return base64 PNG of the middle normalized slice."""
    mid_idx = len(slices) // 2
    norm = normalize_slice(slices[mid_idx])
    img = Image.fromarray(norm)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
