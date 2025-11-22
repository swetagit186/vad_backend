# inference.py
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from preprocessing import (
    collect_dicom_paths_and_metadata,
    load_and_normalize_slice,
    resize_to_model,
    middle_slice_base64,
)

DEVICE = torch.device("cpu")  # keep CPU for simplicity

# 👉 IMPORTANT: adjust order to match how you encoded labels during training
# Example assumption:
# 0 -> VAD, 1 -> Alzheimer, 2 -> Normal   (change if needed)
CLASS_NAMES = ["VAD", "Alzheimer", "Normal"]


def build_resnet18(num_classes: int = 3) -> torch.nn.Module:
    """
    Build a plain ResNet18 with fc replaced for num_classes.
    This matches a typical Jupyter training setup:
        model = resnet18(weights=...)
        model.fc = nn.Linear(model.fc.in_features, 3)
    """
    model = resnet18(weights=None)  # pretrained weights are overwritten by your .pth
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(
    model_dir: str = "models",
    filename: str = "resnet_model.pth"
) -> torch.nn.Module:
    """
    Load your trained ResNet18 from a .pth whose keys look like:
      'conv1.weight', 'bn1.weight', 'layer1.0.conv1.weight', 'fc.weight', ...
    """
    model = build_resnet18(num_classes=len(CLASS_NAMES))
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    state = torch.load(path, map_location=DEVICE)

    # If you saved with torch.save(model.state_dict())
    # then 'state' is already the state_dict
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        model.load_state_dict(state, strict=True)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        raise RuntimeError("Unexpected checkpoint format. Expected plain state_dict.")

    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_patient_inference(
    root: str,
    model: torch.nn.Module,
    max_slices: int = 300,
) -> Dict:
    """
    Run slice-level inference on all DICOM slices in 'root'.

    For memory safety on low-RAM servers:
      - we process ONE slice at a time
      - optionally downsample number of slices using 'max_slices'
    """
    dicom_paths, metadata = collect_dicom_paths_and_metadata(root)

    # OPTIONAL: downsample slices if too many
    if len(dicom_paths) > max_slices:
        # simple uniform sampling
        indices = np.linspace(0, len(dicom_paths) - 1, max_slices).astype(int)
        dicom_paths = [dicom_paths[i] for i in indices]

    slice_probs: List[np.ndarray] = []

    for p in dicom_paths:
        try:
            arr = load_and_normalize_slice(p)
        except Exception:
            continue


        # Skip non-2D slices
        if arr.ndim != 2:
            continue
        if arr.shape[0] < 10 or arr.shape[1] < 10:
            continue

        arr = resize_to_model(arr)  # Now guaranteed 2D

        # Convert to tensor
        img_t = torch.from_numpy(arr).float()

        # Skip if still not 2D
        if img_t.ndim != 2:
            continue

        # Normalize + reshape
        img_t = img_t / 255.0
        img_t = img_t.unsqueeze(0)          # (1, H, W)
        img_t = img_t.repeat(3, 1, 1)       # (3, H, W)
        img_t = img_t.unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        logits = model(img_t)                           # (1, C)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        slice_probs.append(probs)

        del img_t, logits, probs

    if not slice_probs:
        raise RuntimeError("No valid slices could be processed from DICOMs.")

    probs_arr = np.stack(slice_probs, axis=0)    # (N_slices, C)
    mean_probs = probs_arr.mean(axis=0)          # (C,)

    pred_idx = int(mean_probs.argmax())
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(mean_probs[pred_idx])

    preview_b64 = middle_slice_base64(dicom_paths)
    per_class = {CLASS_NAMES[i]: float(mean_probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": per_class,
        "num_slices_used": int(probs_arr.shape[0]),
        "patient_metadata": metadata,
        "preview_base64": preview_b64,
    }
