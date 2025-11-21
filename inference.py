# inference.py
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np

from architectures.resnet25d import ResNet18_25D
from preprocessing import (
    load_dicom_slices,
    create_25d_tensors,
    get_middle_slice_base64,
)

DEVICE = torch.device("cpu")  # for deployment; change to "cuda" if GPU available

CLASS_NAMES = ["Normal", "Alzheimer", "Vascular Dementia"]


def load_model(model_dir: str = "models", filename: str = "resnet18_25d_vad.pth") -> torch.nn.Module:
    model = ResNet18_25D(in_channels=3, num_classes=len(CLASS_NAMES))
    path = os.path.join(model_dir, filename)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_model_on_folder(folder_path: str, model: torch.nn.Module) -> Dict:
    """
    Run ResNet18 2.5D on all slices of a single patient folder.
    Returns dict with prediction, confidence, metadata and middle slice image.
    """
    slices, metadata = load_dicom_slices(folder_path)
    stacks_np = create_25d_tensors(slices)

    if not stacks_np:
        raise RuntimeError("Not enough slices to form 2.5D stacks.")

    all_logits: List[torch.Tensor] = []

    for stack in stacks_np:
        # stack: (3, H, W) numpy
        x = torch.from_numpy(stack).float() / 255.0  # (3,H,W)
        x = x.unsqueeze(0).to(DEVICE)               # (1,3,H,W)

        logits = model(x)                           # (1,C)
        all_logits.append(logits)

    logits_cat = torch.cat(all_logits, dim=0)       # (N_stacks, C)
    probs = F.softmax(logits_cat, dim=1).mean(dim=0)  # (C,)

    conf_val, pred_idx = torch.max(probs, dim=0)
    pred_idx = int(pred_idx.item())
    confidence = float(conf_val.item())
    prediction = CLASS_NAMES[pred_idx]

    middle_slice_b64 = get_middle_slice_base64(slices)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "patient_metadata": metadata,
        "middle_slice_base64": middle_slice_b64,
    }
