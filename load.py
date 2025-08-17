import os
import random
from pathlib import Path
import pickle  # (kept if you need it later)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----------------------------
# Config
# ----------------------------

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "model_weights.pth"
TEST_ROOT = ROOT / "data" / "unseen"
TARGET_NETWORK = 6
trait_columns = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']

LABELS_CSV = Path(r"C:/Users/namgy/OneDrive/Desktop/APS360-NEOFFI-Project/data/unrestricted_5_22_56_15.csv")
TEST85_NUM = Path(r"C:/Users/namgy/OneDrive/Desktop/APS360-NEOFFI-Project/data/unseen/test_ids_85_numeric.txt")

# ----------------------------
# Model import
# ----------------------------
try:
    from model import fconnCNN  # your architecture
except ImportError:
    print("Warning: model.py not found. Please ensure your model.py file exists and defines fconnCNN.")
    fconnCNN = None

# ----------------------------
# Dataset
# ----------------------------
class fconnDataset(Dataset):
    def __init__(self, root_dir, target_network=None, label_map=None, transform=None):
        self.root_dir = root_dir
        self.target_network = target_network
        self.label_map = label_map or {}
        self.transform = transform

        valid_patients = []
        for patient_name in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_name)
            patient_id = os.path.basename(patient_path).replace("subject_", "")

            if not os.path.isdir(patient_path):
                continue

            # must have a label
            label = self.label_map.get(patient_id)
            if label is None:
                continue
            label = np.asarray(label, dtype=np.float32)
            if np.isnan(label).any():
                continue

            # must have the target network file
            file_path = os.path.join(patient_path, f"net_{self.target_network:02d}.npy")
            if not os.path.exists(file_path):
                continue

            valid_patients.append(patient_path)

        self.valid_patients = valid_patients

    def __len__(self):
        return len(self.valid_patients)

    def __getitem__(self, idx):
        patient_path = self.valid_patients[idx]
        patient_id = os.path.basename(patient_path).replace("subject_", "")

        file_path = os.path.join(patient_path, f"net_{self.target_network:02d}.npy")
        data = np.load(file_path)                     # [N, 10]
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1, N, 10] (C=1)

        if self.transform:
            tensor = self.transform(tensor)

        label_array = self.label_map.get(patient_id, np.full(5, np.nan, dtype=np.float32))
        label_tensor = torch.tensor(label_array, dtype=torch.float32)

        return tensor, label_tensor

    def getpatientid(self, idx):
        patient_path = self.valid_patients[idx]
        patient_id = os.path.basename(patient_path).replace("subject_", "")
        return patient_id

# ----------------------------
# Labels
# ----------------------------
def build_label_map():
    label_df_all = pd.read_csv(LABELS_CSV)
    label_df_all['Subject'] = label_df_all['Subject'].astype(str)
    label_df_all = label_df_all.dropna(subset=trait_columns)

    try:
        test_ids = set([ln.strip() for ln in TEST85_NUM.read_text(encoding='utf-8').splitlines() if ln.strip()])
    except UnicodeDecodeError:
        test_ids = set([ln.strip() for ln in TEST85_NUM.read_text(encoding='cp1252').splitlines() if ln.strip()])

    labels_test = label_df_all[label_df_all['Subject'].isin(test_ids)].copy()
    labels_trainval = label_df_all[~label_df_all['Subject'].isin(test_ids)].copy()

    # sanity check
    assert set(labels_test['Subject']).isdisjoint(labels_trainval['Subject']), "Leakage: overlap between test and dev!"
    print(f"Total with labels: {len(label_df_all)} | Hold-out TEST (85): {len(labels_test)} | DEV: {len(labels_trainval)}")

    label_map_test = {row['Subject']: row[trait_columns].values.astype(np.float32)
                      for _, row in labels_test.iterrows()}
    return label_map_test

# ----------------------------
# Robust model loader
# ----------------------------
def load_model_robust(weights_path, device, model_ctor, **model_kwargs):
    """
    Handles:
      - torch.save(model, path)          -> nn.Module
      - torch.save(model.state_dict(),)  -> OrderedDict
      - torch.save({'state_dict': ...})  -> dict checkpoint
      - also strips 'module.' prefixes
    """
    # Always load raw checkpoint to CPU first
    ckpt = torch.load(weights_path, map_location='cpu')

    # Case A: full model object saved
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    state_dict = None

    # Case B: plain state_dict
    if isinstance(ckpt, (dict,)):
        # If it's already a state_dict (mapping of param name -> tensor)
        is_state_dict_like = all(
            isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in ckpt.values()
        )
        if is_state_dict_like and any(k.endswith(('.weight', '.bias')) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            # Common checkpoint wrappers
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt:
                    if isinstance(ckpt[key], nn.Module):
                        model = ckpt[key].to(device)
                        model.eval()
                        return model
                    if isinstance(ckpt[key], dict):
                        state_dict = ckpt[key]
                        break

    # Case C: OrderedDict (state_dict)
    from collections import OrderedDict
    if isinstance(ckpt, OrderedDict):
        state_dict = ckpt

    if state_dict is None:
        raise TypeError(f"Unexpected checkpoint type/format: {type(ckpt)} with keys {list(ckpt) if isinstance(ckpt, dict) else 'N/A'}")

    # Instantiate fresh model
    model = model_ctor(**model_kwargs)

    # Strip 'module.' if present (DDP/DataParallel)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load] Missing keys: {missing}")
    if unexpected:
        print(f"[load] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model

# ----------------------------
# Public helpers
# ----------------------------
def get_available_subjects():
    try:
        label_map_test = build_label_map()
        test_dataset_external = fconnDataset(
            root_dir=TEST_ROOT,
            label_map=label_map_test,
            target_network=TARGET_NETWORK
        )

        num_subjects = min(5, len(test_dataset_external))
        subjects_info = []
        for i in range(num_subjects):
            subject_id = test_dataset_external.getpatientid(i)
            subjects_info.append({
                'index': i,
                'id': subject_id,
                'display_name': f"Subject {i+1} (ID: {subject_id})"
            })
        return subjects_info
    except Exception as e:
        print(f"Error getting available subjects: {e}")
        return []

def load_model_and_predict_single(subject_idx):
    if fconnCNN is None:
        raise ImportError("fconnCNN not found. Ensure model.py exists and defines class fconnCNN(N, output_dim).")

    # Build label map and dataset
    label_map_test = build_label_map()
    test_dataset_external = fconnDataset(
        root_dir=TEST_ROOT,
        label_map=label_map_test,
        target_network=TARGET_NETWORK
    )

    print(f"Total available subjects in test dataset: {len(test_dataset_external)}")
    if subject_idx >= len(test_dataset_external):
        raise ValueError(f"Subject index {subject_idx} is out of range. Available subjects: 0-{len(test_dataset_external)-1}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Infer N (ROIs) from one sample
    sample_tensor, _ = test_dataset_external[0]   # [1, N, 10]
    N = sample_tensor.shape[1]                    # C=1, N is dim=1 here
    output_dim = 5

    print(f"Model parameters: N={N}, output_dim={output_dim}")

    # Load model robustly (this is the fix)
    model = load_model_robust(
        weights_path=WEIGHTS,
        device=device,
        model_ctor=fconnCNN,
        N=N,
        output_dim=output_dim
    )

    # Predict a single subject
    tensor, label = test_dataset_external[subject_idx]
    subject_id = test_dataset_external.getpatientid(subject_idx)

    X = tensor.unsqueeze(0).to(device)  # [B=1, C=1, N, 10]
    y = label.to(device)

    with torch.no_grad():
        pred = model(X).cpu().numpy().flatten()

    gt = y.cpu().numpy().flatten()

    print(f"Successfully processed subject {subject_id}")
    print(f"Ground Truth: {gt}")
    print(f"Prediction: {pred}")

    return gt, pred, subject_id, trait_columns

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        subjects = get_available_subjects()
        print(f"Available subjects: {len(subjects)}")
        for subject in subjects:
            print(f"  - {subject['display_name']}")

        if subjects:
            print(f"\nTesting prediction on {subjects[0]['display_name']}...")
            gt, pred, subject_id, traits = load_model_and_predict_single(0)

            print(f"\nPrediction Results for {subject_id}:")
            for trait, gt_val, pred_val in zip(traits, gt, pred):
                err = abs(gt_val - pred_val)
                print(f"  {trait}: GT={gt_val:.3f} | Pred={pred_val:.3f} | Error={err:.3f}")

    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
