"""
Pipeline : Sélection d'électrodes par SBFS + entraînement top-K
Objectif :
- Sélection avec modèle léger (SBFS)
- Évaluation finale avec EEGFormer
- Affichage harmonisé des métriques (Acc, Prec, Rec, F1, ROC-AUC, MCC)
"""

import os
import sys

# ============================
# PYTHON PATH
# ============================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from selection.sbfs_selection import sbfs_select_channels
from train_utils import train_onecycle_amp, device, set_seed
from models import EEGFormerHybridV2

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

# ============================
# Dataset simple
# ============================
class EEGDatasetSimple(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# Normalisation robuste
# ============================
def compute_normalization(X_train):
    median = np.median(X_train, axis=(0, 2), keepdims=True)
    q75 = np.percentile(X_train, 75, axis=(0, 2), keepdims=True)
    q25 = np.percentile(X_train, 25, axis=(0, 2), keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6
    return median, iqr


def robust_norm_clip(X, median, iqr, clip_val=3.0):
    Xn = (X - median) / iqr
    return np.clip(Xn, -clip_val, clip_val)


# ============================
# Chargement SEED
# ============================
def load_seed(DATA_DIR):
    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    assert os.path.exists(npz_path), f"Fichier introuvable : {npz_path}"

    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    y = npz["y"]

    # Forcer format (N, C, T)
    if X.ndim == 3 and X.shape[1] != 62:
        X = np.transpose(X, (0, 2, 1))

    return X, y


# ============================
# Évaluation harmonisée
# ============================
def evaluate_model(model, loader):
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            prob = torch.softmax(model(xb), dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)

            all_true.extend(yb.numpy())
            all_pred.extend(pred)
            all_prob.extend(prob)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC": roc_auc_score(y_true, y_prob, multi_class="ovr"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ============================
# MAIN
# ============================
def main():
    set_seed(42)

    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n=== Pipeline SBFS (Top-K) ===")

    # Charger données
    X, y = load_seed(DATA_DIR)
    print("Données :", X.shape, y.shape)

    # Split 80 / 20 (test final)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Normalisation
    median, iqr = compute_normalization(X_train)
    X_train_n = robust_norm_clip(X_train, median, iqr).astype(np.float32)
    X_test_n  = robust_norm_clip(X_test, median, iqr).astype(np.float32)

    # Configurations Top-K (RECOMMANDÉ : commencer par [16])
    top_ks = [16]

    results = {}

    for k in top_ks:
        print(f"\n>>> SBFS sélection Top-{k}")

        # --- SBFS SELECTION
        idxs = sbfs_select_channels(
            X_train, y_train,
            target_k=k
        )

        # Sauvegarde des canaux
        csv_path = os.path.join(MODEL_DIR, f"top{k}_sbfs.csv")
        np.savetxt(csv_path, idxs, fmt="%d")
        print("Canaux sauvegardés :", csv_path)

        # Sous-ensemble des données
        X_sub = X_train_n[:, idxs, :]
        X_test_sub = X_test_n[:, idxs, :]

        # Split interne (validation 10 %)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y_train,
            test_size=0.10,
            stratify=y_train,
            random_state=123
        )

        # Entraînement EEGFormer
        model, hist, best_val = train_onecycle_amp(
            X_train_n=X_tr, y_train=y_tr,
            X_eval_n=X_val, y_eval=y_val,
            num_channels=k,
            save_name=os.path.join(MODEL_DIR, f"best_model_sbfs_{k}.pth"),
            plot_name_prefix=f"sbfs_{k}",
            epochs=100,
            batch_size=128
        )

        # Test final
        test_loader = DataLoader(
            EEGDatasetSimple(X_test_sub, y_test),
            batch_size=128,
            shuffle=False
        )

        model.to(device)
        metrics = evaluate_model(model, test_loader)
        results[k] = metrics

        # === AFFICHAGE FINAL HARMONISÉ ===
        print(f"\n=== Résultats SBFS Top-{k} ===")
        print(f"Accuracy  : {metrics['Acc']*100:.2f}%")
        print(f"Precision : {metrics['Prec']*100:.2f}%")
        print(f"Recall    : {metrics['Rec']*100:.2f}%")
        print(f"F1-score  : {metrics['F1']*100:.2f}%")
        print(f"ROC-AUC   : {metrics['ROC']:.3f}")
        print(f"MCC       : {metrics['MCC']:.3f}")

    # Résumé global
    print("\n===== RÉSUMÉ GLOBAL SBFS =====")
    for k, m in results.items():
        print(
            f"{k} canaux -> "
            f"Acc={m['Acc']*100:.2f}% | "
            f"Prec={m['Prec']*100:.2f}% | "
            f"Rec={m['Rec']*100:.2f}% | "
            f"F1={m['F1']*100:.2f}% | "
            f"ROC-AUC={m['ROC']:.3f} | "
            f"MCC={m['MCC']:.3f}"
        )


if __name__ == "__main__":
    main()
