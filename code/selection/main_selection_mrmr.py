"""
Pipeline : Sélection d'électrodes par mRMR + entraînement top-K
Étapes :
1. Charge SEED
2. Normalisation robuste
3. Split train / val / test
4. Calcul du ranking mRMR
5. Affichage complet (62 électrodes)
6. Barplot complet du ranking
7. Sélections top-48 / 32 / 16 / 8 / 6 / 4 / 2
8. Pour chaque top-K : tableau, CSV, barplot
9. Entraînement des modèles top-K
10. Évaluation harmonisée
"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

# Rendre accessibles models.py, train_utils.py, selection/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection.mrmr_selection import get_mrmr_ranking
from train_utils import train_onecycle_amp, device, set_seed
from models import EEGFormerHybridV2
from torch.utils.data import Dataset, DataLoader


# ============================
# Normalisation
# ============================
def compute_normalization(X_train):
    median = np.median(X_train, axis=(0, 2), keepdims=True)
    q75 = np.percentile(X_train, 75, axis=(0, 2), keepdims=True)
    q25 = np.percentile(X_train, 25, axis=(0, 2), keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6
    return median, iqr


def robust_norm_clip(X_in, median, iqr, clip_val=3.0):
    Xn = (X_in - median) / iqr
    return np.clip(Xn, -clip_val, clip_val)


# ============================
# Charger SEED
# ============================
def load_seed(DATA_DIR):
    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    assert os.path.exists(npz_path), f"Fichier introuvable : {npz_path}"

    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    y = npz["y"]

    # remettre au format (N, C, T)
    if X.ndim == 3 and X.shape[1] not in (62, 48, 32, 16) and X.shape[2] in (62, 48, 32, 16):
        X = np.transpose(X, (0, 2, 1))

    return X, y


# ============================
# Dataset Test simple
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
# Fonction barplot
# ============================
def plot_bar(values, labels, title, save_path):
    plt.figure(figsize=(8, 14))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Score d'importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================
# Évaluation harmonisée
# ============================
def evaluate_on_test_loader(model, loader):
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

    metrics = {
        "Acc": float(accuracy_score(y_true, y_pred)),
        "Prec": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Rec": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC": float(roc_auc_score(y_true, y_prob, multi_class="ovr")),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    return metrics


# ============================
# PIPELINE PRINCIPAL
# ============================
def main():

    set_seed(42)
    
    print("\n=== Pipeline mRMR (Top-48 / 32 / 16 / 8 / 6 / 4 / 2) ===")

    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    PLOT_DIR = os.path.join(BASE, "plots")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Charger les données SEED
    X, y = load_seed(DATA_DIR)
    print("Données :", X.shape, y.shape)

    # Split 80/20 identique à main_train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Normalisation
    median, iqr = compute_normalization(X_train)
    X_train_n = robust_norm_clip(X_train, median, iqr).astype(np.float32)
    X_test_n  = robust_norm_clip(X_test, median, iqr).astype(np.float32)

    # Ranking mRMR
    print("\n>>> Calcul du ranking mRMR...")
    ranking = get_mrmr_ranking(X_train_n, y_train, random_state=42)

    # Noms électrodes
    elec_names = [
        "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
        "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
        "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
        "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
        "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
        "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
        "O1","OZ","O2","CB1","CB2"
    ]

    print("\n=== Ranking mRMR complet (62 électrodes) ===")
    for pos, idx in enumerate(ranking, start=1):
        print(f"{pos:2d}. {elec_names[idx]} (index={idx})")

    # Sauvegarde CSV ranking complet
    csv_path = os.path.join(MODEL_DIR, "ranking_mrmr_62.csv")
    np.savetxt(csv_path, ranking, fmt="%d")
    print("Ranking sauvegardé :", csv_path)

    # Barplot complet mRMR
    scores_62 = np.array(list(range(62, 0, -1)))
    labels_62 = [elec_names[i] for i in ranking]
    plot_bar(scores_62, labels_62, "Ranking mRMR (62 électrodes)",
             os.path.join(PLOT_DIR, "mrmr_full_ranking_62.png"))

    # TOP-K demandés
    top_configs = {
        "48": ranking[:48],
        "32": ranking[:32],
        "16": ranking[:16],
        "8":  ranking[:8],
        "6":  ranking[:6],
        "4":  ranking[:4],
        "2":  ranking[:2],
    }

    # Évaluations finales
    test_results = {}

    # Boucle sur chaque K
    for k_name, idxs in top_configs.items():
        k = int(k_name)

        print(f"\n=== Sélection mRMR top-{k} ===")
        for pos, idx in enumerate(idxs, start=1):
            print(f"{pos:2d}. {elec_names[idx]} (index={idx})")

        # CSV top-K
        csv_k = os.path.join(MODEL_DIR, f"top{k}_mrmr_list.csv")
        with open(csv_k, "w") as f:
            for idx in idxs:
                f.write(f"{elec_names[idx]},{idx}\n")

        # Barplot top-K
        scores_k = np.array(list(range(k, 0, -1)))
        labels_k = [elec_names[i] for i in idxs]
        plot_bar(scores_k, labels_k,
                 f"mRMR Top-{k}",
                 os.path.join(PLOT_DIR, f"barplot_top{k}_mrmr.png"))

        # Sous-ensemble des canaux
        X_train_sub = X_train_n[:, idxs, :]
        X_test_sub  = X_test_n[:, idxs, :]

        # Split interne (10% val)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sub, y_train, test_size=0.10, stratify=y_train, random_state=123
        )

        print(f"\n=== Entraînement mRMR top-{k} ===")
        save_path = os.path.join(MODEL_DIR, f"best_model_mrmr_{k}.pth")

        model_k, hist_k, best_val = train_onecycle_amp(
            X_train_n=X_tr, y_train=y_tr,
            X_eval_n=X_val, y_eval=y_val,
            num_channels=k,
            save_name=save_path,
            plot_name_prefix=f"mrmr_{k}",
            batch_size=128,
            epochs=100,
            max_lr=5.5e-4,
            weight_decay=3.5e-4,
            use_focal=True,
            label_smoothing=0.035,
            embed_dim=288,
            gate_hidden=64,
            pdrop_cnn=0.20,
            pdrop_head=0.18,
            heads=6,
            patience=10,
            div_factor=20.0,
            plots_dir=None
        )

        # Charger meilleur modèle
        model_k.load_state_dict(torch.load(save_path, map_location=device))
        model_k.to(device)

        # Test final
        test_loader = DataLoader(
            EEGDatasetSimple(X_test_sub, y_test),
            batch_size=128,
            shuffle=False
        )
        metrics = evaluate_on_test_loader(model_k, test_loader)
        test_results[k_name] = metrics

        # Affichage harmonisé
        print(f"\n=== Test final top-{k} ===")
        print(f"Accuracy  : {metrics['Acc']*100:.2f}%")
        print(f"Precision : {metrics['Prec']*100:.2f}%")
        print(f"Recall    : {metrics['Rec']*100:.2f}%")
        print(f"F1-score  : {metrics['F1']*100:.2f}%")
        print(f"ROC-AUC   : {metrics['ROC']:.3f}")
        print(f"MCC       : {metrics['MCC']:.3f}")

    # Résumé final
    print("\n\n========== RÉSUMÉ GLOBAL mRMR ==========")
    for k, m in test_results.items():
        print(f"{k} canaux -> "
              f"Acc={m['Acc']*100:.2f}% | "
              f"Prec={m['Prec']*100:.2f}% | "
              f"Rec={m['Rec']*100:.2f}% | "
              f"F1={m['F1']*100:.2f}% | "
              f"ROC-AUC={m['ROC']:.3f} | "
              f"MCC={m['MCC']:.3f}")


if __name__ == "__main__":
    main()
