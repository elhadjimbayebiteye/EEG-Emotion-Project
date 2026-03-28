"""
Pipeline : Sélection d'électrodes par ReliefF + entraînement top-k
Étapes :
1. Charge SEED
2. Normalisation robuste
3. Splits train / val / test
4. Calcul du ranking ReliefF
5. Affichage du ranking + barplot + CSV
6. Entraînement des modèles top-48 / 32 / 16 / 8 / 6 / 4 / 2 via LazyDataset
7. Évaluation sur le même test set
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Chemin pour importer models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EEGFormerHybridV2
from train_utils import train_onecycle_amp, device
from main_train import evaluate_on_test
from selection.relieff_selection import get_relieff_ranking
from lazy_dataset import LazyEEGDataset
from torch.utils.data import DataLoader


# -------------------------------------------------------
#  NORMALISATION
# -------------------------------------------------------
def robust_norm_clip(X_in, median, iqr, clip_val=3.0):
    Xn = (X_in - median) / iqr
    return np.clip(Xn, -clip_val, clip_val)


def compute_normalization(X_train):
    median = np.median(X_train, axis=(0, 2), keepdims=True)
    q75 = np.percentile(X_train, 75, axis=(0, 2), keepdims=True)
    q25 = np.percentile(X_train, 25, axis=(0, 2), keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6
    return median, iqr


# -------------------------------------------------------
#  CHARGEMENT SEED
# -------------------------------------------------------
def load_seed(DATA_DIR):
    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    y = npz["y"]

    # Vérifie si l'ordre est (N, T, C)
    if X.ndim == 3 and X.shape[1] not in (62, 48, 32, 16) and X.shape[2] in (62, 48, 32, 16):
        X = np.transpose(X, (0, 2, 1))  # → (N, C, T)

    return X, y


# -------------------------------------------------------
#  MAIN PIPELINE
# -------------------------------------------------------
def main():
    print("=== Pipeline ReliefF (Top-48 / 32 / 16 / 8 / 6 / 4 / 2) ===")

    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    PLOT_DIR = os.path.join(BASE, "plots")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1) CHARGER SEED
    X, y = load_seed(DATA_DIR)
    print("Données :", X.shape, y.shape)

    # 2) SPLIT GLOBAL 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3) NORMALISATION
    median, iqr = compute_normalization(X_train)
    X_train_n = robust_norm_clip(X_train, median, iqr).astype(np.float32)
    X_test_n = robust_norm_clip(X_test, median, iqr).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 4) SPLIT INTERNE 10%
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_n, y_train, test_size=0.10, stratify=y_train, random_state=123
    )

    # 5) CALCUL RELIEFF
    print("\n>>> Calcul du ranking ReliefF...")
    scores, ranking = get_relieff_ranking(X_tr, y_tr)
    print("Top 10 ReliefF :", ranking[:10])

    # 5b) Afficher le ranking avec noms d’électrodes
    elec_names = [
        "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
        "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
        "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
        "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
        "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
        "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
        "O1","OZ","O2","CB1","CB2"
    ]

    df_rank = pd.DataFrame({
        "electrode": [elec_names[i] for i in ranking],
        "index": ranking,
        "score": scores[ranking]
    })

    csv_path = os.path.join(MODEL_DIR, "ranking_relieff_62.csv")
    df_rank.to_csv(csv_path, index=False)
    print("Ranking sauvegardé :", csv_path)

    # Barplot
    plt.figure(figsize=(8, 16))
    plt.barh(df_rank["electrode"], df_rank["score"])
    plt.gca().invert_yaxis()
    plt.title("Scores ReliefF (62 électrodes)")
    plt.xlabel("Score")
    plt.grid(axis="x", alpha=0.3)
    barplot_path = os.path.join(PLOT_DIR, "barplot_relieff_62.png")
    plt.tight_layout()
    plt.savefig(barplot_path, dpi=150)
    plt.close()
    print("Barplot sauvegardé :", barplot_path)

    # Indices top-k
    top48 = ranking[:48]
    top32 = ranking[:32]
    top16 = ranking[:16]
    top8  = ranking[:8]
    top6  = ranking[:6]
    top4  = ranking[:4]
    top2  = ranking[:2]

    # 6) ENTRAÎNEMENT DES MODÈLES POUR CHAQUE TOP-K
    RESULTS = {}

    for name, idxs in [
        ("48", top48),
        ("32", top32),
        ("16", top16),
        ("8",  top8),
        ("6",  top6),
        ("4",  top4),
        ("2",  top2),
    ]:
        n_ch = int(name)
        print(f"\n=== Entraînement ReliefF top-{name} ===")

        # Sous-jeux réduits
        X_tr_sub = X_tr[:, idxs, :]
        X_val_sub = X_val[:, idxs, :]
        X_test_sub = X_test_n[:, idxs, :]

        save_path = os.path.join(MODEL_DIR, f"best_relieff_{name}.pth")
        plot_prefix = f"relieff_{name}"

        # Entraînement
        model_sub, hist_sub, best_val = train_onecycle_amp(
            X_train_n=X_tr_sub,
            y_train=y_tr,
            X_eval_n=X_val_sub,
            y_eval=y_val,
            num_channels=n_ch,
            save_name=save_path,
            plot_name_prefix=plot_prefix,
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
            plots_dir=PLOT_DIR,
        )

        model_sub.load_state_dict(torch.load(save_path, map_location=device))
        model_sub.to(device)

        # Évaluation
        metrics = evaluate_on_test(model_sub, X_test_sub, y_test, device, n_ch, f"ReliefF top-{name}")
        RESULTS[name] = metrics

    # 7) RÉSUMÉ GLOBAL
    print("\n\n=== Résumé global ReliefF ===")
    for k, m in RESULTS.items():
        print(
            f"{k} canaux --> "
            f"Acc={m['Acc']*100:.2f}% | "
            f"Prec={m['Prec']*100:.2f}% | "
            f"Rec={m['Rec']*100:.2f}% | "
            f"F1={m['F1']*100:.2f}% | "
            f"ROC-AUC={m['ROC']:.3f} | "
            f"MCC={m['MCC']:.3f}"
        )

    print("\n=== Job terminé ===")


if __name__ == "__main__":
    main()
