"""
Version CROSS-ATTENTION du K-Fold :

"""

import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

from train_utils_cross import set_seed, train_onecycle_amp, device   
from lazy_dataset import LazyEEGDataset
from models_with_cross import EEGFormerHybridV3                       
import matplotlib.pyplot as plt


# ===========================================
# Config générale
# ===========================================
BASE = os.path.expanduser("~/EEG_Emotion_Project")
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
PLOTS = os.path.join(BASE, "plots")

os.makedirs(MODELS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)


# ===========================================
# Normalisation robuste (médiane/IQR)
# ===========================================
def compute_median_iqr(X_train):
    median = np.median(X_train, axis=(0, 2))
    median = median[:, None]

    q75 = np.percentile(X_train, 75, axis=(0, 2))
    q25 = np.percentile(X_train, 25, axis=(0, 2))
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6
    iqr = iqr[:, None]

    return median.astype(np.float32), iqr.astype(np.float32)


# ===========================================
# Main CROSS
# ===========================================
def main():
    set_seed(42)

    print("\n=== CROSS-K-FOLD ===")

    # Charger dataset
    npz_path = os.path.join(DATA, "dataset_seed_windowed_compressed.npz")
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    X = npz["X"]
    y = npz["y"]

    # Mettre dans format (N, C, T)
    if X.ndim == 3 and X.shape[1] not in (62, 48, 32, 16) and X.shape[2] in (62, 48, 32, 16):
        X = X.transpose(0, 2, 1)

    print("Shape réel X =", X.shape)
    N, C, T = X.shape

    # Charger indices CROSS
    attn_path = os.path.join(MODELS, "top_indices_attention_cross.npz")
    assert os.path.exists(attn_path), (
        "\nERREUR : Tu dois d'abord exécuter main_train_cross.py "
        "pour générer les top-k CROSS.\n"
    )

    idxs = np.load(attn_path)

    # ======= Choisir nombre de canaux (ex: 62, 48, 32…) =======
    TARGET_CHANNELS = 62
    if TARGET_CHANNELS == 62:
        selected_channels = np.arange(62)
    else:
        selected_channels = idxs[f"top{TARGET_CHANNELS}"]
    selected_channels = np.array(selected_channels, dtype=int)

    print(f"\n=== K-Fold CROSS avec {len(selected_channels)} canaux ===")

    # Split 80/20
    train_idx, test_idx = train_test_split(
        np.arange(N),
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    y_train = y[train_idx]

    print(f"Train={len(train_idx)} | Test={len(test_idx)}")

    # Médiane + IQR (train)
    median, iqr = compute_median_iqr(X[train_idx])

    # ========= K-fold ==========
    K = 10
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    metrics = {k: [] for k in ["Acc", "F1", "Rec", "Prec", "ROC", "MCC"]}

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_idx, y_train)):
        print(f"\n========== Fold {fold+1}/{K} (CROSS) ==========")

        fold_train_idx = train_idx[idx_tr]
        fold_val_idx   = train_idx[idx_va]

        # Datasets lazy
        ds_tr = LazyEEGDataset(
            npz_path=npz_path,
            indices=fold_train_idx,
            median=median,
            iqr=iqr,
            channels=selected_channels,
            augment=True
        )

        ds_va = LazyEEGDataset(
            npz_path=npz_path,
            indices=fold_val_idx,
            median=median,
            iqr=iqr,
            channels=selected_channels,
            augment=False
        )

        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=4)
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=4)

        # Entraînement CROSS via train_utils_cross
        save_path = os.path.join(MODELS, f"best_kf_cross_{TARGET_CHANNELS}_fold{fold+1}.pth")

        model, _, _ = train_onecycle_amp(
            X_train_n=None,
            y_train=y[fold_train_idx],
            X_eval_n=None,
            y_eval=None,
            num_channels=len(selected_channels),
            save_name=save_path,
            plot_name_prefix=f"kfold_cross_{TARGET_CHANNELS}_fold{fold+1}",
            batch_size=128,
            epochs=100,
            max_lr=0.00055,
            weight_decay=0.00035,
            use_focal=True,
            label_smoothing=0.035,
            embed_dim=288,
            gate_hidden=64,
            pdrop_cnn=0.20,
            pdrop_head=0.18,
            heads=6,
            patience=10,
            div_factor=20.0,
            plots_dir=PLOTS,
            train_loader=dl_tr,
            eval_loader=dl_va,
            num_classes=len(np.unique(y)),
            seq_len=T
        )

        # Évaluation
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="macro")
        rec  = recall_score(y_true, y_pred, average="macro")
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

        try:
            roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except:
            roc = 0.0

        mcc = matthews_corrcoef(y_true, y_pred)

        metrics["Acc"].append(acc)
        metrics["F1"].append(f1)
        metrics["Rec"].append(rec)
        metrics["Prec"].append(prec)
        metrics["ROC"].append(roc)
        metrics["MCC"].append(mcc)

        print(f"Fold {fold+1}: Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} "
              f"| F1={f1:.3f} | ROC={roc:.3f} | MCC={mcc:.3f}")

    # Résumé global CROSS
    print("\n========== RÉSULTATS GLOBAUX K-FOLD (CROSS) ==========")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    # Barplot
    plt.figure(figsize=(10, 6))
    names = ["Acc", "F1", "Rec", "Prec", "ROC", "MCC"]
    means = [np.mean(metrics[m]) for m in names]
    stds = [np.std(metrics[m]) for m in names]

    plt.bar(names, means, yerr=stds, capsize=6)
    plt.ylabel("Score moyen (± écart-type)")
    plt.title(f"K-Fold CROSS — {len(selected_channels)} canaux")
    plt.grid(axis="y", alpha=0.3)

    out = os.path.join(
        PLOTS,
        f"kfold_cross_results_{len(selected_channels)}channels_top{TARGET_CHANNELS}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nGraphique CROSS sauvegardé :", out)


if __name__ == "__main__":
    main()
