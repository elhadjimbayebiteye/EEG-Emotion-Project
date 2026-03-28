"""
Version CROSS-ATTENTION du pipeline complet :

"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)

from train_utils_cross import set_seed, train_onecycle_amp, device
from models_with_cross import EEGFormerHybridV3   # <-- Le nouveau modèle CROSS

###############################################################################
# Normalisation robuste
###############################################################################
def robust_norm_clip(X_in, median, iqr, clip_val=3.0):
    Xn = (X_in - median) / iqr
    return np.clip(Xn, -clip_val, clip_val)

###############################################################################
# Récupération des poids d’attention CROSS
###############################################################################
def get_attention_ranking(model, X_ref, y_ref, device, batch_size=256):
    from torch.utils.data import DataLoader
    from train_utils import EEGDataset

    loader = DataLoader(
        EEGDataset(X_ref, y_ref, augment=False),
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    attn_sum = None
    n_batches = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _ = model(xb)

            if hasattr(model, "get_attention_weights"):
                w = model.get_attention_weights()
            else:
                raise RuntimeError("Le modèle doit fournir get_attention_weights()")

            w = w.detach().float().cpu()
            attn_sum = w if attn_sum is None else attn_sum + w
            n_batches += 1
            if n_batches >= 8:
                break

    attn_mean = (attn_sum / n_batches).numpy()
    attn_mean = attn_mean / (attn_mean.sum() + 1e-9)
    sorted_idx = np.argsort(attn_mean)[::-1]
    return attn_mean, sorted_idx


###############################################################################
# Evaluation sur test
###############################################################################
def evaluate_on_test(model, X_test, y_test, device, num_channels, tag):
    from torch.utils.data import DataLoader
    from train_utils import EEGDataset

    loader = DataLoader(
        EEGDataset(X_test, y_test, augment=False),
        batch_size=128,
        shuffle=False
    )

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc  = roc_auc_score(y_true, y_prob, multi_class="ovr")
    mcc  = matthews_corrcoef(y_true, y_pred)

    print(f"\n=== Test final ({num_channels} canaux CROSS) — {tag} ===")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-score  : {f1*100:.2f}%")
    print(f"ROC-AUC   : {roc:.3f}")
    print(f"MCC       : {mcc:.3f}")

    return {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "ROC": roc, "MCC": mcc}


###############################################################################
# MAIN — Pipeline CROSS
###############################################################################
def main():
    set_seed(42)
    print("Device :", device)

    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    PLOT_DIR = os.path.join(BASE, "plots")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    ###########################################################################
    # 1) Chargement
    ###########################################################################
    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    npz = np.load(npz_path, allow_pickle=True)

    X = npz["X"]
    y = npz["y"]

    if X.ndim == 3 and X.shape[1] not in (62,48,32,16) and X.shape[2] in (62,48,32,16):
        X = X.transpose(0,2,1)

    N, C, T = X.shape
    print("Dataset :", X.shape, y.shape)

    ###########################################################################
    # 2) Split 80/20
    ###########################################################################
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    ###########################################################################
    # 3) Normalisation robuste
    ###########################################################################
    median = np.median(X_train, axis=(0,2), keepdims=True)
    q75    = np.percentile(X_train, 75, axis=(0,2), keepdims=True)
    q25    = np.percentile(X_train, 25, axis=(0,2), keepdims=True)
    iqr    = q75 - q25
    iqr[iqr < 1e-6] = 1e-6

    X_train_n = robust_norm_clip(X_train, median, iqr, 3.0).astype(np.float32)
    X_test_n  = robust_norm_clip(X_test,  median, iqr, 3.0).astype(np.float32)
    y_train   = y_train.astype(np.int64)
    y_test    = y_test.astype(np.int64)

    ###########################################################################
    # 4) Split interne 10% pour validation
    ###########################################################################
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_n, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=123
    )

    ###########################################################################
    # 5) Entraînement CROSS — 62 canaux
    ###########################################################################
    save_62 = os.path.join(MODEL_DIR, "best_model_62_cross.pth")
    plot_62 = "plot_62_cross"

    model_62, hist, best = train_onecycle_amp(
        X_train_n=X_tr,
        y_train=y_tr,
        X_eval_n=X_val,
        y_eval=y_val,
        num_channels=62,
        save_name=save_62,
        plot_name_prefix=plot_62,
        embed_dim=288,
        gate_hidden=64,
        pdrop_cnn=0.20,
        pdrop_head=0.18,
        heads=6,
    )

    model_62.load_state_dict(torch.load(save_62, map_location=device))
    model_62 = model_62.to(device)

    ###########################################################################
    # 6) Ranking électrodes (CROSS)
    ###########################################################################
    attn_mean, sorted_idx = get_attention_ranking(model_62, X_train_n, y_train, device)
    top48 = sorted_idx[:48]
    top32 = sorted_idx[:32]
    top16 = sorted_idx[:16]
    top8  = sorted_idx[:8]
    top6  = sorted_idx[:6]
    top4  = sorted_idx[:4]
    top2  = sorted_idx[:2]

    np.savez(os.path.join(MODEL_DIR, "top_indices_attention_cross.npz"),
             top48=top48, top32=top32, top16=top16,
             top8=top8, top6=top6, top4=top4, top2=top2)

    ###########################################################################
    # 7) Test final 62 CROSS
    ###########################################################################
    test_62 = evaluate_on_test(model_62, X_test_n, y_test, device, 62, "62 canaux CROSS")

    ###########################################################################
    # 8) Boucle 48/32/16/8/6/4/2 CROSS
    ###########################################################################
    configs = [
        ("48", top48),
        ("32", top32),
        ("16", top16),
        ("8",  top8),
        ("6",  top6),
        ("4",  top4),
        ("2",  top2),
    ]

    results = {"62": test_62}

    for name, idxs in configs:
        n = int(name)
        print(f"\n===== {n} canaux CROSS =====")

        X_train_sub = X_train_n[:, idxs, :]
        X_test_sub  = X_test_n[:, idxs, :]

        X_trs, X_vals, y_trs, y_vals = train_test_split(
            X_train_sub, y_train,
            test_size=0.10, stratify=y_train,
            random_state=123
        )

        save_path = os.path.join(MODEL_DIR, f"best_model_{name}_cross.pth")
        plot_pref = f"plot_{name}_cross"

        model_sub, _, best = train_onecycle_amp(
            X_train_n=X_trs,
            y_train=y_trs,
            X_eval_n=X_vals,
            y_eval=y_vals,
            num_channels=n,
            save_name=save_path,
            plot_name_prefix=plot_pref,
            embed_dim=288,
            gate_hidden=64,
            pdrop_cnn=0.20,
            pdrop_head=0.18,
            heads=6,
        )

        model_sub.load_state_dict(torch.load(save_path, map_location=device))
        model_sub = model_sub.to(device)

        results[name] = evaluate_on_test(
            model_sub, X_test_sub, y_test, device, n, f"{n} canaux CROSS"
        )

        ##############################################
        # 9) Résumé global
        ##############################################
        print("\n===== RÉSUMÉ GLOBAL CROSS =====")

        for k, m in results.items():
            print(
                f"{k} canaux --> "
                f"Acc={m['Acc']*100:.2f}% | "
                f"Prec={m['Prec']*100:.2f}% | "
                f"Rec={m['Rec']*100:.2f}% | "
                f"F1={m['F1']*100:.2f}% | "
                f"ROC={m['ROC']:.3f} | "
                f"MCC={m['MCC']:.3f}"
            )



if __name__ == "__main__":
    main()
