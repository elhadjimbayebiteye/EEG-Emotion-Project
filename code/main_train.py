"""
Pipeline complet :
- Split global : 80% train / 20% test
- Normalisation médiane/IQR sur train
- Split interne : 10% des 80% pour validation (72/8/20)
- Entraînement modèle 62 canaux
- Sélection d'électrodes par attention
- Entraînement et test pour 48, 32, 16, 8, 6, 4, 2 électrodes
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

from train_utils import set_seed, train_onecycle_amp, device
from models import EEGFormerHybridV2


def robust_norm_clip(X_in, median, iqr, clip_val=3.0):
    Xn = (X_in - median) / iqr
    return np.clip(Xn, -clip_val, clip_val)


def get_attention_ranking(model_62, X_ref, y_ref, device, batch_size=256):
    """
    Calcule les poids d'attention moyens par électrode
    en passant quelques batchs du jeu de référence (train).
    Retourne attn_mean, sorted_idx.
    """
    from torch.utils.data import DataLoader
    from train_utils import EEGDataset

    loader = DataLoader(
        EEGDataset(X_ref, y_ref, augment=False),
        batch_size=batch_size,
        shuffle=False
    )

    model_62.eval()
    attn_sum = None
    n_batches = 0

    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            xb = xb.to(device)
            _ = model_62(xb)
            if hasattr(model_62, "get_attention_weights"):
                w = model_62.get_attention_weights()
            else:
                raise RuntimeError("Le modèle doit exposer get_attention_weights() -> (num_channels,)")

            w = w.detach().float().cpu()
            attn_sum = w if attn_sum is None else (attn_sum + w)
            n_batches += 1
            if n_batches >= 8:
                break

    attn_mean = (attn_sum / n_batches).numpy()
    attn_mean = attn_mean / (attn_mean.sum() + 1e-9)
    sorted_idx = np.argsort(attn_mean)[::-1]
    return attn_mean, sorted_idx


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

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n=== Test final ({num_channels} canaux) — {tag} ===")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-score  : {f1*100:.2f}%")
    print(f"ROC-AUC   : {roc:.3f}")
    print(f"MCC       : {mcc:.3f}")

    return {
        "Acc": acc, "Prec": prec, "Rec": rec,
        "F1": f1, "ROC": roc, "MCC": mcc
    }


def main():
    set_seed(42)
    print("Device :", device)

    # ----------------------------
    # 0) Chemins
    # ----------------------------
    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    PLOT_DIR = os.path.join(BASE, "plots")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ----------------------------
    # 1) Chargement NPZ
    # ----------------------------
    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    assert os.path.exists(npz_path), f"Fichier introuvable : {npz_path}"

    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    y = npz["y"]

    if X.ndim == 3 and X.shape[1] not in (62, 48, 32, 16) and X.shape[2] in (62, 48, 32, 16):
        X = np.transpose(X, (0, 2, 1))

    N, C, T = X.shape
    print(f"Chargé : X {X.shape} | y {y.shape}")

    # ----------------------------
    # 2) Split global 80/20
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Split global -> Train : {X_train.shape}, Test : {X_test.shape}")

    # ----------------------------
    # 3) Normalisation robuste (train only)
    # ----------------------------
    median = np.median(X_train, axis=(0, 2), keepdims=True)
    q75 = np.percentile(X_train, 75, axis=(0, 2), keepdims=True)
    q25 = np.percentile(X_train, 25, axis=(0, 2), keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6

    X_train_n = robust_norm_clip(X_train, median, iqr, 3.0).astype(np.float32)
    X_test_n = robust_norm_clip(X_test, median, iqr, 3.0).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    print("Normalisation terminée (train only).")
    print("Train_n :", X_train_n.shape, "| Test_n :", X_test_n.shape)

    # ----------------------------
    # 4) Split interne 10% des 80% ( 72/8/20)
    # ----------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_n, y_train, test_size=0.10, stratify=y_train, random_state=123
    )
    print(f"Split interne -> Train_eff : {X_tr.shape}, Val : {X_val.shape}")

    # ----------------------------
    # 5) Entraînement 62 canaux
    # ----------------------------
    save_62 = os.path.join(MODEL_DIR, "best_model_62_final.pth")
    plot_prefix_62 = "loss_acc_f1_62_final"

    model_62, hist_62, best62 = train_onecycle_amp(
        X_train_n=X_tr,
        y_train=y_tr,
        X_eval_n=X_val,
        y_eval=y_val,
        num_channels=62,
        save_name=save_62,
        plot_name_prefix=plot_prefix_62,
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

    print(f"\nBest Val Acc (62 canaux) = {best62*100:.2f}%")

    # Recharge meilleur modèle
    model_62.load_state_dict(torch.load(save_62, map_location=device))
    model_62.to(device)

    # ----------------------------
    # 6) Attention + ranking électrodes
    # ----------------------------
    attn_mean, sorted_idx = get_attention_ranking(
        model_62, X_train_n, y_train, device=device
    )

    # noms d'électrodes
    elec_names = None
    for k in ["channel_names", "channels", "ch_names", "electrode_names", "eeg_channels"]:
        if k in npz.files:
            elec_names = list(npz[k])
            break

    if elec_names is None or len(elec_names) != 62:
        elec_names = [
            "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
            "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
            "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
            "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
            "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
            "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
            "O1","OZ","O2","CB1","CB2"
        ]

    # dataframe ranking + CSV + barplot
    import pandas as pd

    attn_by_name = {elec_names[i]: float(attn_mean[i]) for i in range(62)}
    df_rank = pd.DataFrame({
        "rank": list(range(1, 63)),
        "electrode": list(attn_by_name.keys()),
        "attention_weight": list(attn_by_name.values())
    }).sort_values("attention_weight", ascending=False)

    csv_rank_path = os.path.join(MODEL_DIR, "Electrodes_Attention_Ranking_62.csv")
    df_rank.to_csv(csv_rank_path, index=False)
    print("Ranking sauvegardé dans :", csv_rank_path)

    plt.figure(figsize=(8, 16))
    plt.barh(df_rank["electrode"], df_rank["attention_weight"])
    plt.gca().invert_yaxis()
    plt.xlabel("Poids d'attention normalisé")
    plt.title("Poids d'attention par électrode (ordre décroissant) — 62 canaux")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    barplot_path = os.path.join(PLOT_DIR, "barplot_attention_62.png")
    plt.savefig(barplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Barplot sauvegardé :", barplot_path)

    # indices top-k
    top48_idx = sorted_idx[:48]
    top32_idx = sorted_idx[:32]
    top16_idx = sorted_idx[:16]
    top8_idx  = sorted_idx[:8]
    top6_idx  = sorted_idx[:6]
    top4_idx  = sorted_idx[:4]
    top2_idx  = sorted_idx[:2]

    np.savez(os.path.join(MODEL_DIR, "top_indices_attention.npz"),
             top48=top48_idx,
             top32=top32_idx,
             top16=top16_idx,
             top8=top8_idx,
             top6=top6_idx,
             top4=top4_idx,
             top2=top2_idx)
    print("\nIndices attention sauvegardés dans top_indices_attention.npz")

    # ----------------------------
    # 7) Test final 62 canaux (20%)
    # ----------------------------
    test_metrics_62 = evaluate_on_test(
        model_62, X_test_n, y_test, device, num_channels=62, tag="62 canaux"
    )

    # ----------------------------
    # 8) Entraînement et test pour 48, 32, 16, 8, 6, 4, 2 canaux
    # ----------------------------
    configs = [
        ("48", top48_idx),
        ("32", top32_idx),
        ("16", top16_idx),
        ("8",  top8_idx),
        ("6",  top6_idx),
        ("4",  top4_idx),
        ("2",  top2_idx),
    ]

    all_test_results = {"62": test_metrics_62}

    for name, idxs in configs:
        n_ch = int(name)
        print(f"\n========== {n_ch} canaux (top-{n_ch}) ==========")

        X_train_sub = X_train_n[:, idxs, :]
        X_test_sub  = X_test_n[:, idxs, :]

        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_train_sub, y_train, test_size=0.10,
            stratify=y_train, random_state=123
        )

        save_path = os.path.join(MODEL_DIR, f"best_model_{name}_final.pth")
        plot_prefix = f"loss_acc_f1_{name}_final"

        model_sub, hist_sub, best_val = train_onecycle_amp(
            X_train_n=X_tr_sub,
            y_train=y_tr_sub,
            X_eval_n=X_val_sub,
            y_eval=y_val_sub,
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

        print(f"Best Val Acc ({n_ch} canaux) = {best_val*100:.2f}%")

        model_sub.load_state_dict(torch.load(save_path, map_location=device))
        model_sub.to(device)

        test_metrics = evaluate_on_test(
            model_sub, X_test_sub, y_test, device,
            num_channels=n_ch, tag=f"{n_ch} canaux"
        )
        all_test_results[name] = test_metrics

    # ----------------------------
    # 9) Résumé global
    # ----------------------------
    print("\n\n========== RÉSUMÉ GLOBAL TEST (20%) ==========")
    for k, m in all_test_results.items():
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
