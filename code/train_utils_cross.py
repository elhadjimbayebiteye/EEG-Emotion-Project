import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    recall_score,
)


from models_with_cross import EEGFormerHybridV3


# =========================
# 1) Seed + device
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_utils_cross] Device utilisé : {device}")


# =========================
# 2) Dataset "en RAM"
# =========================
class EEGDataset(Dataset):
    def __init__(self, X, y, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)  # [N,C,T]
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]  # [C,T]

        if self.augment:
            if torch.rand(1).item() < 0.45:
                x = x + 0.02 * torch.randn_like(x)
            if torch.rand(1).item() < 0.40:
                shift = torch.randint(-8, 9, (1,)).item()
                x = torch.roll(x, shifts=shift, dims=1)
            if torch.rand(1).item() < 0.30:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                x = x * scale
            if torch.rand(1).item() < 0.30:
                L = torch.randint(8, 21, (1,)).item()
                s = torch.randint(0, x.shape[1] - L + 1, (1,)).item()
                x[:, s:s + L] = 0.0
            if torch.rand(1).item() < 0.20:
                C = x.shape[0]
                k = max(1, int(0.1 * C))
                idxs = torch.randperm(C)[:k]
                perm = idxs[torch.randperm(k)]
                x[idxs] = x[perm]

        return x, y


# =========================
# 3) Poids de classes + FocalLoss
# =========================
def compute_class_weights(y_np: np.ndarray) -> torch.Tensor:
    uniq, cnts = np.unique(y_np, return_counts=True)
    freq = cnts / cnts.sum()
    cls_w = 1.0 / (freq + 1e-9)
    cls_w = cls_w / cls_w.sum() * len(uniq)
    return torch.tensor(cls_w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 1.7, label_smoothing: float = 0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# =========================
# 4) Entraînement OneCycle + AMP (CROSS)
# =========================
def train_onecycle_amp(
    X_train_n=None,
    y_train=None,
    X_eval_n=None,
    y_eval=None,
    num_channels=62,
    save_name="best_model_cross.pth",
    plot_name_prefix="training_plot_cross",
    batch_size=128,
    epochs=100,
    max_lr=1.5e-3,
    weight_decay=1.68e-5,
    use_focal=True,
    label_smoothing=0.02,
    embed_dim=256,
    gate_hidden=64,
    pdrop_cnn=0.30,
    pdrop_head=0.15,
    heads=6,
    patience=12,
    div_factor=20.0,
    plots_dir=None,
    train_loader=None,
    eval_loader=None,
    num_classes=None,
    seq_len=None,
):
    assert y_train is not None, "y_train est requis pour calculer les poids de classes."

    if plots_dir is None:
        plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # ----- Dataloaders (mode RAM) -----
    if train_loader is None:
        assert X_train_n is not None, "X_train_n doit être fourni si train_loader est None."
        train_loader = DataLoader(
            EEGDataset(X_train_n, y_train, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    if eval_loader is None and X_eval_n is not None and y_eval is not None:
        eval_loader = DataLoader(
            EEGDataset(X_eval_n, y_eval, augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    # ----- num_classes / seq_len -----
    if num_classes is None:
        num_classes = len(np.unique(y_train))

    if seq_len is None:
        assert X_train_n is not None, "seq_len non fourni et X_train_n absent."
        seq_len = X_train_n.shape[2]

    # ----- Modèle CROSS -----
    model = EEGFormerHybridV3(
        num_channels=num_channels,
        seq_len=seq_len,
        num_classes=num_classes,
        embed_dim=embed_dim,
        gate_hidden=gate_hidden,
        pdrop_cnn=pdrop_cnn,
        pdrop_head=pdrop_head,
        heads=heads,
    ).to(device)

    cls_w = compute_class_weights(np.array(y_train)).to(device)
    if use_focal:
        criterion = FocalLoss(alpha=cls_w, gamma=1.7, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.25,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=1e4,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    clip_max_norm = 1.0

    hist = {
        "train_loss": [],
        "eval_loss": [],
        "train_acc": [],
        "eval_acc": [],
        "eval_f1": [],
        "eval_recall": [],
    }
    best_eval_acc = 0.0
    patience_cnt = 0

    print(f"\n[train_onecycle_amp_cross] Entraînement — {num_channels} canaux (CROSS)\n")

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t_loss += loss.item() * xb.size(0)
            t_corr += (out.argmax(1) == yb).sum().item()
            t_tot += yb.size(0)

        train_loss = t_loss / t_tot
        train_acc = t_corr / t_tot

        # ================= EVAL =================
        if eval_loader is not None:
            model.eval()
            e_loss, e_corr, e_tot = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                for xb, yb in eval_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    e_loss += loss.item() * xb.size(0)

                    preds = out.argmax(1).cpu().numpy()
                    labs = yb.cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labs)
                    e_corr += (preds == labs).sum()
                    e_tot += len(labs)

            eval_loss = e_loss / e_tot
            eval_acc = e_corr / e_tot
            eval_f1 = f1_score(all_labels, all_preds, average="macro")
            eval_rec = recall_score(all_labels, all_preds, average="macro")
        else:
            eval_loss = float("nan")
            eval_acc = float("nan")
            eval_f1 = float("nan")
            eval_rec = float("nan")

        hist["train_loss"].append(train_loss)
        hist["eval_loss"].append(eval_loss)
        hist["train_acc"].append(train_acc)
        hist["eval_acc"].append(eval_acc)
        hist["eval_f1"].append(eval_f1)
        hist["eval_recall"].append(eval_rec)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | Eval Loss={eval_loss:.4f} | "
            f"Train Acc={train_acc*100:6.2f}% | Eval Acc={eval_acc*100:6.2f}% | "
            f"F1={eval_f1:.3f} | Recall={eval_rec:.3f}"
        )

        if not np.isnan(eval_acc) and eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            patience_cnt = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("⏹ Early stopping déclenché (CROSS).")
                break

    # ============ Courbes ============
    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(hist["train_loss"], label="Train Loss")
    if not all(np.isnan(x) for x in hist["eval_loss"]):
        plt.plot(hist["eval_loss"], label="Eval Loss")
    plt.legend()
    plt.title("Pertes (CROSS)")
    plt.grid(alpha=0.3)

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(np.array(hist["train_acc"]) * 100, label="Train Acc")
    if not all(np.isnan(x) for x in hist["eval_acc"]):
        plt.plot(np.array(hist["eval_acc"]) * 100, label="Eval Acc")
    plt.legend()
    plt.title("Accuracy (CROSS)")
    plt.grid(alpha=0.3)

    # F1 / Recall
    plt.subplot(1, 3, 3)
    if not all(np.isnan(x) for x in hist["eval_f1"]):
        plt.plot(np.array(hist["eval_f1"]) * 100, label="Eval F1")
        plt.plot(np.array(hist["eval_recall"]) * 100, label="Eval Recall")
        plt.legend()
        plt.title("F1 / Recall (CROSS)")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(plots_dir, f"{plot_name_prefix}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Courbes CROSS sauvegardées :", png_path)

    return model, hist, best_eval_acc
