import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from train_utils import set_seed, device, EEGDataset
from models_conv_emotion import CompactEmotionCNNArticle
from train_utils import train_emotion_cnn_article


# ---------- Normalisation robuste ----------
def robust_norm(X, median, iqr, clip=3.0):
    Xn = (X - median) / iqr
    return np.clip(Xn, -clip, clip)


# ---------- Évaluation ----------
def evaluate(model, X_test, y_test, num_channels):

    from torch.utils.data import DataLoader

    loader = DataLoader(
        EEGDataset(X_test, y_test, augment=False),
        batch_size=64,
        shuffle=False
    )

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n===== TEST FINAL =====")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")
    print(f"ROC-AUC  : {roc:.3f}")
    print(f"MCC      : {mcc:.3f}")

    return {
        "Acc": acc, "Prec": prec, "Rec": rec,
        "F1": f1, "ROC": roc, "MCC": mcc
    }


# -------------------------------------------------------
#                         MAIN
# -------------------------------------------------------
def main():

    set_seed(42)
    print("Device :", device)

    BASE = os.path.expanduser("~/EEG_Emotion_Project")
    DATA_DIR = os.path.join(BASE, "data")
    MODEL_DIR = os.path.join(BASE, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    npz_path = os.path.join(DATA_DIR, "dataset_seed_windowed_compressed.npz")
    npz = np.load(npz_path, allow_pickle=True)

    X = npz["X"]
    y = npz["y"]

    # Assurer la forme N, C, T
    if X.shape[1] != 62 and X.shape[2] == 62:
        X = np.transpose(X, (0, 2, 1))

    N, C, T = X.shape
    assert C == 62, "Le modèle de l’article doit utiliser 62 canaux."

    # -------- Global split 80/20 --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # -------- Normalisation robuste --------
    median = np.median(X_train, axis=(0, 2), keepdims=True)
    q75 = np.percentile(X_train, 75, axis=(0, 2), keepdims=True)
    q25 = np.percentile(X_train, 25, axis=(0, 2), keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1e-6

    X_train_n = robust_norm(X_train, median, iqr).astype(np.float32)
    X_test_n  = robust_norm(X_test, median, iqr).astype(np.float32)

    # -------- Internal split 72/8 --------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_n, y_train, test_size=0.10, stratify=y_train, random_state=123
    )

    # -------- Training --------
    save_path = os.path.join(MODEL_DIR, "best_compactcnn_article.pth")

    model = train_emotion_cnn_article(
    X_tr, y_tr,
    X_val, y_val,
    num_channels=62,
    seq_len=T,
    num_classes=3,
    save_path=save_62
)



    # -------- Recharger best checkpoint --------
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    # -------- Test final --------
    evaluate(model, X_test_n, y_test, num_channels=62)


# POINT D’ENTRÉE OBLIGATOIRE
if __name__ == "__main__":
    main()
