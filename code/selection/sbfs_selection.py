
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def extract_simple_features(X):
    # X : (N, C, T)
    mean = X.mean(axis=2)
    var = X.var(axis=2)
    energy = (X ** 2).mean(axis=2)
    return np.concatenate([mean, var, energy], axis=1)


def evaluate_subset(X_feat, y):
    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )
    return cross_val_score(
        clf, X_feat, y,
        cv=3,
        scoring="accuracy"
    ).mean()


def sbfs_select_channels(X, y, target_k):
    """
    X : (N, C, T) -> non normalisé OK
    """
    N, C, T = X.shape
    X_feat = extract_simple_features(X)

    channels = list(range(C))
    best_score = evaluate_subset(X_feat, y)

    while len(channels) > target_k:
        scores = []

        for ch in channels:
            tmp = [c for c in channels if c != ch]

            idx = []
            for c in tmp:
                idx.extend([3*c, 3*c+1, 3*c+2])

            score = evaluate_subset(X_feat[:, idx], y)
            scores.append((score, ch))

        scores.sort(reverse=True)
        best_score, removed = scores[0]
        channels.remove(removed)

        print(f"[SBFS] remove ch={removed} | score={best_score:.4f}")

    return np.array(channels, dtype=int)
