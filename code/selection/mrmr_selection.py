import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def extract_features_from_eeg(X):
    """
    Convertit X (N,62,T) → X_feat (N,62)
    en utilisant la moyenne temporelle.
    """
    return X.mean(axis=2)


def mrmr_miq(X, y, K=None, random_state=42):
    """
    Implémentation mRMR-MIQ maison (version déterministe).
    X : (N,62)
    y : (N,)
    K : nombre de features à sélectionner (par défaut = toutes)
    """
    N, D = X.shape
    if K is None:
        K = D

    selected = []
    remaining = list(range(D))

    # 1) Pertinence MI(feature, label)
    relevance = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=random_state
    )

    # 2) Sélection initiale = feature la plus pertinente
    first = np.argmax(relevance)
    selected.append(first)
    remaining.remove(first)

    # 3) Sélection itérative MIQ
    for _ in range(1, K):
        scores = []

        for j in remaining:
            # pertinence
            rel = relevance[j]

            # redondance : MI(feature j, feature déjà sélectionnée)
            red_list = []
            for s in selected:
                mi_js = mutual_info_regression(
                    X[:, [j]], X[:, s].reshape(-1, 1),
                    random_state=random_state
                )[0]

                red_list.append(mi_js)

            red = np.mean(red_list) if len(red_list) > 0 else 0.0

            # score MIQ
            score = rel / (red + 1e-9)
            scores.append(score)

        j_best = remaining[np.argmax(scores)]
        selected.append(j_best)
        remaining.remove(j_best)

    return selected


def get_mrmr_ranking(X_train, y_train, random_state=42):
    """
    Pipeline complet mRMR (version déterministe).
    Retourne un classement complet des 62 électrodes.
    """
    X_feat = extract_features_from_eeg(X_train)
    ranking = mrmr_miq(X_feat, y_train, K=X_feat.shape[1], random_state=random_state)
    return ranking
