import numpy as np
from sklearn.neighbors import NearestNeighbors

def relieff_scores(X, y, n_neighbors=10):
    """
    Implémentation simplifiée de ReliefF pour données EEG.
    X : (N, C) = N échantillons, C canaux (feature par canal)
    y : (N,)
    Retourne : scores (C,)
    """

    N, C = X.shape
    scores = np.zeros(C)

    # On utilise les k plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Pour chaque échantillon
    for i in range(N):
        Xi = X[i]
        yi = y[i]

        # voisins (le 1er est lui-même → on l'enlève)
        neigh_ids = indices[i][1:]

        for neigh in neigh_ids:
            Xn = X[neigh]
            yn = y[neigh]

            if yn == yi:  # hit
                scores -= np.abs(Xi - Xn)
            else:         # miss
                scores += np.abs(Xi - Xn)

    # Normalisation
    scores = scores / np.max(np.abs(scores) + 1e-9)
    return scores


def get_relieff_ranking(X_train_n, y_train, n_neighbors=10):
    """
    Calcule les scores ReliefF par canal sur X_train_n (N, 62, T)
    → Résume par moyenne temporelle → (N, 62)
    → Applique ReliefF → Renvoie ranking trié
    """

    # Résume chaque fenêtre en (N, 62)
    X_flat = X_train_n.mean(axis=2)   # moyenne temporelle
    y_flat = y_train

    # Calcul ReliefF
    scores = relieff_scores(X_flat, y_flat, n_neighbors=n_neighbors)

    # Tri (du plus important au moins important)
    ranking = np.argsort(scores)[::-1]

    return scores, ranking
