"""Small, transparent separability statistics for the hypergraph features.

The manuscript does not do machine-learning classification.  To compare
feature sets we use three easy-to-interpret quantities: the Kruskal-Wallis
statistic per feature, a Fisher-style between/within variance ratio, and the
accuracy of a nearest-centroid rule under stratified cross-validation.
"""
import numpy as np
from scipy import stats as sstats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score


def to_matrix(feat_list, keys):
    return np.array([[d.get(k, 0.0) for k in keys] for d in feat_list], float)


def kw_table(F, y, keys):
    rows = []
    for j, k in enumerate(keys):
        groups = [F[y == c, j] for c in np.unique(y)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            rows.append({"feature": k, "H": 0.0, "p": 1.0})
            continue
        H, p = sstats.kruskal(*groups)
        rows.append({"feature": k, "H": float(H), "p": float(p)})
    return rows


def fisher_ratio(F, y):
    mu = F.mean(0)
    num = np.zeros(F.shape[1])
    den = np.zeros(F.shape[1])
    for c in np.unique(y):
        Fc = F[y == c]
        num += len(Fc) * (Fc.mean(0) - mu) ** 2
        den += ((Fc - Fc.mean(0)) ** 2).sum(0)
    r = num / np.maximum(den, 1e-12)
    return float(np.mean(r)), r.tolist()


def nearest_centroid_cv(F, y, n_splits=5, seed=0):
    y = np.asarray(y)
    n_min = min(np.bincount(y)[np.bincount(y) > 0])
    n_splits = int(max(2, min(n_splits, n_min)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(F, y):
        mu = F[tr].mean(0)
        sd = F[tr].std(0) + 1e-9
        Ztr = (F[tr] - mu) / sd
        Zte = (F[te] - mu) / sd
        cent = {c: Ztr[y[tr] == c].mean(0) for c in np.unique(y[tr])}
        pred = np.array([min(cent, key=lambda c: np.linalg.norm(z - cent[c]))
                         for z in Zte])
        accs.append(float(np.mean(pred == y[te])))
    return float(np.mean(accs)), float(np.std(accs))


def silhouette(F, y):
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 5:
        return 0.0
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    try:
        return float(silhouette_score(Z, y))
    except Exception:
        return 0.0


def summarise(F, y, keys):
    fr, per = fisher_ratio(F, y)
    acc, acc_sd = nearest_centroid_cv(F, y)
    return {
        "kw": kw_table(F, y, keys),
        "fisher_mean": fr,
        "fisher_per_feature": dict(zip(keys, per)),
        "nc_cv_acc": acc,
        "nc_cv_std": acc_sd,
        "silhouette": silhouette(F, y),
    }
