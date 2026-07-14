"""Cluster diagnostic on AURA-cleaned features.

Answers: does 'low' form a separable cluster, or is it a continuum
between 'normal' and 'high'?

Outputs:
  paper/figures/fig4_umap_load3.pdf   — 2D UMAP coloured by load_3
  paper/figures/fig5_tsne_load3.pdf   — 2D t-SNE coloured by load_3
  prints   — k-NN purity per class, PERMANOVA-style F statistic
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv" / "eeg_window_features_aura.csv"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

META = {"file_name", "subject_id", "task_type", "condition_4",
        "load_3", "load_2", "window_start_idx", "window_end_idx"}

COLORS = {"normal": "#3b6fb0", "low": "#e69138", "high": "#c0392b"}


def feature_cols(df):
    cs = [c for c in df.columns if c not in META]
    return [c for c in cs if pd.api.types.is_numeric_dtype(df[c])]


def per_subject_calibrate(df, cols):
    out = df.copy()
    for s, g in df.groupby("subject_id"):
        base = g[g.load_3 == "normal"]
        if len(base) < 5:
            mu, sd = g[cols].mean(), g[cols].std().replace(0, 1)
        else:
            mu, sd = base[cols].mean(), base[cols].std().replace(0, 1)
        idx = g.index
        z = (g[cols].values - mu.values) / sd.values
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out.loc[idx, cols] = np.clip(z, -50.0, 50.0)
    return out


def knn_purity(X, y, k=15):
    """Fraction of k nearest neighbours that share the class, per class."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    same = (y[idx] == y[:, None]).mean(axis=1)
    labels = np.unique(y)
    return {lab: float(same[y == lab].mean()) for lab in labels}


def permanova_F(X, y, perm=200, seed=0):
    """Pseudo-F from PERMANOVA on pairwise sq distances. Returns F + p-value."""
    rng = np.random.default_rng(seed)
    n = len(y)
    # subsample to keep it fast
    if n > 1500:
        keep = rng.choice(n, 1500, replace=False)
        X = X[keep]; y = y[keep]; n = 1500
    D = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    SS_T = D.sum() / (2 * n)
    labels = np.unique(y)
    SS_W = 0.0
    for lab in labels:
        mask = (y == lab); m = mask.sum()
        if m < 2:
            continue
        sub = D[np.ix_(mask, mask)]
        SS_W += sub.sum() / (2 * m)
    a = len(labels)
    F = ((SS_T - SS_W) / (a - 1)) / (SS_W / (n - a))
    # permutation null
    Fperm = np.zeros(perm)
    for i in range(perm):
        yp = rng.permutation(y)
        ss_w = 0.0
        for lab in labels:
            mask = (yp == lab); m = mask.sum()
            if m < 2:
                continue
            sub = D[np.ix_(mask, mask)]
            ss_w += sub.sum() / (2 * m)
        Fperm[i] = ((SS_T - ss_w) / (a - 1)) / (ss_w / (n - a))
    p = (Fperm >= F).mean()
    return float(F), float(p)


def plot_2d(emb, y, title, outfile):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for lab in ["normal", "low", "high"]:
        m = (y == lab)
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.55,
                   c=COLORS[lab], label=lab, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"saved {outfile}")


def main():
    df = pd.read_csv(CSV)
    cols = feature_cols(df)
    df_c = per_subject_calibrate(df, cols)
    df_c = df_c[df_c.load_3.isin(["normal", "low", "high"])]
    X = df_c[cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df_c.load_3.astype(str).to_numpy()
    print(f"X: {X.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # global StandardScaler so UMAP/t-SNE see comparable scales
    Xs = StandardScaler().fit_transform(X)

    # ---- k-NN purity ----
    purity = knn_purity(Xs, y, k=15)
    print("\nk-NN purity (k=15, fraction of nearest neighbours sharing the class):")
    for k, v in purity.items():
        print(f"  {k:7s}: {v:.3f}")

    # ---- PERMANOVA pseudo-F (3-class) ----
    F3, p3 = permanova_F(Xs, y, perm=200)
    print(f"\nPERMANOVA pseudo-F (3-class): F={F3:.2f}, p={p3:.3f}")

    # ---- low vs high only ----
    mask_lh = np.isin(y, ["low", "high"])
    F2, p2 = permanova_F(Xs[mask_lh], y[mask_lh], perm=200)
    print(f"PERMANOVA pseudo-F (low vs high only): F={F2:.2f}, p={p2:.3f}")
    plh = knn_purity(Xs[mask_lh], y[mask_lh], k=15)
    print("k-NN purity restricted to low+high:")
    for k, v in plh.items():
        print(f"  {k:7s}: {v:.3f}")

    # ---- UMAP ----
    print("\nFitting UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean",
                       random_state=42, n_jobs=-1)
    emb_umap = reducer.fit_transform(Xs)
    plot_2d(emb_umap, y,
            "UMAP, AURA features per-subject calibrated",
            FIG_DIR / "fig4_umap_load3.pdf")

    # ---- t-SNE on a subsample for speed ----
    print("Fitting t-SNE (subsample 2500)...")
    rng = np.random.default_rng(0)
    if len(Xs) > 2500:
        idx = rng.choice(len(Xs), 2500, replace=False)
        Xs_sub, y_sub = Xs[idx], y[idx]
    else:
        Xs_sub, y_sub = Xs, y
    emb_tsne = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=42).fit_transform(Xs_sub)
    plot_2d(emb_tsne, y_sub,
            "t-SNE, AURA features per-subject calibrated",
            FIG_DIR / "fig5_tsne_load3.pdf")


if __name__ == "__main__":
    main()
