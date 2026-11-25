"""Stage 2-3 - quality control and exploratory analysis (distributions, PCA, batch effects)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from .data import Dataset
from .plots import PALETTES, savefig


def filter_low_expression(ds: Dataset, quantile: float = 0.3) -> tuple[Dataset, dict]:
    """Drop genes whose mean log2 expression is in the bottom `quantile` (mostly unexpressed
    Illumina probes that only add noise and multiple-testing burden)."""
    means = ds.expr.mean(axis=1)
    cut = means.quantile(quantile)
    keep = means[means > cut].index
    out = Dataset(ds.name, ds.expr.loc[keep], ds.meta)
    return out, {"genes_before": ds.shape[0], "genes_after": len(keep), "mean_log2_cutoff": float(cut)}


def sample_qc_table(ds: Dataset) -> pd.DataFrame:
    e = ds.expr
    corr = np.corrcoef(e.values.T)
    np.fill_diagonal(corr, np.nan)
    t = pd.DataFrame({
        "median": e.median(),
        "iqr": e.quantile(0.75) - e.quantile(0.25),
        "mean_corr_to_others": np.nanmean(corr, axis=1),
    })
    # Robust z-score of each sample's average correlation: low values = outlier arrays
    med = t["mean_corr_to_others"].median()
    mad = stats.median_abs_deviation(t["mean_corr_to_others"], scale="normal")
    t["corr_robust_z"] = (t["mean_corr_to_others"] - med) / (mad if mad > 0 else 1)
    t["outlier"] = t["corr_robust_z"] < -4
    return t.join(ds.meta[["tb_status", "hiv_status", "site"]])


def plot_distributions(ds: Dataset, order_by=("site", "hiv_status", "tb_status"), max_samples=600, path=None):
    meta = ds.meta.sort_values(list(order_by))
    cols = meta.index[:max_samples]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    q = ds.expr[cols].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    x = np.arange(len(cols))
    colors = meta.loc[cols, "site"].map(PALETTES["site"]).fillna("grey")
    axes[0].fill_between(x, q.loc[0.05], q.loc[0.95], color="lightgrey", label="5-95%")
    axes[0].fill_between(x, q.loc[0.25], q.loc[0.75], color="silver", label="IQR")
    axes[0].scatter(x, q.loc[0.5], c=colors, s=4, label="median")
    axes[0].set(xlabel="samples (ordered by site, HIV, TB)", ylabel="log2 expression",
                title="Per-sample expression distribution")
    axes[0].legend(loc="upper right", fontsize=8)
    for site, grp in meta.loc[cols].groupby("site"):
        vals = ds.expr[grp.index].values.ravel()
        axes[1].hist(vals, bins=80, histtype="step", density=True, label=site,
                     color=PALETTES["site"].get(site, "grey"))
    axes[1].set(xlabel="log2 expression", title="Density by site")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, path)


def run_pca(ds: Dataset, n_components: int = 10, top_variable: int = 5000):
    """PCA on the most variable genes (centred, not scaled)."""
    v = ds.expr.var(axis=1).sort_values(ascending=False)
    X = ds.expr.loc[v.index[:top_variable]].T
    X = X - X.mean()
    n = min(n_components, min(X.shape) - 1)
    pca = PCA(n_components=n, random_state=0).fit(X)
    scores = pd.DataFrame(pca.transform(X), index=X.index, columns=[f"PC{i+1}" for i in range(n)])
    return scores, pd.Series(pca.explained_variance_ratio_, index=scores.columns)


def pc_associations(scores: pd.DataFrame, meta: pd.DataFrame, factors=("tb_status", "hiv_status", "site"), n_pcs=5):
    """R-squared of each PC explained by each factor (one-way ANOVA). Big R2 for 'site' on an
    early PC means a site/batch effect that the DE model must adjust for."""
    rows = []
    for pc in scores.columns[:n_pcs]:
        for f in factors:
            y = scores[pc]
            g = meta.loc[scores.index, f]
            ok = g.notna()
            y, g = y[ok], g[ok]
            groups = [y[g == lvl].values for lvl in g.unique()]
            if len(groups) < 2:
                continue
            ss_tot = ((y - y.mean()) ** 2).sum()
            ss_between = sum(len(x) * (x.mean() - y.mean()) ** 2 for x in groups)
            p = stats.f_oneway(*groups).pvalue
            rows.append({"PC": pc, "factor": f, "R2": ss_between / ss_tot, "p": p})
    return pd.DataFrame(rows).pivot(index="PC", columns="factor", values="R2").loc[list(scores.columns[:n_pcs])]


def plot_pca(scores, var, meta, color_by=("tb_status", "hiv_status", "site"), path=None):
    fig, axes = plt.subplots(1, len(color_by), figsize=(5 * len(color_by), 4.4))
    for ax, f in zip(np.atleast_1d(axes), color_by):
        pal = PALETTES.get(f, {})
        for lvl, idx in meta.loc[scores.index].groupby(f).groups.items():
            ax.scatter(scores.loc[idx, "PC1"], scores.loc[idx, "PC2"], s=12, alpha=0.75,
                       label=f"{lvl} (n={len(idx)})", color=pal.get(lvl))
        ax.set(xlabel=f"PC1 ({var['PC1']:.1%})", ylabel=f"PC2 ({var['PC2']:.1%})", title=f"PCA coloured by {f}")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return savefig(fig, path)
