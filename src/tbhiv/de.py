"""Stage 4 - differential expression (Objective 1).

Implements the limma approach in pure NumPy/SciPy so it runs on Colab without R:

1. per-gene ordinary least squares with a design matrix (TB status + covariates such as site);
2. empirical-Bayes moderation of the gene variances (Smyth 2004, `fitFDist` / `squeezeVar`),
   which borrows strength across ~15k genes and is more reliable than plain t-tests at n~90;
3. Benjamini-Hochberg FDR.

It also provides the TB x HIV interaction model, which is the formal test of the hypothesis that
HIV weakens the active-vs-latent signal, and an attenuation ratio with a bootstrap CI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from statsmodels.stats.multitest import multipletests

from .data import Dataset
from .plots import INK_2, PALETTES, SLOTS, dodge_labels, savefig


# --------------------------------------------------------------------------------------
# Linear model + empirical Bayes
# --------------------------------------------------------------------------------------
def build_design(meta: pd.DataFrame, terms: dict, covariates=()) -> pd.DataFrame:
    """terms: {name: boolean/float Series}. Categorical covariates become 0/1 dummies
    (first level dropped). Constant covariates are skipped."""
    X = pd.DataFrame({"intercept": 1.0}, index=meta.index)
    for name, col in terms.items():
        X[name] = pd.Series(col, index=meta.index).astype(float)
    for cov in covariates:
        c = meta[cov]
        if c.nunique(dropna=True) < 2:
            continue
        if pd.api.types.is_numeric_dtype(c):
            X[cov] = (c - c.mean()).astype(float)
        else:
            d = pd.get_dummies(c.astype("category"), prefix=cov, drop_first=True, dtype=float)
            X = X.join(d)
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise ValueError(f"Design matrix is not full rank: {list(X.columns)}")
    return X


def lm_fit(Y: np.ndarray, X: np.ndarray):
    """Y genes x n, X n x p -> coef (genes x p), s2 (genes), df, stdev_unscaled (p)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = Y @ X @ XtX_inv
    resid = Y - coef @ X.T
    df = X.shape[0] - X.shape[1]
    s2 = (resid ** 2).sum(axis=1) / df
    return coef, s2, df, np.sqrt(np.diag(XtX_inv))


def trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x by Newton's method (limma::trigammaInverse)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_f_dist(s2: np.ndarray, df: float) -> tuple[float, float]:
    """Estimate prior degrees of freedom d0 and prior variance s0^2 (limma::fitFDist)."""
    s2 = np.maximum(s2, 1e-5 * np.median(s2))
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, df / 2)
    if evar > 0:
        d0 = 2 * trigamma_inverse(evar)
        s0_2 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s0_2 = np.inf, np.exp(emean)
    return d0, s0_2


def ebayes(coef, s2, df, stdev_unscaled, moderate=True):
    if moderate:
        d0, s0_2 = fit_f_dist(s2, df)
    else:
        d0, s0_2 = 0.0, 0.0
    if np.isinf(d0):
        s2_post = np.full_like(s2, s0_2)
        df_total = np.full_like(s2, df * len(s2), dtype=float)
    else:
        s2_post = (d0 * s0_2 + df * s2) / (d0 + df)
        df_total = np.minimum(df + d0, df * len(s2)) * np.ones_like(s2)
    t = coef / (stdev_unscaled[None, :] * np.sqrt(s2_post)[:, None])
    p = 2 * stats.t.sf(np.abs(t), df_total[:, None])
    return t, p, {"d0": float(d0), "s0_2": float(s0_2), "df_residual": int(df)}


def fit_coefficient(expr: pd.DataFrame, X: pd.DataFrame, coef_name: str, moderate=True) -> tuple[pd.DataFrame, dict]:
    Y = expr[X.index].values
    coef, s2, df, su = lm_fit(Y, X.values)
    t, p, prior = ebayes(coef, s2, df, su, moderate)
    j = list(X.columns).index(coef_name)
    res = pd.DataFrame({
        "log2FC": coef[:, j],
        "AveExpr": Y.mean(axis=1),
        "t": t[:, j],
        "P.Value": p[:, j],
    }, index=expr.index)
    res["adj.P.Val"] = multipletests(res["P.Value"], method="fdr_bh")[1]
    return res.sort_values("P.Value"), prior


def _mark(res: pd.DataFrame, fdr: float, min_lfc: float) -> pd.DataFrame:
    sig = (res["adj.P.Val"] < fdr) & (res["log2FC"].abs() >= min_lfc)
    res["significant"] = sig
    res["direction"] = np.where(~sig, "ns", np.where(res["log2FC"] > 0, "up", "down"))
    return res


# --------------------------------------------------------------------------------------
# Analyses used in the notebook
# --------------------------------------------------------------------------------------
def differential_expression(ds: Dataset, case="ATB", control="LTBI", covariates=("site",),
                            fdr=0.05, min_lfc=0.5) -> tuple[pd.DataFrame, dict]:
    """Active TB vs latent TB within whatever samples `ds` holds (e.g. HIV- only).
    Positive log2FC = higher in active TB."""
    d = ds.subset(ds.meta["tb_status"].isin([case, control]))
    X = build_design(d.meta, {"tb": d.meta["tb_status"] == case}, covariates)
    res, prior = fit_coefficient(d.expr, X, "tb")
    info = {**prior, "n_case": int((d.meta["tb_status"] == case).sum()),
            "n_control": int((d.meta["tb_status"] == control).sum()), "design": list(X.columns)}
    return _mark(res, fdr, min_lfc), info


def interaction_model(ds: Dataset, covariates=("site",), fdr=0.05) -> tuple[pd.DataFrame, dict]:
    """expr ~ TB + HIV + TB:HIV + site, on active + latent TB samples from both HIV groups.

    tb        : active-vs-latent effect in HIV- patients
    tb_x_hiv  : how much that effect changes in HIV+ patients (HIV+ effect = tb + tb_x_hiv)
    A gene is 'attenuated' when tb_x_hiv is significant and opposite in sign to tb.
    """
    d = ds.subset(ds.meta["tb_status"].isin(["ATB", "LTBI"]) & ds.meta["hiv_status"].isin(["HIV-", "HIV+"]))
    tb = (d.meta["tb_status"] == "ATB").astype(float)
    hiv = (d.meta["hiv_status"] == "HIV+").astype(float)
    X = build_design(d.meta, {"tb": tb, "hiv": hiv, "tb_x_hiv": tb * hiv}, covariates)
    Y = d.expr[X.index].values
    coef, s2, df, su = lm_fit(Y, X.values)
    t, p, prior = ebayes(coef, s2, df, su)
    cols = list(X.columns)
    jt, ji = cols.index("tb"), cols.index("tb_x_hiv")
    res = pd.DataFrame({
        "tb_effect_HIVneg": coef[:, jt],
        "tb_effect_HIVpos": coef[:, jt] + coef[:, ji],
        "interaction": coef[:, ji],
        "t_interaction": t[:, ji],
        "P.Value": p[:, ji],
    }, index=d.expr.index)
    res["adj.P.Val"] = multipletests(res["P.Value"], method="fdr_bh")[1]
    res["attenuated"] = (res["adj.P.Val"] < fdr) & (np.sign(res["interaction"]) == -np.sign(res["tb_effect_HIVneg"]))
    res["amplified"] = (res["adj.P.Val"] < fdr) & ~res["attenuated"]
    return res.sort_values("P.Value"), prior


def compare_gene_lists(res_neg: pd.DataFrame, res_pos: pd.DataFrame) -> dict:
    """Overlap of significant genes in HIV- and HIV+, with a hypergeometric enrichment test."""
    universe = res_neg.index.intersection(res_pos.index)
    a = set(res_neg.index[res_neg["significant"]]) & set(universe)
    b = set(res_pos.index[res_pos["significant"]]) & set(universe)
    both = a & b
    N, K, n, k = len(universe), len(a), len(b), len(both)
    p = stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
    concordant = sum(np.sign(res_neg.loc[g, "log2FC"]) == np.sign(res_pos.loc[g, "log2FC"]) for g in both)
    m = res_neg[["log2FC"]].merge(res_pos[["log2FC"]], how="left", left_index=True, right_index=True,
                                       suffixes=("_HIVneg", "_HIVpos")).loc[universe]
    return {
        "n_HIVneg": K, "n_HIVpos": n, "n_both": k,
        "only_HIVneg": sorted(a - b), "only_HIVpos": sorted(b - a), "shared": sorted(both),
        "jaccard": k / len(a | b) if (a | b) else 0.0,
        "hypergeom_p": float(p),
        "shared_same_direction": int(concordant),
        "lfc_spearman_all_genes": float(stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1]).statistic),
    }


def attenuation_ratio(ds: Dataset, genes, covariates=("site",), n_boot=200, seed=0) -> dict:
    """Ratio of the average signed TB effect in HIV+ to that in HIV-, over `genes`.

    Genes should be chosen symmetrically (e.g. significant in the pooled analysis): selecting
    them in HIV- only would inflate HIV- effects (winner's curse) and bias the ratio below 1.
    1.0 = same strength; 0.6 = the HIV+ signal is 40% weaker. 95% CI from a bootstrap that
    resamples patients within each TB x HIV cell.
    """
    d = ds.subset(ds.meta["tb_status"].isin(["ATB", "LTBI"]))
    genes = [g for g in genes if g in d.expr.index]
    if not genes:
        return {"ratio": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_genes": 0}
    E = d.expr.loc[genes]

    def effects(meta, expr):
        out = {}
        for h in ("HIV-", "HIV+"):
            m = meta[meta["hiv_status"] == h]
            X = build_design(m, {"tb": m["tb_status"] == "ATB"}, covariates)
            coef, *_ = lm_fit(expr[X.index].values, X.values)
            out[h] = coef[:, 1]
        return out

    base = effects(d.meta, E)
    sign = np.sign(base["HIV-"] + base["HIV+"])  # pooled direction

    def ratio(eff):
        return float(np.mean(sign * eff["HIV+"]) / np.mean(sign * eff["HIV-"]))

    r0 = ratio(base)
    rng = np.random.default_rng(seed)
    cells = d.meta.groupby(["hiv_status", "tb_status"]).groups
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.asarray(v), size=len(v), replace=True) for v in cells.values()])
        new_ids = [f"{s}#{i}" for i, s in enumerate(idx)]
        m = d.meta.loc[idx].copy()
        m.index = new_ids
        Eb = E[idx].copy()
        Eb.columns = new_ids
        boots.append(ratio(effects(m, Eb)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ratio": r0, "ci_low": float(lo), "ci_high": float(hi), "n_genes": len(genes), "n_boot": n_boot}


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def volcano(res: pd.DataFrame, title: str, fdr=0.05, min_lfc=0.5, label_top=12, ax=None, path=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5))
    y = -np.log10(res["P.Value"].clip(lower=1e-300))
    for d, col in PALETTES["direction"].items():
        m = res["direction"] == d
        ax.scatter(res.loc[m, "log2FC"], y[m], s=6 if d == "ns" else 10, c=col,
                   alpha=0.5 if d == "ns" else 0.85, linewidths=0,
                   label=f"{ {'up': 'higher in active TB', 'down': 'lower in active TB', 'ns': 'not significant'}[d]} ({m.sum()})")
    sig = res[res["significant"]]
    if len(sig):
        p_cut = sig["P.Value"].max()
        ax.axhline(-np.log10(p_cut), color="#c3c2b7", lw=0.8, ls="--")
    for x in (-min_lfc, min_lfc):
        ax.axvline(x, color="#c3c2b7", lw=0.8, ls="--")
    ax.set(xlabel="log2 fold change (active TB vs latent TB)", ylabel="-log10 p", title=title)
    top = res[res["significant"]].head(label_top)
    if len(top):
        dodge_labels(ax, list(top["log2FC"]), list(-np.log10(top["P.Value"].clip(lower=1e-300))), list(top.index))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    if own:
        fig.tight_layout()
        return savefig(fig, path)
    return ax


def lfc_scatter(res_neg, res_pos, highlight=None, path=None):
    m = res_neg[["log2FC", "significant"]].merge(res_pos[["log2FC", "significant"]], how="left", left_index=True,
                                                right_index=True, suffixes=("_neg", "_pos")).dropna()
    cat = np.select(
        [m["significant_neg"] & m["significant_pos"], m["significant_neg"], m["significant_pos"]],
        ["both", "HIV- only", "HIV+ only"], "neither")
    colors = {"neither": "#d9d8d2", "HIV- only": SLOTS[0], "HIV+ only": SLOTS[1], "both": SLOTS[6]}
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for c in ["neither", "HIV- only", "HIV+ only", "both"]:
        k = cat == c
        ax.scatter(m.loc[k, "log2FC_neg"], m.loc[k, "log2FC_pos"], s=6 if c == "neither" else 10,
                   c=colors[c], linewidths=0, alpha=0.8, label=f"{c} ({k.sum()})")
    lim = np.nanmax(np.abs(m[["log2FC_neg", "log2FC_pos"]].values)) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--", label="equal effect (y = x)")
    if highlight:
        for g in highlight:
            if g in m.index:
                ax.annotate(g, (m.loc[g, "log2FC_neg"], m.loc[g, "log2FC_pos"]), fontsize=7, color=INK_2)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="log2FC active vs latent TB, HIV-",
           ylabel="log2FC active vs latent TB, HIV+",
           title="Same genes, two HIV groups\n(points below the diagonal on the right = weaker in HIV+)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    return savefig(fig, path)


def overlap_diagram(cmp: dict, path=None):
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.add_patch(Circle((0.38, 0.5), 0.3, color=SLOTS[0], alpha=0.35, lw=0))
    ax.add_patch(Circle((0.62, 0.5), 0.3, color=SLOTS[1], alpha=0.35, lw=0))
    ax.text(0.22, 0.5, len(cmp["only_HIVneg"]), ha="center", va="center", fontsize=13)
    ax.text(0.5, 0.5, cmp["n_both"], ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.78, 0.5, len(cmp["only_HIVpos"]), ha="center", va="center", fontsize=13)
    ax.text(0.30, 0.9, f"HIV- ({cmp['n_HIVneg']})", ha="center", color=INK_2)
    ax.text(0.70, 0.9, f"HIV+ ({cmp['n_HIVpos']})", ha="center", color=INK_2)
    ax.text(0.5, 0.05, f"Jaccard {cmp['jaccard']:.2f} | hypergeometric p = {cmp['hypergeom_p']:.1e}",
            ha="center", fontsize=8.5, color=INK_2)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Overlap of significant genes")
    ax.axis("off")
    return savefig(fig, path)
