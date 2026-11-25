"""Stage 7-8 - machine learning with baselines, HIV transfer test, external validation (Objective 3).

Design choices (each guards against a common way ML results get overstated):
* Feature filtering (top-k genes by F-test) happens INSIDE each training fold, never on the full
  data, so test folds never influence which genes are chosen.
* Repeated stratified K-fold CV (stratified on TB status x site), AUC with bootstrap 95% CIs.
* Baselines: majority class (AUC 0.5 by construction), the published Sweeney 3-gene score and
  the Kaforou 27-gene score. A model is only interesting if it beats these.
* Decision thresholds are chosen to reach a target sensitivity (default 90%, WHO triage TPP)
  on TRAINING predictions, then applied unchanged to the test group. That is what reveals how
  many active-TB cases a model tuned on HIV- patients would miss in HIV+ patients.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .plots import INK_2, MUTED, PALETTES, SLOTS, savefig
from .signatures import available, signature_score

MODEL_COLORS = {
    "Majority class": MUTED,
    "Sweeney 3-gene score": SLOTS[2],
    "Kaforou 27-gene score": SLOTS[3],
    "L1 logistic regression": SLOTS[6],
    "Random forest": SLOTS[4],
}


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class SignatureScore(ClassifierMixin, BaseEstimator):
    """A published signature as a classifier: score = mean(up) - mean(down), then a 1-D
    logistic regression fitted on the training scores turns it into a probability."""

    def __init__(self, signature: str = "Sweeney_3"):
        self.signature = signature

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SignatureScore needs a DataFrame with gene-symbol columns")
        return signature_score(X.T, self.signature).values.reshape(-1, 1)

    def fit(self, X, y):
        self.genes_used_ = available(self.signature, X.columns)
        self.lr_ = LogisticRegression().fit(self._score(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._score(X))

    def predict(self, X):
        return self.lr_.predict(self._score(X))


def _l1_logistic(C=0.1, random_state=0):
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    common = dict(solver="liblinear", C=C, max_iter=5000, random_state=random_state, class_weight="balanced")
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


def make_models(cfg, n_features: int, genes=None, include=None) -> dict:
    k = min(cfg.k_features, n_features)
    rs = cfg.random_state
    models = {
        "Majority class": DummyClassifier(strategy="prior"),
        "L1 logistic regression": GridSearchCV(
            Pipeline([("filter", SelectKBest(f_classif, k=k)), ("scale", StandardScaler()),
                      ("clf", _l1_logistic(random_state=rs))]),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]},
            cv=StratifiedKFold(3, shuffle=True, random_state=rs), scoring="roc_auc",
        ),
        "Random forest": Pipeline([
            ("filter", SelectKBest(f_classif, k=k)),
            ("clf", RandomForestClassifier(n_estimators=cfg.rf_trees, max_features="sqrt", min_samples_leaf=2,
                                           class_weight="balanced", n_jobs=-1, random_state=rs)),
        ]),
    }
    if genes is not None:
        for name, sig in [("Sweeney 3-gene score", "Sweeney_3"), ("Kaforou 27-gene score", "Kaforou_27")]:
            av = available(sig, genes)
            if av["up"] and av["down"]:
                models[name] = SignatureScore(sig)
    order = list(MODEL_COLORS)
    models = {m: models[m] for m in order if m in models}
    if include:
        models = {m: v for m, v in models.items() if m in include}
    return models


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def xy(ds: Dataset) -> tuple[pd.DataFrame, np.ndarray]:
    d = ds.subset(ds.meta["tb_status"].isin(["ATB", "LTBI"]))
    return d.expr.T, (d.meta["tb_status"] == "ATB").astype(int).values


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """Per-gene z-score within one dataset (genes x samples). Used before cross-platform transfer."""
    sd = expr.std(axis=1).replace(0, 1)
    return expr.sub(expr.mean(axis=1), axis=0).div(sd, axis=0)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def auc_ci(y, p, n_boot=1000, seed=0) -> tuple[float, float, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def threshold_for_sensitivity(y, p, target=0.9) -> float:
    """Highest cut-off that still classifies >= target of the active-TB cases as positive."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def metrics_at(y, p, thr) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp, fn = int(((y == 1) & pred).sum()), int(((y == 1) & ~pred).sum())
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    return {
        "threshold": thr,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "missed_TB": fn, "false_alarms": fp, "n_TB": tp + fn, "n_LTBI": tn + fp,
    }


def evaluate(y, p, thr=None, target=0.9, n_boot=1000, seed=0) -> dict:
    auc, lo, hi = auc_ci(y, p, n_boot, seed)
    thr = threshold_for_sensitivity(y, p, target) if thr is None else thr
    return {"AUC": auc, "AUC_low": lo, "AUC_high": hi, **metrics_at(y, p, thr)}


# --------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------
def cross_validate(models: dict, X: pd.DataFrame, y, strata=None, folds=5, repeats=5, seed=42,
                   verbose=True) -> tuple[pd.DataFrame, dict]:
    """Repeated stratified K-fold. Returns per-fold AUCs and out-of-fold probabilities
    (averaged over repeats) for every model."""
    y = np.asarray(y)
    strata = y if strata is None else pd.Series(strata).astype(str).values + "_" + y.astype(str)
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    splits = list(cv.split(X, strata))
    rows, oof = [], {}
    for name, model in models.items():
        acc, cnt = np.zeros(len(y)), np.zeros(len(y))
        for i, (tr, te) in enumerate(splits):
            m = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            acc[te] += p
            cnt[te] += 1
            rows.append({"model": name, "repeat": i // folds, "fold": i % folds,
                         "AUC": roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan})
        oof[name] = acc / np.maximum(cnt, 1)
        if isinstance(model, DummyClassifier):
            # Each fold predicts its own training prevalence, which differs slightly between folds
            # and would give the pooled out-of-fold predictions a spurious ranking (AUC != 0.5).
            oof[name] = np.full(len(y), y.mean())
        if verbose:
            aucs = [r["AUC"] for r in rows if r["model"] == name]
            print(f"  {name:<24} CV AUC {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    return pd.DataFrame(rows), oof


def summarise_cv(oof: dict, y, target=0.9, n_boot=1000, label=None) -> pd.DataFrame:
    rows = []
    for name, p in oof.items():
        r = evaluate(y, p, target=target, n_boot=n_boot)
        r["model"] = name
        if label:
            r["group"] = label
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def subgroup_report(y, p, meta: pd.DataFrame, by=("hiv_status", "site"), thr=None, target=0.9) -> pd.DataFrame:
    """Metrics for each subgroup at ONE shared threshold - the 'report misses by HIV status' safeguard."""
    thr = threshold_for_sensitivity(y, p, target) if thr is None else thr
    rows = []
    for col in by:
        for lvl in meta[col].dropna().unique():
            m = (meta[col] == lvl).values
            if len(set(np.asarray(y)[m])) < 2:
                continue
            r = evaluate(np.asarray(y)[m], np.asarray(p)[m], thr=thr, n_boot=300)
            rows.append({"by": col, "group": lvl, "n": int(m.sum()), **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Transfer between groups / datasets
# --------------------------------------------------------------------------------------
def transfer(models, X_src, y_src, X_tgt, y_tgt, src_oof: dict | None = None, target=0.9, n_boot=1000,
             seed=42) -> pd.DataFrame:
    """Fit on source, test on target.

    threshold_source : chosen on source out-of-fold predictions, applied unchanged to the target
    threshold_target : re-chosen on half of the target and tested on the other half (2-fold,
                       averaged) - shows how much a group-specific threshold recovers.
    """
    rows = []
    y_tgt = np.asarray(y_tgt)
    for name, model in models.items():
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_src, y_src)
        p = m.predict_proba(X_tgt)[:, 1]
        thr_src = threshold_for_sensitivity(y_src, src_oof[name], target) if src_oof and name in src_oof else 0.5
        r = evaluate(y_tgt, p, thr=thr_src, n_boot=n_boot)
        # group-specific threshold, honest estimate via split halves of the target group
        skf = StratifiedKFold(2, shuffle=True, random_state=seed)
        sens, spec = [], []
        for a, b in skf.split(p.reshape(-1, 1), y_tgt):
            t = threshold_for_sensitivity(y_tgt[a], p[a], target)
            mb = metrics_at(y_tgt[b], p[b], t)
            sens.append(mb["sensitivity"])
            spec.append(mb["specificity"])
        rows.append({"model": name, **r, "sensitivity_group_threshold": np.mean(sens),
                     "specificity_group_threshold": np.mean(spec)})
    return pd.DataFrame(rows).set_index("model")


def fitted_feature_table(ds: Dataset, cfg, top=25) -> pd.DataFrame:
    """Fit L1 logistic and random forest on all samples of `ds`; report the genes they use."""
    X, y = xy(ds)
    models = make_models(cfg, X.shape[1], include=["L1 logistic regression", "Random forest"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l1 = clone(models["L1 logistic regression"]).fit(X, y).best_estimator_
        rf = clone(models["Random forest"]).fit(X, y)
    g1 = X.columns[l1.named_steps["filter"].get_support()]
    coef = pd.Series(l1.named_steps["clf"].coef_[0], index=g1)
    g2 = X.columns[rf.named_steps["filter"].get_support()]
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=g2)
    t = pd.DataFrame({"L1_coefficient": coef[coef != 0], "RF_importance": imp}).fillna(0)
    t["rank_score"] = t["L1_coefficient"].abs().rank(ascending=False) + t["RF_importance"].rank(ascending=False)
    t = t.sort_values("rank_score").drop(columns="rank_score")
    t.attrs["n_L1_nonzero"] = int((coef != 0).sum())
    t.attrs["L1_C"] = float(l1.named_steps["clf"].C)
    return t.head(top)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_roc(oofs: dict, ys: dict, path=None):
    """oofs/ys keyed by group label (e.g. 'HIV-', 'HIV+'), each oof a {model: probs} dict."""
    fig, axes = plt.subplots(1, len(oofs), figsize=(5.2 * len(oofs), 4.8), squeeze=False)
    for ax, (grp, oof) in zip(axes[0], oofs.items()):
        for name, p in oof.items():
            fpr, tpr, _ = roc_curve(ys[grp], p)
            auc = roc_auc_score(ys[grp], p)
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), lw=2 if name != "Majority class" else 1.2,
                    ls="--" if name == "Majority class" else "-", label=f"{name} ({auc:.2f})")
        ax.set(xlabel="1 - specificity", ylabel="sensitivity", title=f"Cross-validated ROC, {grp}",
               xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right", title="model (AUC)", title_fontsize=7.5)
    fig.tight_layout()
    return savefig(fig, path)


def plot_auc_by_group(summary: pd.DataFrame, path=None, title="AUC by HIV status (95% CI)"):
    """summary: rows = (model, group) with AUC, AUC_low, AUC_high; group in HIV-/HIV+."""
    models = [m for m in MODEL_COLORS if m in summary.index.unique()]
    groups = list(dict.fromkeys(summary["group"]))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(models) + 1.4))
    offs = np.linspace(-0.15, 0.15, len(groups))
    for gi, g in enumerate(groups):
        s = summary[summary["group"] == g]
        for mi, m in enumerate(models):
            if m not in s.index:
                continue
            r = s.loc[m]
            y = mi + offs[gi]
            col = PALETTES["hiv_status"].get(g, SLOTS[gi])
            ax.plot([r["AUC_low"], r["AUC_high"]], [y, y], color=col, lw=2)
            ax.scatter(r["AUC"], y, color=col, s=45, zorder=3, edgecolors="white", linewidths=1.5,
                       label=g if mi == 0 else None)
    ax.set_yticks(range(len(models)), models)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set(xlabel="AUC (active TB vs latent TB)", title=title, xlim=(0.4, 1.0))
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return savefig(fig, path)


def plot_transfer(tables: dict, path=None, target=0.9):
    """tables: {label: transfer() output}.

    Left bars: sensitivity and specificity when the threshold tuned on the training group is
    applied unchanged. Right bar: specificity when the threshold is re-tuned on the tested group
    to reach the target sensitivity. A threshold that does not transfer shows up either as
    missed TB (sensitivity below the dashed line) or as lost specificity."""
    labels = list(tables)
    models = [m for m in MODEL_COLORS if m != "Majority class" and m in tables[labels[0]].index]
    bars = [("sensitivity", "sensitivity, training-group threshold", SLOTS[1]),
            ("specificity", "specificity, training-group threshold", SLOTS[0]),
            ("specificity_group_threshold", f"specificity, threshold re-tuned for {target:.0%} sensitivity", SLOTS[2])]
    fig, axes = plt.subplots(1, len(labels), figsize=(5.8 * len(labels), 4.2), squeeze=False, sharey=True)
    w = 0.26
    for ax, lab in zip(axes[0], labels):
        t = tables[lab].loc[models]
        x = np.arange(len(models))
        for i, (col, name, c) in enumerate(bars):
            ax.bar(x + (i - 1) * (w + 0.02), t[col], w, color=c, label=name)
        ax.axhline(target, color=INK_2, lw=0.8, ls="--")
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in models], fontsize=7.5)
        ax.set(ylim=(0, 1.05), title=lab, ylabel="proportion")
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0][0].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return savefig(fig, path)
