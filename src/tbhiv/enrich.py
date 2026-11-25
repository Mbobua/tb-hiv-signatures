"""Stage 5 - biological interpretation by GO / pathway enrichment (Objective 2).

Online : Enrichr through gseapy (GO Biological Process, KEGG, Reactome), with the set of
         tested genes as the background so that array content does not bias results.
Offline: over-representation analysis against any local GMT file (hypergeometric + BH).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .plots import INK_2, SLOTS, savefig

COLUMNS = ["library", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]


def run_enrichr(genes, libraries, background=None, outdir=None) -> pd.DataFrame:
    """Query Enrichr. Returns an empty frame (with a 'note' attribute) if offline."""
    genes = sorted(set(genes))
    if len(genes) < 5:
        out = pd.DataFrame(columns=COLUMNS)
        out.attrs["note"] = f"only {len(genes)} genes - enrichment skipped"
        return out
    try:
        import gseapy as gp

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=list(libraries),
            organism="human",
            background=sorted(set(background)) if background is not None else None,
            outdir=str(outdir) if outdir else None,
            no_plot=True,
        )
        res = enr.results.rename(columns={"Gene_set": "library"})
        if "Overlap" not in res.columns:
            # With a custom background, gseapy (>=1.1) queries Enrichr's background endpoint,
            # whose results lack 'Overlap'. Rebuild it as hits / term size within the background.
            res["Overlap"] = _overlap_from_libraries(res, libraries, background)
        return res[[c for c in COLUMNS if c in res.columns]].sort_values("Adjusted P-value")
    except Exception as e:  # network / API errors
        out = pd.DataFrame(columns=COLUMNS)
        out.attrs["note"] = f"Enrichr unavailable ({type(e).__name__}: {e})"
        return out


@lru_cache(maxsize=None)
def _enrichr_library(name: str) -> dict:
    import gseapy as gp

    return gp.get_library(name=name, organism="Human")


def _overlap_from_libraries(res: pd.DataFrame, libraries, background=None) -> list[str]:
    """'k/m' strings: k = genes hit, m = term size (restricted to the background if given).
    Falls back to 'k' alone if the gene-set libraries cannot be downloaded."""
    hits = res["Genes"].fillna("").astype(str).map(lambda s: len([g for g in s.split(";") if g]))
    try:
        bg = set(background) if background is not None else None
        sizes = {}
        for lib in libraries:
            for term, members in _enrichr_library(lib).items():
                sizes[(lib, term)] = len(set(members) & bg) if bg is not None else len(members)
        return [f"{k}/{sizes[(lib, t)]}" if (lib, t) in sizes else str(k)
                for k, lib, t in zip(hits, res["library"], res["Term"])]
    except Exception:
        return [str(k) for k in hits]


def read_gmt(path) -> dict[str, list[str]]:
    sets = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def ora(genes, gene_sets: dict, universe, min_size=5, max_size=500, library="local") -> pd.DataFrame:
    """Hypergeometric over-representation test with BH correction."""
    universe = set(universe)
    hits = set(genes) & universe
    N, n = len(universe), len(hits)
    rows = []
    for term, members in gene_sets.items():
        m = set(members) & universe
        if not (min_size <= len(m) <= max_size):
            continue
        k = len(hits & m)
        if k == 0:
            continue
        p = stats.hypergeom.sf(k - 1, N, len(m), n)
        rows.append({"library": library, "Term": term, "Overlap": f"{k}/{len(m)}", "P-value": p,
                     "Genes": ";".join(sorted(hits & m))})
    out = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "Adjusted P-value"])
    if len(out):
        out["Adjusted P-value"] = multipletests(out["P-value"], method="fdr_bh")[1]
        out = out.sort_values("P-value")
    else:
        out["Adjusted P-value"] = []
    return out[COLUMNS]


def plot_enrichment(res: pd.DataFrame, title: str, top=12, path=None):
    fig, ax = plt.subplots(figsize=(8, 0.35 * max(top, 4) + 1))
    if res is None or res.empty:
        ax.text(0.5, 0.5, getattr(res, "attrs", {}).get("note", "no enriched terms"), ha="center", va="center",
                color=INK_2)
        ax.axis("off")
        ax.set_title(title)
        return savefig(fig, path)
    r = res.head(top).iloc[::-1]
    vals = -np.log10(r["Adjusted P-value"].astype(float).clip(lower=1e-300))
    labels = [f"{t[:60]}{'...' if len(t) > 60 else ''}" for t in r["Term"]]
    ax.barh(labels, vals, color=SLOTS[1], height=0.65)
    for y, (v, ov) in enumerate(zip(vals, r["Overlap"])):
        ax.text(v, y, f"  {ov}", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(-np.log10(0.05), color=INK_2, lw=0.8, ls="--")
    ax.set(xlabel="-log10 adjusted p (dashed line = 0.05)", title=title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return savefig(fig, path)
