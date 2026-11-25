"""Published TB blood signatures used as benchmarks and ML baselines.

Gene lists (with up/down direction) are taken from the TBSignatureProfiler R package
(`TBsignaturesSplit`, Johnson lab, Boston University), which curates them from the papers:

* Sweeney_OD_3  - Sweeney et al. 2016, Lancet Respir Med. Score = mean(GBP5, DUSP3) - KLF2
  (on log2 data this equals the published geometric-mean formula).
* Kaforou_27    - Kaforou et al. 2013, PLoS Med (the GSE37250 paper). TB vs LTBI.
  Disease risk score (DRS) = sum(up) - sum(down); we use mean(up) - mean(down) so that
  missing genes on another platform do not change the scale.

Several Kaforou transcripts have had their symbols renamed since 2013, so aliases are
listed together; whichever symbol the platform annotation uses will match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNATURES = {
    "Sweeney_3": {
        "up": ["GBP5", "DUSP3"],
        "down": ["KLF2"],
    },
    "Kaforou_27": {
        "up": [
            "GAS6", "ANKRD22", "LHFPL2", "FCGR1A", "C5", "C1QC", "FLVCR2", "VAMP5",
            "FAM198B", "GASK1B",  # same gene, old/new symbol
            "FCGR1B", "FAM20A", "MPO", "GBP6", "LOC728744",
            "FCGR1C", "FCGR1CP",  # same gene, old/new symbol
            "C1QB", "SMARCD3", "S100A8", "DUSP3",
        ],
        "down": ["GNG7", "CD79A", "ZNF296", "CXCR5", "CCR6", "CD79B"],
    },
}

# Interferon-driven genes repeatedly reported in TB blood signatures (Berry 2010, Kaforou 2013).
# Used only to annotate plots, never to select features.
INTERFERON_MARKERS = [
    "STAT1", "GBP1", "GBP2", "GBP5", "GBP6", "BATF2", "ANKRD22", "FCGR1A", "FCGR1B",
    "SERPING1", "IFITM3", "IFIT3", "IFI44L", "EPSTI1", "SEPT4", "AIM2", "WARS", "WARS1",
]


def signature_genes(name: str) -> list[str]:
    s = SIGNATURES[name]
    return list(dict.fromkeys(s["up"] + s["down"]))


def available(name: str, genes) -> dict:
    """Which signature genes are present in an expression matrix index."""
    genes = set(genes)
    s = SIGNATURES[name]
    return {"up": [g for g in s["up"] if g in genes], "down": [g for g in s["down"] if g in genes]}


def signature_score(expr: pd.DataFrame, name: str, min_genes: int = 1) -> pd.Series:
    """Score each sample: mean(up genes) - mean(down genes).

    expr : genes x samples (log2 scale, or z-scored per gene).
    """
    av = available(name, expr.index)
    if len(av["up"]) + len(av["down"]) < min_genes:
        raise ValueError(f"Only {av} of signature {name} found in data")
    up = expr.loc[av["up"]].mean(axis=0) if av["up"] else 0.0
    dn = expr.loc[av["down"]].mean(axis=0) if av["down"] else 0.0
    score = up - dn
    if np.isscalar(score):  # neither direction present (guarded above)
        score = pd.Series(score, index=expr.columns)
    return score.rename(name)
