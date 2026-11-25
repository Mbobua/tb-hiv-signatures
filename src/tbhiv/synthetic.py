"""SYNTHETIC data in GEO series-matrix format - for offline testing of the code ONLY.

Nothing produced from these files is a research result. The generator copies the sample
layout of GSE37250 (counts per site x HIV x TB group) and of the GSE19491 TB sub-series, and
plants known effects so the tests can check the pipeline recovers them:

* an interferon module up in active TB, with a smaller active-vs-latent gap in HIV+
  (HIV raises it in latent TB too) - i.e. the project hypothesis is TRUE in synthetic data;
* a B-cell module down in active TB;
* an HIV-only module, a Malawi-vs-South-Africa batch shift, and two array platforms.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .signatures import INTERFERON_MARKERS, SIGNATURES

# Real GSE37250 layout (from the GEO characteristics): (site, HIV) -> {group: n}
MAIN_LAYOUT = {
    ("Malawi", "HIV negative"): {"active tuberculosis": 51, "latent TB infection": 35, "other disease": 34},
    ("Malawi", "HIV positive"): {"active tuberculosis": 51, "latent TB infection": 36, "other disease": 30},
    ("South Africa", "HIV negative"): {"active tuberculosis": 46, "latent TB infection": 48, "other disease": 49},
    ("South Africa", "HIV positive"): {"active tuberculosis": 47, "latent TB infection": 48, "other disease": 62},
}
VAL_LAYOUT = {
    "GSE19439": ("London", {"PTB": 13, "Latent": 17, "Control (BCG+)": 12}),
    "GSE19442": ("South Africa", {"PTB": 20, "LATENT TB": 31}),
    "GSE19444": ("London", {"PTB": 21, "Latent": 21, "Control (BCG+)": 12}),
}
BCELL = ["CD79A", "CD79B", "CXCR5", "CCR6", "GNG7", "ZNF296", "MS4A1", "BLK", "CD19", "FCRLA", "KLF2"]


def _gene_universe(n_filler: int):
    sig = {g for s in SIGNATURES.values() for g in s["up"] + s["down"]}
    ifn = sorted(set(INTERFERON_MARKERS) | set(SIGNATURES["Kaforou_27"]["up"]) | {"GBP5", "DUSP3"})
    bcell = sorted(set(BCELL) | set(SIGNATURES["Kaforou_27"]["down"]))
    named = sorted(sig | set(ifn) | set(bcell))
    filler = [f"GENE{i:05d}" for i in range(1, n_filler + 1)]
    return named + filler, ifn, bcell, filler


def _simulate(rng, genes, ifn, bcell, filler, meta, platform_shift=0.0):
    G, n = len(genes), len(meta)
    gidx = {g: i for i, g in enumerate(genes)}
    base = rng.uniform(4.5, 13.0, G)
    sd = rng.uniform(0.15, 0.55, G)
    X = base[:, None] + sd[:, None] * rng.standard_normal((G, n))

    ifn_genes = ifn + filler[:120]
    b_genes = bcell + filler[120:180]
    hiv_genes = filler[180:280]

    atb = (meta["tb"] == "ATB").values
    ltbi = (meta["tb"] == "LTBI").values
    od = (meta["tb"] == "OD").values
    hivp = (meta["hiv"] == "HIV+").values
    # module activity per sample: HIV raises interferon in latent TB -> smaller ATB-LTBI gap
    f_ifn = np.where(atb, np.where(hivp, 1.25, 1.0), 0.0) + np.where(ltbi & hivp, 0.55, 0.0) + np.where(od, 0.5, 0)
    f_ifn = f_ifn + 0.35 * rng.standard_normal(n)
    f_b = np.where(atb, np.where(hivp, -0.75, -0.6), 0.0) + np.where(ltbi & hivp, -0.4, 0.0) + np.where(od, -0.3, 0)
    f_b = f_b + 0.25 * rng.standard_normal(n)
    for g in ifn_genes:
        X[gidx[g]] += rng.uniform(0.6, 1.3) * f_ifn
    for g in b_genes:
        X[gidx[g]] += rng.uniform(0.6, 1.2) * f_b
    for g in hiv_genes:
        X[gidx[g]] += 0.6 * hivp
    # site batch effect
    batch = rng.normal(0, 0.25, G) * (rng.random(G) < 0.3)
    X += batch[:, None] * (meta["site"] == "Malawi").values[None, :]
    X += platform_shift + rng.normal(0, 0.1, G)[:, None]
    return X


def _write_matrix(path: Path, gse: str, gpl: str, probes, X, gsm, characteristics: list[list[str]], titles):
    q = lambda v: f'"{v}"'
    lines = [
        f"!Series_title\t{q('SYNTHETIC test data shaped like ' + gse)}",
        f"!Series_geo_accession\t{q(gse)}",
        f"!Series_platform_id\t{q(gpl)}",
        "!Sample_title\t" + "\t".join(q(t) for t in titles),
        "!Sample_geo_accession\t" + "\t".join(q(s) for s in gsm),
        "!Sample_platform_id\t" + "\t".join(q(gpl) for _ in gsm),
    ]
    for row in characteristics:
        lines.append("!Sample_characteristics_ch1\t" + "\t".join(q(v) for v in row))
    lines.append("!series_matrix_table_begin")
    lines.append("\t".join([q("ID_REF")] + [q(s) for s in gsm]))
    for p, vals in zip(probes, X):
        lines.append("\t".join([q(p)] + [f"{v:.4f}" for v in vals]))
    lines.append("!series_matrix_table_end")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _probes(rng, genes, prefix, start, dup_frac=0.15, drop_frac=0.0, n_unannotated=50):
    keep = [g for g in genes if rng.random() >= drop_frac or not g.startswith("GENE")]
    probes, sym = [], []
    i = start
    for g in keep:
        probes.append(f"{prefix}{i}"); sym.append(g); i += 1
        if rng.random() < dup_frac:
            probes.append(f"{prefix}{i}"); sym.append(g); i += 1  # a second, weaker probe
    for _ in range(n_unannotated):
        probes.append(f"{prefix}{i}"); sym.append(""); i += 1
    return probes, sym


def make_synthetic_geo(outdir, seed=0, n_filler=2500) -> dict:
    """Write synthetic GSE37250 + GSE19439/42/44 series matrices, GPL annotation tables and a
    toy GMT. Returns paths ready for Config(local_matrix_files=..., local_platform_files=...)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    genes, ifn, bcell, filler = _gene_universe(n_filler)

    # ---- main (GSE37250-like, linear scale, GPL10558-like) ----
    rows = []
    for (site, hiv), groups in MAIN_LAYOUT.items():
        for grp, n in groups.items():
            rows += [{"site": site, "hiv_raw": hiv, "grp": grp}] * n
    meta = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    meta["tb"] = meta["grp"].map({"active tuberculosis": "ATB", "latent TB infection": "LTBI", "other disease": "OD"})
    meta["hiv"] = np.where(meta["hiv_raw"].str.contains("positive"), "HIV+", "HIV-")
    Xg = _simulate(rng, genes, ifn, bcell, filler, meta)
    probes, sym = _probes(rng, genes, "ILMN_1", 700000)
    gi = {g: i for i, g in enumerate(genes)}
    Xp = np.vstack([Xg[gi[s]] - (0.8 if k > 0 and sym[k - 1] == s else 0.0) if s else
                    rng.uniform(4, 6, len(meta)) for k, s in enumerate(sym)])
    linear = 2 ** Xp  # deposit on linear scale to exercise the log2 detection
    gsm = [f"GSM9{100000 + i}" for i in range(len(meta))]
    chars = [[f"disease state: {g}" for g in meta["grp"]], [f"hiv status: {h}" for h in meta["hiv_raw"]],
             [f"geographical region: {s}" for s in meta["site"]], ["tissue: whole blood"] * len(meta)]
    p_main = out / "GSE37250_series_matrix.txt.gz"
    _write_matrix(p_main, "GSE37250", "GPL10558", probes, linear, gsm, chars, [f"synthetic_{i}" for i in range(len(meta))])
    gpl_main = out / "GPL10558_annotation.txt"
    with open(gpl_main, "w") as f:
        f.write("#ID = probe id\n#Symbol = gene symbol\n")
        pd.DataFrame({"ID": probes, "Symbol": sym}).to_csv(f, sep="\t", index=False)

    # ---- validation (GSE19491 sub-series-like, log2 scale, GPL6947-like) ----
    vprobes, vsym = _probes(rng, genes, "ILMN_2", 100000, drop_frac=0.1)
    paths = {"GSE37250": str(p_main)}
    k = 0
    for gse, (site, groups) in VAL_LAYOUT.items():
        rows = [{"illness": g, "site": site} for g, n in groups.items() for _ in range(n)]
        vm = pd.DataFrame(rows)
        vm["tb"] = vm["illness"].map(lambda s: "ATB" if s == "PTB" else ("LTBI" if "atent" in s.lower() else "Control"))
        vm["hiv"] = "HIV-"
        Xv = _simulate(rng, genes, ifn, bcell, filler, vm, platform_shift=0.4)
        Xvp = np.vstack([Xv[gi[s]] if s else rng.uniform(4, 6, len(vm)) for s in vsym])
        vgsm = [f"GSM4{80000 + k + i}" for i in range(len(vm))]
        k += len(vm)
        vchars = [[f"illness: {g}" for g in vm["illness"]], [f"geographical region: {site}"] * len(vm)]
        p = out / f"{gse}_series_matrix.txt.gz"
        _write_matrix(p, gse, "GPL6947", vprobes, Xvp, vgsm, vchars, [f"{gse}_synthetic_{i}" for i in range(len(vm))])
        paths[gse] = str(p)
    gpl_val = out / "GPL6947_annotation.txt"
    pd.DataFrame({"ID": vprobes, "ILMN_Gene": vsym}).to_csv(gpl_val, sep="\t", index=False)

    gmt = out / "toy_gene_sets.gmt"
    sets = {"TOY_INTERFERON_RESPONSE": ifn + filler[:120], "TOY_B_CELL": bcell + filler[120:180],
            "TOY_HIV_MODULE": filler[180:280], "TOY_RANDOM": list(rng.choice(filler[300:], 150, replace=False))}
    gmt.write_text("\n".join("\t".join([k, "synthetic"] + v) for k, v in sets.items()) + "\n")
    return {
        "local_matrix_files": paths,
        "local_platform_files": {"GPL10558": str(gpl_main), "GPL6947": str(gpl_val)},
        "gmt_files": {"toy": str(gmt)},
        "truth": {"interferon": ifn, "bcell": bcell, "hiv_module": filler[180:280]},
    }
