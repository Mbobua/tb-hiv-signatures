"""Stage 1 - load data and sample labels.

Downloads GEO series-matrix files, parses sample characteristics into tidy labels
(TB status, HIV status, site), maps Illumina probes to gene symbols and normalises.

Everything here also works on local files (pass them through Config.local_matrix_files /
Config.local_platform_files), so the notebook still runs if NCBI is slow or blocked.
"""

from __future__ import annotations

import gzip
import io
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# Sub-series that do not record the site in their characteristics.
DEFAULT_SITES = {"GSE19439": "London", "GSE19444": "London", "GSE19442": "South Africa"}
# GSE19491 enrolled HIV-negative participants only.
DEFAULT_HIV = {"GSE19439": "HIV-", "GSE19444": "HIV-", "GSE19442": "HIV-"}


# --------------------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------------------
@dataclass
class Dataset:
    """Expression (genes x samples, log2) plus tidy sample metadata (samples x columns)."""

    name: str
    expr: pd.DataFrame
    meta: pd.DataFrame

    def __post_init__(self):
        common = [s for s in self.meta.index if s in self.expr.columns]
        self.meta = self.meta.loc[common]
        self.expr = self.expr[common]

    def subset(self, mask) -> "Dataset":
        mask = pd.Series(mask, index=self.meta.index) if not isinstance(mask, pd.Series) else mask
        keep = self.meta.index[mask.reindex(self.meta.index).fillna(False).astype(bool)]
        return Dataset(self.name, self.expr[keep], self.meta.loc[keep])

    @property
    def shape(self):
        return self.expr.shape

    def __repr__(self):
        return f"Dataset({self.name}: {self.expr.shape[0]} genes x {self.expr.shape[1]} samples)"


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def series_matrix_url(gse: str) -> str:
    stub = gse[:-3] + "nnn"  # GSE37250 -> GSE37nnn
    return f"{GEO_FTP}/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception as e:  # network errors vary by environment
            last = e
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not download {url} ({last}). Download it manually from GEO and pass it via "
        f"Config.local_matrix_files."
    )


# --------------------------------------------------------------------------------------
# Series-matrix parsing
# --------------------------------------------------------------------------------------
def _open_text(path: Path):
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _split_values(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")
    return [p.strip().strip('"') for p in parts]


def read_series_matrix(path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse a GEO *_series_matrix.txt(.gz).

    Returns
    -------
    expr        : probes x samples (raw values as deposited)
    sample_info : samples x fields (title, characteristics as 'key' columns, ...)
    series_info : dict of !Series_ fields
    """
    series_info: dict = {}
    sample_fields: dict[str, list[list[str]]] = {}
    table_lines: list[str] = []
    in_table = False
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Series_"):
                vals = _split_values(line)
                key = vals[0][len("!Series_"):]
                series_info.setdefault(key, []).append(" ".join(vals[1:]))
            elif line.startswith("!Sample_"):
                vals = _split_values(line)
                key = vals[0][len("!Sample_"):]
                sample_fields.setdefault(key, []).append(vals[1:])

    if not table_lines:
        raise ValueError(f"{path}: no expression table found")
    expr = pd.read_csv(
        io.StringIO("".join(table_lines)), sep="\t", index_col=0,
        na_values=["null", "NULL", "NA", "", "NaN"], low_memory=False,
    )
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [str(c).strip('"') for c in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce")

    gsm = sample_fields.get("geo_accession", [list(expr.columns)])[0]
    info = pd.DataFrame(index=pd.Index(gsm, name="sample"))
    for key, rows in sample_fields.items():
        if key == "characteristics_ch1":
            for j, row in enumerate(rows):
                for sample, val in zip(gsm, row):
                    if ": " in val:
                        k, v = val.split(": ", 1)
                    else:
                        k, v = f"characteristic_{j}", val
                    info.loc[sample, "ch:" + k.strip().lower()] = v.strip()
        elif key in ("title", "source_name_ch1", "description", "platform_id"):
            info[key] = rows[0][: len(gsm)]
    series_info = {k: " | ".join(v) for k, v in series_info.items()}
    return expr, info, series_info


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def _tb_status(value: str) -> str | None:
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    if "other disease" in s or s == "od":
        return "OD"
    if "latent" in s or s == "ltbi":
        return "LTBI"
    if ("active" in s and ("tb" in s or "tubercul" in s)) or s in ("ptb", "tb", "active tb"):
        return "ATB"
    if s.startswith("control") or "healthy" in s:
        return "Control"
    return None


def _hiv_status(value: str) -> str | None:
    s = str(value).strip().lower().replace(" ", "")
    if any(t in s for t in ("positive", "hiv+", "pos")):
        return "HIV+"
    if any(t in s for t in ("negative", "hiv-", "neg")):
        return "HIV-"
    return None


def _site(value: str) -> str | None:
    s = str(value).strip().lower()
    if "malawi" in s or "karonga" in s:
        return "Malawi"
    if "south africa" in s or "cape town" in s:
        return "South Africa"
    if "london" in s or "uk" == s or "united kingdom" in s:
        return "London"
    return str(value).strip() if s and s != "nan" else None


_TB_KEYS = re.compile(r"disease|illness|diagnos|tb status|group|class|status")


def annotate_samples(info: pd.DataFrame, gse: str, default_site=None, default_hiv=None) -> pd.DataFrame:
    """Turn raw GEO characteristics into tidy columns: tb_status, hiv_status, site, dataset."""
    ch_cols = [c for c in info.columns if c.startswith("ch:")]
    tb_keys = [c for c in ch_cols if _TB_KEYS.search(c) and "hiv" not in c]
    other = [c for c in ch_cols if c not in tb_keys and "hiv" not in c and "exposure" not in c]

    def first_match(row, cols, fn):
        for c in cols:
            if c in row and pd.notna(row[c]):
                out = fn(row[c])
                if out is not None:
                    return out
        return None

    meta = pd.DataFrame(index=info.index)
    meta["dataset"] = gse
    meta["tb_status"] = [first_match(r, tb_keys + other, _tb_status) for _, r in info.iterrows()]
    hiv_cols = [c for c in ch_cols if "hiv" in c]
    meta["hiv_status"] = [
        first_match(r, hiv_cols, _hiv_status) or default_hiv for _, r in info.iterrows()
    ]
    site_cols = [c for c in ch_cols if re.search(r"region|country|site|location|geograph", c)]
    meta["site"] = [first_match(r, site_cols, _site) or default_site for _, r in info.iterrows()]
    if "title" in info:
        meta["title"] = info["title"]
    return meta


# --------------------------------------------------------------------------------------
# Platform annotation and probe -> gene
# --------------------------------------------------------------------------------------
_SYMBOL_COLS = ["Symbol", "ILMN_Gene", "Gene symbol", "GENE_SYMBOL", "Gene Symbol", "symbol"]


def _pick_symbol_table(table: pd.DataFrame) -> pd.Series:
    id_col = "ID" if "ID" in table.columns else table.columns[0]
    sym_col = next((c for c in _SYMBOL_COLS if c in table.columns), None)
    if sym_col is None:
        raise ValueError(f"No gene-symbol column among {list(table.columns)[:20]}")
    s = table[[id_col, sym_col]].dropna()
    s[sym_col] = s[sym_col].astype(str).str.split(r"\s*///\s*").str[0].str.strip()
    s = s[(s[sym_col] != "") & (s[sym_col].str.lower() != "nan")]
    return pd.Series(s[sym_col].values, index=s[id_col].astype(str).values, name="symbol")


def load_platform_annotation(gpl: str, cfg: Config) -> pd.Series:
    """probe id -> gene symbol. Uses a local file, a cached CSV, or GEOparse."""
    cache = cfg.data_dir / f"{gpl}_symbols.csv"
    if gpl in cfg.local_platform_files:
        path = Path(cfg.local_platform_files[gpl])
        sep = "," if path.suffix == ".csv" else "\t"
        table = pd.read_csv(path, sep=sep, comment="#", low_memory=False, dtype=str)
        return _pick_symbol_table(table)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, dtype=str)["symbol"]
    import GEOparse  # imported lazily: only needed online

    gpl_obj = GEOparse.get_GEO(geo=gpl, destdir=str(cfg.data_dir), silent=True)
    sym = _pick_symbol_table(gpl_obj.table.astype({"ID": str}))
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    sym.to_frame().to_csv(cache)
    return sym


def collapse_probes(expr: pd.DataFrame, probe2gene: pd.Series, method: str = "max_mean") -> pd.DataFrame:
    """Probes x samples -> genes x samples.

    max_mean : keep the single probe with the highest average expression for each gene
               (the usual choice for Illumina arrays; avoids averaging dead probes in).
    mean     : average all probes of a gene.
    """
    p2g = probe2gene.reindex(expr.index).dropna()
    e = expr.loc[p2g.index]
    if method == "mean":
        out = e.groupby(p2g.values).mean()
    elif method == "max_mean":
        means = e.mean(axis=1)
        best = means.groupby(p2g.values).idxmax()
        out = e.loc[best.values]
        out.index = best.index
    else:
        raise ValueError(method)
    out.index.name = "gene"
    return out.sort_index()


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def is_log_scale(expr: pd.DataFrame) -> bool:
    return float(np.nanpercentile(expr.values, 99)) < 50


def to_log2(expr: pd.DataFrame, offset: float = 16.0) -> pd.DataFrame:
    """log2(max(x,0) + offset). The offset (as in limma::neqc) stabilises noisy low probes."""
    if is_log_scale(expr):
        return expr
    return np.log2(expr.clip(lower=0) + offset)


def quantile_normalise(expr: pd.DataFrame) -> pd.DataFrame:
    """Make every sample's distribution identical (Bolstad et al. 2003). NaN-free input."""
    arr = expr.values
    mean_sorted = np.sort(arr, axis=0).mean(axis=1)
    ranks = expr.rank(axis=0, method="average").values - 1
    out = np.interp(ranks, np.arange(len(mean_sorted)), mean_sorted)
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


# --------------------------------------------------------------------------------------
# One-call loaders
# --------------------------------------------------------------------------------------
def load_geo_dataset(gse: str, gpl: str, cfg: Config, verbose: bool = True) -> tuple[Dataset, dict]:
    """Download (or read) one series, label samples, map probes to genes, normalise.

    Returns the Dataset plus a dict of bookkeeping counts for the report.
    """
    path = cfg.local_matrix_files.get(gse)
    if path is None:
        path = download(series_matrix_url(gse), cfg.data_dir / f"{gse}_series_matrix.txt.gz")
    raw, info, series = read_series_matrix(path)
    meta = annotate_samples(info, gse, DEFAULT_SITES.get(gse), DEFAULT_HIV.get(gse))

    log_in = is_log_scale(raw)
    e = to_log2(raw)
    n_probes = e.shape[0]
    e = e.dropna(axis=0, how="any")  # probes missing in any sample
    if cfg.quantile_normalise:
        e = quantile_normalise(e)
    p2g = load_platform_annotation(gpl, cfg)
    genes = collapse_probes(e, p2g, cfg.collapse_method)
    ds = Dataset(gse, genes, meta)
    log = {
        "dataset": gse,
        "platform": gpl,
        "samples": ds.shape[1],
        "probes_raw": n_probes,
        "probes_complete": e.shape[0],
        "genes": ds.shape[0],
        "input_was_log2": log_in,
        "series_title": series.get("title", ""),
    }
    if verbose:
        print(f"{gse}: {log['samples']} samples, {n_probes} probes -> {log['genes']} genes "
              f"(input {'log2' if log_in else 'linear -> log2'})")
    return ds, log


def load_validation(cfg: Config, verbose: bool = True) -> tuple[Dataset, list[dict]]:
    """Load the GSE19491 TB sub-series and stack them on their shared genes."""
    parts, logs = [], []
    for gse in cfg.validation_gses:
        ds, log = load_geo_dataset(gse, cfg.validation_gpl, cfg, verbose)
        parts.append(ds)
        logs.append(log)
    genes = sorted(set.intersection(*[set(p.expr.index) for p in parts]))
    expr = pd.concat([p.expr.loc[genes] for p in parts], axis=1)
    meta = pd.concat([p.meta for p in parts])
    return Dataset("GSE19491", expr, meta), logs


def label_counts(meta: pd.DataFrame, by=("site", "hiv_status")) -> pd.DataFrame:
    return pd.crosstab([meta[c] for c in by], meta["tb_status"], margins=True)


def keep_atb_ltbi(ds: Dataset) -> tuple[Dataset, dict]:
    """Keep active TB and latent TB only; record before/after counts (as the plan requires)."""
    before = ds.meta["tb_status"].value_counts(dropna=False).to_dict()
    out = ds.subset(ds.meta["tb_status"].isin(["ATB", "LTBI"]))
    after = out.meta["tb_status"].value_counts().to_dict()
    return out, {"before": before, "after": after}
