"""tbhiv: blood gene-expression signatures for active vs latent TB in HIV- and HIV+ African adults.

Computing Bioinformatics group project: Matthew Nguithi, Ray Lukorito, William Githinji.

Research question
-----------------
Can blood gene expression distinguish active tuberculosis from latent TB infection in
African adults, and does HIV co-infection weaken that signal?

Modules
-------
config      : all tunable settings in one dataclass
data        : GEO download, series-matrix parsing, probe-to-gene mapping, normalisation
qc          : expression distributions, PCA, batch-effect checks, low-expression filter
de          : limma-style linear models with empirical-Bayes moderation, BH FDR, interaction test
enrich      : GO / pathway enrichment (Enrichr via gseapy, or offline GMT)
network     : STRING protein-interaction network + NetworkX hub genes
ml          : classifiers, baselines, repeated CV, HIV- -> HIV+ transfer, external validation
signatures  : published signatures (Sweeney 3-gene, Kaforou 27-gene) and their scores
plots       : shared plotting helpers
report      : collects headline numbers and writes results/summary.md
pipeline    : run every stage end-to-end (used by scripts/run_pipeline.py and the tests)
synthetic   : GEO-format synthetic data for offline testing ONLY (never for results)
"""

__version__ = "1.0.0"

from .config import Config  # noqa: F401
