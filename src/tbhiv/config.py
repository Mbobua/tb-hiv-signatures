"""Project settings. Change values here (or pass a modified Config) rather than editing code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- data -------------------------------------------------------------------------
    main_gse: str = "GSE37250"  # Kaforou et al. 2013, Cape Town + Karonga, HIV+ and HIV-
    main_gpl: str = "GPL10558"  # Illumina HumanHT-12 V4
    # GSE19491 (Berry et al. 2010) is a SuperSeries; its TB/LTBI whole-blood samples sit in
    # these three sub-series (London training, South Africa validation, London test).
    validation_gses: tuple = ("GSE19439", "GSE19442", "GSE19444")
    validation_gpl: str = "GPL6947"  # Illumina HumanHT-12 V3
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    # Optional local files (e.g. uploaded to Colab or Google Drive). Keys are GSE / GPL ids.
    local_matrix_files: dict = field(default_factory=dict)
    local_platform_files: dict = field(default_factory=dict)

    # ---- preprocessing ----------------------------------------------------------------
    quantile_normalise: bool = True
    collapse_method: str = "max_mean"  # probe -> gene: keep the probe with highest mean
    low_expression_quantile: float = 0.30  # drop genes whose mean is in the bottom 30%

    # ---- differential expression -------------------------------------------------------
    fdr: float = 0.05
    min_abs_log2fc: float = 0.5  # ~1.4-fold
    de_covariates: tuple = ("site",)  # adjust for Cape Town vs Karonga batch/site effect

    # ---- enrichment / network ----------------------------------------------------------
    enrichr_libraries: tuple = (
        "GO_Biological_Process_2023",
        "KEGG_2021_Human",
        "Reactome_2022",
    )
    gmt_files: dict = field(default_factory=dict)  # optional offline gene sets {name: path.gmt}
    network_top_n: int = 150  # genes sent to STRING
    string_min_score: int = 700  # STRING "high confidence"
    coexpression_min_rho: float = 0.7  # fallback network if STRING is unreachable
    online: bool = True  # False = skip Enrichr/STRING calls (offline testing)

    # ---- machine learning --------------------------------------------------------------
    random_state: int = 42
    cv_folds: int = 5
    cv_repeats: int = 5
    k_features: int = 500  # univariate filter inside each CV fold (no leakage)
    rf_trees: int = 500
    target_sensitivity: float = 0.90  # WHO triage-test target product profile
    bootstrap_iterations: int = 1000

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("", "figures", "tables"):
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def fig_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.results_dir / "tables"
