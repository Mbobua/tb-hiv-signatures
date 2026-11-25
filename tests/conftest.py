import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbhiv import Config  # noqa: E402
from tbhiv.synthetic import make_synthetic_geo  # noqa: E402


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    return make_synthetic_geo(tmp_path_factory.mktemp("geo"), seed=1, n_filler=1500)


@pytest.fixture(scope="session")
def cfg(synthetic, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    return Config(
        data_dir=root / "data", results_dir=root / "results",
        local_matrix_files=synthetic["local_matrix_files"],
        local_platform_files=synthetic["local_platform_files"],
        gmt_files=synthetic["gmt_files"], online=False,
        cv_repeats=1, rf_trees=100, bootstrap_iterations=200, k_features=300,
    )


@pytest.fixture(scope="session")
def main_ds(cfg):
    from tbhiv import data, qc

    ds, _ = data.load_geo_dataset("GSE37250", "GPL10558", cfg, verbose=False)
    ds, _ = data.keep_atb_ltbi(ds)
    ds, _ = qc.filter_low_expression(ds, cfg.low_expression_quantile)
    return ds
