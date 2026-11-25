import numpy as np
import pandas as pd
import pytest

from tbhiv import data
from tbhiv.synthetic import MAIN_LAYOUT


@pytest.mark.parametrize("raw,expected", [
    ("active tuberculosis", "ATB"),       # GSE37250
    ("latent TB infection", "LTBI"),      # GSE37250
    ("other disease", "OD"),              # GSE37250
    ("PTB", "ATB"),                       # GSE19439/42/44
    ("Latent", "LTBI"),                   # GSE19439/44
    ("LATENT TB", "LTBI"),                # GSE19442
    ("Control (BCG+)", "Control"),
    ("whole blood", None),
])
def test_tb_status_mapping(raw, expected):
    assert data._tb_status(raw) == expected


@pytest.mark.parametrize("raw,expected", [("HIV negative", "HIV-"), ("HIV positive", "HIV+"),
                                          ("HIV+", "HIV+"), ("negative", "HIV-")])
def test_hiv_mapping(raw, expected):
    assert data._hiv_status(raw) == expected


def test_series_matrix_and_labels(synthetic):
    expr, info, series = data.read_series_matrix(synthetic["local_matrix_files"]["GSE37250"])
    assert expr.shape[1] == 537
    assert series["geo_accession"] == "GSE37250"
    meta = data.annotate_samples(info, "GSE37250")
    for (site, hiv), groups in MAIN_LAYOUT.items():
        h = "HIV+" if "positive" in hiv else "HIV-"
        sub = meta[(meta["site"] == site) & (meta["hiv_status"] == h)]
        assert (sub["tb_status"] == "ATB").sum() == groups["active tuberculosis"]
        assert (sub["tb_status"] == "LTBI").sum() == groups["latent TB infection"]
        assert (sub["tb_status"] == "OD").sum() == groups["other disease"]


def test_validation_defaults(cfg):
    val, logs = data.load_validation(cfg, verbose=False)
    assert set(val.meta["dataset"]) == {"GSE19439", "GSE19442", "GSE19444"}
    assert (val.meta["hiv_status"] == "HIV-").all()
    assert set(val.meta["site"]) == {"London", "South Africa"}
    counts = val.meta["tb_status"].value_counts()
    assert counts["ATB"] == 54 and counts["LTBI"] == 69


def test_log_detection_and_quantile_normalisation():
    rng = np.random.default_rng(0)
    lin = pd.DataFrame(2 ** rng.uniform(5, 14, (500, 6)))
    assert not data.is_log_scale(lin)
    lg = data.to_log2(lin)
    assert data.is_log_scale(lg)
    qn = data.quantile_normalise(lg)
    s = np.sort(qn.values, axis=0)
    assert np.allclose(s, s[:, [0]])  # every column has the same distribution


def test_collapse_keeps_highest_mean_probe():
    expr = pd.DataFrame({"a": [5.0, 9.0, 7.0], "b": [5.0, 9.0, 7.0]}, index=["p1", "p2", "p3"])
    p2g = pd.Series({"p1": "G1", "p2": "G1", "p3": "G2"})
    out = data.collapse_probes(expr, p2g, "max_mean")
    assert out.loc["G1", "a"] == 9.0 and out.loc["G2", "a"] == 7.0
    assert data.collapse_probes(expr, p2g, "mean").loc["G1", "a"] == 7.0


def test_keep_atb_ltbi_records_counts(cfg):
    ds, _ = data.load_geo_dataset("GSE37250", "GPL10558", cfg, verbose=False)
    sub, sel = data.keep_atb_ltbi(ds)
    assert sel["before"]["OD"] == 175
    assert set(sub.meta["tb_status"]) == {"ATB", "LTBI"} and sub.shape[1] == 362
