import numpy as np
import pandas as pd
from scipy import special, stats

from tbhiv import de
from tbhiv.data import Dataset


def test_trigamma_inverse_roundtrip():
    for y in [0.3, 1.0, 4.0, 25.0]:
        assert np.isclose(de.trigamma_inverse(special.polygamma(1, y)), y, rtol=1e-6)


def test_fit_f_dist_recovers_prior():
    rng = np.random.default_rng(0)
    d0, s0_2, df = 8.0, 0.2, 20
    true_var = s0_2 * d0 / rng.chisquare(d0, 20000)          # scaled inverse-chi2 prior
    s2 = true_var * rng.chisquare(df, 20000) / df            # sample variances
    d0_hat, s0_hat = de.fit_f_dist(s2, df)
    assert abs(d0_hat - d0) / d0 < 0.15
    assert abs(s0_hat - s0_2) / s0_2 < 0.1


def test_unmoderated_matches_students_t():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(50, 30))
    Y[:5, :15] += 1.5
    group = np.r_[np.ones(15), np.zeros(15)]
    X = np.c_[np.ones(30), group]
    coef, s2, df, su = de.lm_fit(Y, X)
    t, p, _ = de.ebayes(coef, s2, df, su, moderate=False)
    ref = stats.ttest_ind(Y[:, :15], Y[:, 15:], axis=1)
    assert np.allclose(t[:, 1], ref.statistic) and np.allclose(p[:, 1], ref.pvalue)


def test_design_adjusts_for_site_confounding():
    """A site shift that is confounded with TB status must not appear as a TB effect."""
    rng = np.random.default_rng(2)
    n = 200
    site = np.where(rng.random(n) < 0.5, "A", "B")
    tb = np.where(rng.random(n) < np.where(site == "A", 0.8, 0.2), "ATB", "LTBI")
    Y = rng.normal(size=(300, n)) + 2.0 * (site == "A")      # pure batch, no TB effect
    meta = pd.DataFrame({"tb_status": tb, "site": site, "hiv_status": "HIV-"}, index=[f"s{i}" for i in range(n)])
    ds = Dataset("x", pd.DataFrame(Y, columns=meta.index, index=[f"g{i}" for i in range(300)]), meta)
    naive, _ = de.differential_expression(ds, covariates=())
    adj, _ = de.differential_expression(ds, covariates=("site",))
    assert naive["significant"].mean() > 0.5
    assert adj["significant"].sum() == 0


def test_de_recovers_planted_effects_and_attenuation(main_ds, synthetic):
    neg = main_ds.subset(main_ds.meta["hiv_status"] == "HIV-")
    pos = main_ds.subset(main_ds.meta["hiv_status"] == "HIV+")
    rn, _ = de.differential_expression(neg)
    rp, _ = de.differential_expression(pos)
    ifn = set(synthetic["truth"]["interferon"]) & set(rn.index)
    up = set(rn.index[rn["direction"] == "up"])
    assert len(ifn & up) / len(ifn) > 0.6               # most planted up-genes found
    hiv_module = set(synthetic["truth"]["hiv_module"]) & set(rn.index)
    assert len(hiv_module & set(rn.index[rn["significant"]])) == 0   # HIV-only genes are not TB genes
    assert rn["significant"].sum() > rp["significant"].sum()          # weaker signal in HIV+
    cmp = de.compare_gene_lists(rn, rp)
    assert cmp["hypergeom_p"] < 1e-10
    pooled, _ = de.differential_expression(main_ds, covariates=("hiv_status", "site"))
    att = de.attenuation_ratio(main_ds, pooled.index[pooled["significant"]], n_boot=50)
    assert att["ci_high"] < 1.0                                       # planted attenuation detected
    inter, _ = de.interaction_model(main_ds)
    assert inter["attenuated"].sum() > inter["amplified"].sum()
