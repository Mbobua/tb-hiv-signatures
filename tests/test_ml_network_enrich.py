import networkx as nx
import numpy as np
import pandas as pd

from tbhiv import enrich, ml, network
from tbhiv.signatures import signature_score


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(97), np.zeros(83)].astype(int)
    p = np.r_[rng.normal(1, 1, 97), rng.normal(0, 1, 83)]
    thr = ml.threshold_for_sensitivity(y, p, 0.9)
    assert ml.metrics_at(y, p, thr)["sensitivity"] >= 0.9


def test_signature_score_direction():
    expr = pd.DataFrame({"s1": [10, 10, 5], "s2": [6, 6, 9]}, index=["GBP5", "DUSP3", "KLF2"])
    s = signature_score(expr, "Sweeney_3")
    assert s["s1"] == 5 and s["s2"] == -3


def test_no_leakage_on_pure_noise(cfg):
    """With random labels, cross-validated AUC must hover around 0.5. If feature selection
    leaked test information, filtered models would score far higher."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 2000)), columns=[f"g{i}" for i in range(2000)])
    y = rng.integers(0, 2, 120)
    models = ml.make_models(cfg, X.shape[1], include=["L1 logistic regression", "Random forest"])
    folds, oof = ml.cross_validate(models, X, y, folds=5, repeats=1, seed=0, verbose=False)
    for name, p in oof.items():
        assert 0.3 < ml.roc_auc_score(y, p) < 0.7, name


def test_models_learn_planted_signal(main_ds, cfg):
    neg = main_ds.subset(main_ds.meta["hiv_status"] == "HIV-")
    X, y = ml.xy(neg)
    models = ml.make_models(cfg, X.shape[1], genes=X.columns)
    assert {"Sweeney 3-gene score", "Kaforou 27-gene score"} <= set(models)
    _, oof = ml.cross_validate(models, X, y, folds=5, repeats=1, seed=0, verbose=False)
    s = ml.summarise_cv(oof, y, n_boot=100)
    assert abs(s.loc["Majority class", "AUC"] - 0.5) < 0.1
    assert s.loc["L1 logistic regression", "AUC"] > 0.85


def test_ora_finds_planted_pathway(main_ds, synthetic):
    sets = enrich.read_gmt(synthetic["gmt_files"]["toy"])
    hits = [g for g in synthetic["truth"]["interferon"] if g in main_ds.expr.index]
    res = enrich.ora(hits, sets, main_ds.expr.index)
    assert res.iloc[0]["Term"] == "TOY_INTERFERON_RESPONSE"
    assert res.iloc[0]["Adjusted P-value"] < 1e-5


def test_string_edges_parsing_and_hubs():
    tsv = pd.DataFrame({
        "preferredName_A": ["STAT1", "GBP1", "STAT1", "STAT1", "IRF1"],
        "preferredName_B": ["GBP1", "STAT1", "IRF1", "GBP5", "GBP5"],
        "score": [0.99, 0.99, 0.95, 0.9, 0.8],
    })
    edges = network.parse_string_edges(tsv)
    assert len(edges) == 4  # duplicate A-B / B-A removed
    G = nx.Graph()
    G.add_weighted_edges_from(edges.itertuples(index=False))
    hubs = network.hub_table(G)
    assert hubs.index[0] == "STAT1"


def test_coexpression_fallback():
    rng = np.random.default_rng(0)
    f = rng.normal(size=50)
    expr = pd.DataFrame([f + 0.1 * rng.normal(size=50) for _ in range(5)] + [rng.normal(size=50)],
                        index=["A", "B", "C", "D", "E", "Z"])
    e = network.coexpression_edges(expr, list(expr.index), 0.7)
    assert set(e["a"]).union(e["b"]) == {"A", "B", "C", "D", "E"}
