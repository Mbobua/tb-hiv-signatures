"""End-to-end smoke test: every stage runs on synthetic GEO files and writes the report."""

from tbhiv.pipeline import run_all


def test_run_all(cfg):
    out = run_all(cfg, data_note="SYNTHETIC TEST DATA")
    text = out["report_path"].read_text()
    for heading in ["Sample selection", "Quality control", "Differential expression", "Does HIV weaken",
                    "Enrichment", "Network", "Machine learning", "Transfer", "External validation"]:
        assert heading in text, heading
    n = out["report"].numbers
    assert n["n_samples_atb_ltbi"] == 362
    assert n["attenuation_ratio"] < 1
    assert n["external_auc_L1 logistic regression"] > 0.8
    figs = list((cfg.results_dir / "figures").glob("*.png"))
    assert len(figs) >= 12
