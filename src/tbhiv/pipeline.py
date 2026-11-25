"""All analysis stages as functions. The notebook calls them one at a time (with narrative in
between); scripts/run_pipeline.py and the tests call run_all()."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import data, de, enrich, ml, network, qc
from .config import Config
from .report import Report
from .signatures import SIGNATURES, signature_genes


def _save(df: pd.DataFrame, cfg: Config, name: str, index=True):
    path = cfg.table_dir / name
    df.to_csv(path, index=index)
    return path


# --------------------------------------------------------------------------------------
# Stage 1: load
# --------------------------------------------------------------------------------------
def stage_load(cfg: Config, rep: Report, load_validation=True) -> dict:
    cfg.ensure_dirs()
    main, main_log = data.load_geo_dataset(cfg.main_gse, cfg.main_gpl, cfg)
    val, val_logs = None, []
    if load_validation:
        try:
            val, val_logs = data.load_validation(cfg)
        except Exception as e:
            print(f"Validation data not loaded ({e}). External validation will be skipped.")
    counts = data.label_counts(main.meta)
    logs = pd.DataFrame([main_log, *val_logs]).set_index("dataset")
    _save(logs, cfg, "00_data_sources.csv")
    _save(counts, cfg, "00_sample_counts_all.csv")
    rep.add("Data", f"Main dataset {cfg.main_gse}: {main.shape[1]} samples, {main.shape[0]} genes after "
            "probe-to-gene mapping.", table=logs.drop(columns=["series_title"], errors="ignore"))
    rep.number("n_samples_main_all", main.shape[1])
    return {"main": main, "val": val, "logs": logs, "counts": counts}


def stage_select(main: data.Dataset, cfg: Config, rep: Report) -> dict:
    ds, sel = data.keep_atb_ltbi(main)
    missing = ds.meta[["hiv_status", "site"]].isna().any(axis=1)
    if missing.any():
        print(f"Dropping {missing.sum()} samples with missing HIV status or site")
        ds = ds.subset(~missing)
    counts = data.label_counts(ds.meta)
    _save(counts, cfg, "01_sample_counts_atb_ltbi.csv")
    before = ", ".join(f"{k}: {v}" for k, v in sel["before"].items())
    rep.add("Sample selection", f"Before: {before}. Kept active TB (ATB) and latent TB (LTBI) only: "
            f"{ds.shape[1]} samples.", table=counts)
    rep.number("n_samples_atb_ltbi", ds.shape[1])
    return {"ds": ds, "counts": counts, "selection": sel}


# --------------------------------------------------------------------------------------
# Stage 2-3: QC + EDA
# --------------------------------------------------------------------------------------
def stage_qc(ds: data.Dataset, cfg: Config, rep: Report) -> dict:
    filt, flog = qc.filter_low_expression(ds, cfg.low_expression_quantile)
    qct = qc.sample_qc_table(filt)
    f1 = cfg.fig_dir / "01_expression_distributions.png"
    qc.plot_distributions(filt, path=f1)
    scores, var = qc.run_pca(filt)
    assoc = qc.pc_associations(scores, filt.meta)
    f2 = cfg.fig_dir / "02_pca.png"
    qc.plot_pca(scores, var, filt.meta, path=f2)
    _save(qct, cfg, "02_sample_qc.csv")
    _save(assoc, cfg, "02_pc_associations_R2.csv")
    n_out = int(qct["outlier"].sum())
    rep.add("Quality control", f"Low-expression filter kept {flog['genes_after']} of {flog['genes_before']} genes "
            f"(mean log2 > {flog['mean_log2_cutoff']:.2f}). Arrays flagged as outliers (robust z of mean "
            f"inter-sample correlation < -4): {n_out}.", figure=f1)
    rep.add("PCA and batch check", "Share of variance in each PC explained by each factor (R-squared). A "
            "large value for `site` is a batch/site effect; the DE models adjust for site.",
            table=assoc, figure=f2)
    rep.number("genes_after_filter", flog["genes_after"])
    rep.number("outlier_arrays", n_out)
    rep.number("PC1_R2_site", float(assoc.loc["PC1", "site"]))
    rep.number("PC1_R2_tb", float(assoc.loc["PC1", "tb_status"]))
    return {"ds": filt, "qc_table": qct, "scores": scores, "var": var, "assoc": assoc, "filter": flog}


# --------------------------------------------------------------------------------------
# Stage 4: differential expression (Objective 1 + hypothesis test)
# --------------------------------------------------------------------------------------
def stage_de(ds: data.Dataset, cfg: Config, rep: Report) -> dict:
    neg = ds.subset(ds.meta["hiv_status"] == "HIV-")
    pos = ds.subset(ds.meta["hiv_status"] == "HIV+")
    kw = dict(covariates=cfg.de_covariates, fdr=cfg.fdr, min_lfc=cfg.min_abs_log2fc)
    res_neg, info_neg = de.differential_expression(neg, **kw)
    res_pos, info_pos = de.differential_expression(pos, **kw)
    res_all, info_all = de.differential_expression(
        ds, covariates=("hiv_status", *cfg.de_covariates), fdr=cfg.fdr, min_lfc=cfg.min_abs_log2fc)
    cmp = de.compare_gene_lists(res_neg, res_pos)
    inter, _ = de.interaction_model(ds, cfg.de_covariates, cfg.fdr)
    pooled_sig = list(res_all.index[res_all["significant"]])
    att = de.attenuation_ratio(ds, pooled_sig, cfg.de_covariates, n_boot=200, seed=cfg.random_state)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    de.volcano(res_neg, f"HIV- (n={info_neg['n_case']} ATB, {info_neg['n_control']} LTBI)", cfg.fdr,
               cfg.min_abs_log2fc, ax=axes[0])
    de.volcano(res_pos, f"HIV+ (n={info_pos['n_case']} ATB, {info_pos['n_control']} LTBI)", cfg.fdr,
               cfg.min_abs_log2fc, ax=axes[1])
    fig.tight_layout()
    f_v = cfg.fig_dir / "03_volcano_hivneg_hivpos.png"
    fig.savefig(f_v, bbox_inches="tight", facecolor="white")
    f_s = cfg.fig_dir / "04_log2fc_hivneg_vs_hivpos.png"
    sig_genes = signature_genes("Sweeney_3") + ["BATF2", "ANKRD22", "FCGR1A", "GBP1", "STAT1"]
    de.lfc_scatter(res_neg, res_pos, highlight=sig_genes, path=f_s)
    f_o = cfg.fig_dir / "05_overlap.png"
    de.overlap_diagram(cmp, path=f_o)

    for name, r in [("HIVneg", res_neg), ("HIVpos", res_pos), ("pooled_adjHIV", res_all)]:
        _save(r, cfg, f"03_de_ATB_vs_LTBI_{name}.csv")
    _save(inter, cfg, "03_de_TBxHIV_interaction.csv")

    sig_rows = []
    for sname in SIGNATURES:
        for g in signature_genes(sname):
            if g in res_neg.index:
                sig_rows.append({"signature": sname, "gene": g,
                                 "log2FC_HIVneg": res_neg.loc[g, "log2FC"], "FDR_HIVneg": res_neg.loc[g, "adj.P.Val"],
                                 "log2FC_HIVpos": res_pos.loc[g, "log2FC"], "FDR_HIVpos": res_pos.loc[g, "adj.P.Val"]})
    sig_tbl = pd.DataFrame(sig_rows)
    _save(sig_tbl, cfg, "03_published_signature_genes.csv", index=False)

    n_att, n_amp = int(inter["attenuated"].sum()), int(inter["amplified"].sum())
    summary = pd.DataFrame({
        "HIV-": [info_neg["n_case"], info_neg["n_control"], int(res_neg["significant"].sum()),
                 int((res_neg["direction"] == "up").sum()), int((res_neg["direction"] == "down").sum())],
        "HIV+": [info_pos["n_case"], info_pos["n_control"], int(res_pos["significant"].sum()),
                 int((res_pos["direction"] == "up").sum()), int((res_pos["direction"] == "down").sum())],
    }, index=["active TB", "latent TB", "significant genes", "up in active TB", "down in active TB"])
    rep.add("Differential expression: active vs latent TB",
            f"limma-style moderated t-test adjusted for site; significant = BH FDR < {cfg.fdr} and "
            f"|log2FC| >= {cfg.min_abs_log2fc}. Overlap: {cmp['n_both']} shared genes, Jaccard "
            f"{cmp['jaccard']:.2f}, hypergeometric p = {cmp['hypergeom_p']:.1e}; "
            f"{cmp['shared_same_direction']} of the shared genes change in the same direction.",
            table=summary, figure=f_v)
    rep.add("Does HIV weaken the signal?",
            f"Attenuation ratio over the {att['n_genes']} genes significant in the pooled (HIV-adjusted) "
            f"analysis: **{att['ratio']:.2f}** (95% bootstrap CI {att['ci_low']:.2f}-{att['ci_high']:.2f}); "
            "1.0 = equally strong in HIV+ and HIV-, below 1 = weaker in HIV+. Formal TB x HIV interaction "
            f"test: {n_att} genes significantly attenuated and {n_amp} amplified in HIV+ (FDR < {cfg.fdr}).",
            table=inter.head(15)[["tb_effect_HIVneg", "tb_effect_HIVpos", "interaction", "adj.P.Val"]],
            figure=f_s)
    for k, v in {"de_sig_HIVneg": summary.loc["significant genes", "HIV-"],
                 "de_sig_HIVpos": summary.loc["significant genes", "HIV+"],
                 "de_overlap": cmp["n_both"], "de_jaccard": cmp["jaccard"], "de_overlap_p": cmp["hypergeom_p"],
                 "attenuation_ratio": att["ratio"], "attenuation_ci_low": att["ci_low"],
                 "attenuation_ci_high": att["ci_high"], "interaction_attenuated": n_att,
                 "interaction_amplified": n_amp, "ebayes_prior_df_HIVneg": info_neg["d0"]}.items():
        rep.number(k, v)
    return {"res_neg": res_neg, "res_pos": res_pos, "res_pooled": res_all, "compare": cmp,
            "interaction": inter, "attenuation": att, "summary": summary, "signature_genes": sig_tbl,
            "info": {"HIV-": info_neg, "HIV+": info_pos, "pooled": info_all}}


# --------------------------------------------------------------------------------------
# Stage 5: enrichment
# --------------------------------------------------------------------------------------
def stage_enrichment(de_out: dict, ds: data.Dataset, cfg: Config, rep: Report) -> dict:
    rn, rp, cmp = de_out["res_neg"], de_out["res_pos"], de_out["compare"]
    lists = {
        "HIVneg_up": list(rn.index[rn["direction"] == "up"]),
        "HIVneg_down": list(rn.index[rn["direction"] == "down"]),
        "HIVpos_up": list(rp.index[rp["direction"] == "up"]),
        "lost_in_HIVpos": cmp["only_HIVneg"],
    }
    universe = list(ds.expr.index)
    gmt = {name: enrich.read_gmt(p) for name, p in cfg.gmt_files.items()}
    out = {}
    for name, genes in lists.items():
        if cfg.online:
            r = enrich.run_enrichr(genes, cfg.enrichr_libraries, background=universe)
        else:
            r = pd.DataFrame(columns=enrich.COLUMNS)
            r.attrs["note"] = "offline run - Enrichr skipped"
        if r.empty and gmt:
            r = pd.concat([enrich.ora(genes, s, universe, library=lib) for lib, s in gmt.items()])
            r = r.sort_values("Adjusted P-value") if len(r) else r
        out[name] = r
        if len(r):
            _save(r, cfg, f"04_enrichment_{name}.csv", index=False)
    f1 = cfg.fig_dir / "06_enrichment_HIVneg_up.png"
    enrich.plot_enrichment(out["HIVneg_up"], "Pathways up in active TB, HIV-", path=f1)
    f2 = cfg.fig_dir / "07_enrichment_lost_in_HIVpos.png"
    enrich.plot_enrichment(out["lost_in_HIVpos"], "Pathways of genes significant in HIV- only", path=f2)
    top = out["HIVneg_up"].head(10)
    note = out["HIVneg_up"].attrs.get("note", "")
    rep.add("Enrichment (GO / KEGG / Reactome via Enrichr)",
            note or "Top terms for genes higher in active TB (HIV-); background = all tested genes.",
            table=top[["library", "Term", "Overlap", "Adjusted P-value"]] if len(top) else None,
            figure=f1, index=False)
    rep.number("enrichment_terms_HIVneg_up_FDR05",
               int((out["HIVneg_up"]["Adjusted P-value"].astype(float) < 0.05).sum()) if len(out["HIVneg_up"]) else 0)
    return {"lists": lists, "results": out}


# --------------------------------------------------------------------------------------
# Stage 6: network
# --------------------------------------------------------------------------------------
def _top_genes(res, n):
    sig = res[res["significant"]]
    pick = sig if len(sig) >= 10 else res
    return list(pick.sort_values("P.Value").index[:n])


def stage_network(de_out: dict, ds: data.Dataset, cfg: Config, rep: Report) -> dict:
    out = {}
    for label, res, hiv in [("HIV-", de_out["res_neg"], "HIV-"), ("HIV+", de_out["res_pos"], "HIV+")]:
        genes = _top_genes(res, cfg.network_top_n)
        expr = ds.subset(ds.meta["hiv_status"] == hiv).expr
        if cfg.online:
            G, _ = network.build_network(genes, res, expr=expr, min_score=cfg.string_min_score,
                                           fallback_rho=cfg.coexpression_min_rho)
        else:
            G = network.nx.Graph()
            G.add_nodes_from(genes)
            G.add_weighted_edges_from(network.coexpression_edges(expr, genes, cfg.coexpression_min_rho)
                                      .itertuples(index=False))
            for g in G.nodes:
                G.nodes[g]["log2FC"] = float(res.loc[g, "log2FC"])
            G.graph["source"] = "co-expression"
        hubs = network.hub_table(G, top=20)
        out[label] = {"graph": G, "hubs": hubs, "summary": network.network_summary(G)}
        _save(hubs, cfg, f"05_hub_genes_{label.replace('+', 'pos').replace('-', 'neg')}.csv")
    f = cfg.fig_dir / "08_network_HIVneg.png"
    network.plot_network(out["HIV-"]["graph"], out["HIV-"]["hubs"],
                         f"{out['HIV-']['summary']['source']} network of top active-vs-latent TB genes (HIV-)", path=f)
    shared = sorted(set(out["HIV-"]["hubs"].index[:10]) & set(out["HIV+"]["hubs"].index[:10]))
    s = out["HIV-"]["summary"]
    rep.add("Network and hub genes",
            f"Source: {s['source']}; {s['genes_connected']} of {s['genes_queried']} genes connected by "
            f"{s['edges']} edges. Top-10 hubs shared between HIV- and HIV+ networks: "
            f"{', '.join(shared) if shared else 'none'}.",
            table=out["HIV-"]["hubs"].head(10), figure=f)
    rep.number("network_source", s["source"])
    rep.number("hubs_HIVneg_top5", list(out["HIV-"]["hubs"].index[:5]))
    return out


# --------------------------------------------------------------------------------------
# Stage 7: machine learning (Objective 3)
# --------------------------------------------------------------------------------------
def stage_ml(ds: data.Dataset, cfg: Config, rep: Report, repeats=None) -> dict:
    repeats = cfg.cv_repeats if repeats is None else repeats
    groups = {h: ds.subset(ds.meta["hiv_status"] == h) for h in ("HIV-", "HIV+")}
    XY, oofs, ys, summaries, fold_tables, models_by = {}, {}, {}, [], [], {}
    for g, d in groups.items():
        print(f"Cross-validating within {g} ({d.shape[1]} samples, {cfg.cv_folds}-fold x {repeats})")
        X, y = ml.xy(d)
        models = ml.make_models(cfg, X.shape[1], genes=X.columns)
        folds, oof = ml.cross_validate(models, X, y, strata=d.meta.loc[X.index, "site"], folds=cfg.cv_folds,
                                       repeats=repeats, seed=cfg.random_state)
        folds["group"] = g
        XY[g], oofs[g], ys[g], models_by[g] = (X, y), oof, y, models
        summaries.append(ml.summarise_cv(oof, y, cfg.target_sensitivity, cfg.bootstrap_iterations, label=g))
        fold_tables.append(folds)
    summary = pd.concat(summaries)
    _save(summary, cfg, "06_ml_cv_summary_by_HIV.csv")
    _save(pd.concat(fold_tables), cfg, "06_ml_cv_folds.csv", index=False)
    f_roc = cfg.fig_dir / "09_roc_by_HIV.png"
    ml.plot_roc(oofs, ys, path=f_roc)
    f_auc = cfg.fig_dir / "10_auc_by_HIV.png"
    ml.plot_auc_by_group(summary, path=f_auc)

    # Pooled model, one threshold, metrics reported separately by HIV status and site
    print("Cross-validating pooled model (HIV- and HIV+ together)")
    Xa, ya = ml.xy(ds)
    meta_a = ds.meta.loc[Xa.index]
    models_a = ml.make_models(cfg, Xa.shape[1], genes=Xa.columns)
    _, oof_a = ml.cross_validate(models_a, Xa, ya, strata=meta_a["site"] + meta_a["hiv_status"],
                                 folds=cfg.cv_folds, repeats=max(1, repeats // 2), seed=cfg.random_state)
    sub = []
    for name, p in oof_a.items():
        if name == "Majority class":
            continue
        r = ml.subgroup_report(ya, p, meta_a, target=cfg.target_sensitivity)
        r.insert(0, "model", name)
        sub.append(r)
    subgroup = pd.concat(sub, ignore_index=True)
    _save(subgroup, cfg, "07_pooled_model_metrics_by_subgroup.csv", index=False)

    # Transfer between HIV groups
    print("Transfer: train HIV-, test HIV+ (and the reverse)")
    t_np = ml.transfer(models_by["HIV-"], *XY["HIV-"], *XY["HIV+"], src_oof=oofs["HIV-"],
                       target=cfg.target_sensitivity, n_boot=cfg.bootstrap_iterations, seed=cfg.random_state)
    t_pn = ml.transfer(models_by["HIV+"], *XY["HIV+"], *XY["HIV-"], src_oof=oofs["HIV+"],
                       target=cfg.target_sensitivity, n_boot=cfg.bootstrap_iterations, seed=cfg.random_state)
    _save(t_np, cfg, "08_transfer_HIVneg_to_HIVpos.csv")
    _save(t_pn, cfg, "08_transfer_HIVpos_to_HIVneg.csv")
    f_tr = cfg.fig_dir / "11_transfer_thresholds.png"
    ml.plot_transfer({"Trained HIV-, tested HIV+": t_np, "Trained HIV+, tested HIV-": t_pn},
                     path=f_tr, target=cfg.target_sensitivity)

    # Leave-one-site-out (Cape Town <-> Karonga)
    print("Leave-one-site-out")
    lso = []
    for src_site in sorted(meta_a["site"].unique()):
        s_mask = (meta_a["site"] == src_site).values
        if s_mask.all() or (~s_mask).all():
            continue
        Xs, ys_, Xt, yt = Xa[s_mask], ya[s_mask], Xa[~s_mask], ya[~s_mask]
        _, oof_s = ml.cross_validate(models_a, Xs, ys_, strata=meta_a.loc[Xs.index, "hiv_status"],
                                     folds=cfg.cv_folds, repeats=1, seed=cfg.random_state, verbose=False)
        t = ml.transfer(models_a, Xs, ys_, Xt, yt, src_oof=oof_s, target=cfg.target_sensitivity,
                        n_boot=300, seed=cfg.random_state)
        t["train_site"], t["test_site"] = src_site, ", ".join(sorted(meta_a.loc[Xt.index, "site"].unique()))
        lso.append(t)
    lso = pd.concat(lso) if lso else pd.DataFrame()
    if len(lso):
        _save(lso, cfg, "09_leave_one_site_out.csv")

    print("Fitting final models on all HIV- samples for feature inspection")
    feats = ml.fitted_feature_table(groups["HIV-"], cfg)
    _save(feats, cfg, "10_top_model_genes_HIVneg.csv")

    # Report
    cols = ["group", "AUC", "AUC_low", "AUC_high", "sensitivity", "specificity"]
    rep.add("Machine learning: cross-validated performance by HIV status",
            f"{cfg.cv_folds}-fold CV x {repeats} repeats within each HIV group; gene filter (top {cfg.k_features} by "
            f"F-test) fitted inside each training fold. Sensitivity/specificity at the threshold giving "
            f">= {cfg.target_sensitivity:.0%} sensitivity.", table=summary[cols], figure=f_auc)
    tcols = ["AUC", "AUC_low", "AUC_high", "sensitivity", "missed_TB", "n_TB", "specificity",
             "specificity_group_threshold"]
    rep.add("Transfer: model trained on HIV- patients, tested on HIV+ patients",
            "`sensitivity`/`specificity` use the threshold chosen on HIV- training data, applied unchanged. "
            "`specificity_group_threshold` re-chooses the threshold on HIV+ patients to reach the target "
            "sensitivity (split-half, so the estimate is honest).", table=t_np[tcols], figure=f_tr)
    sub_view = subgroup[subgroup["by"] == "hiv_status"][["model", "group", "n", "AUC", "sensitivity",
                                                           "specificity", "missed_TB", "n_TB"]]
    rep.add("Pooled model: misses reported by HIV status (one shared threshold)", table=sub_view, index=False)
    if len(lso):
        rep.add("Leave-one-site-out (South Africa <-> Malawi)", table=lso[["train_site", "test_site", "AUC",
                                                                           "AUC_low", "AUC_high"]])
    rep.add("Genes used by the final HIV- models",
            f"L1 logistic regression kept {feats.attrs.get('n_L1_nonzero')} genes (C = {feats.attrs.get('L1_C')}).",
            table=feats.head(15))
    for g in ("HIV-", "HIV+"):
        for m in summary.index.unique():
            r = summary[(summary["group"] == g)].loc[m]
            rep.number(f"cv_auc_{g}_{m}", r["AUC"])
    for m in t_np.index:
        rep.number(f"transfer_auc_HIVneg_to_HIVpos_{m}", t_np.loc[m, "AUC"])
        rep.number(f"transfer_sens_HIVneg_to_HIVpos_{m}", t_np.loc[m, "sensitivity"])
    return {"summary": summary, "oofs": oofs, "ys": ys, "subgroup": subgroup, "transfer_neg_to_pos": t_np,
            "transfer_pos_to_neg": t_pn, "leave_site_out": lso, "features": feats, "XY": XY}


# --------------------------------------------------------------------------------------
# Stage 8: external validation on GSE19491
# --------------------------------------------------------------------------------------
def stage_external(ds: data.Dataset, val: data.Dataset | None, cfg: Config, rep: Report) -> dict:
    if val is None:
        rep.add("External validation (GSE19491)", "Skipped: validation data not available.")
        return {}
    train = ds.subset((ds.meta["hiv_status"] == "HIV-") & ds.meta["tb_status"].isin(["ATB", "LTBI"]))
    test = val.subset(val.meta["tb_status"].isin(["ATB", "LTBI"]))
    genes = sorted(set(train.expr.index) & set(test.expr.index))
    Xs = ml.zscore_genes(train.expr.loc[genes]).T
    zt = pd.concat([ml.zscore_genes(test.subset(test.meta["dataset"] == d).expr.loc[genes])
                    for d in test.meta["dataset"].unique()], axis=1)[test.meta.index]
    Xt = zt.T
    ys = (train.meta.loc[Xs.index, "tb_status"] == "ATB").astype(int).values
    yt = (test.meta.loc[Xt.index, "tb_status"] == "ATB").astype(int).values
    models = ml.make_models(cfg, len(genes), genes=genes)
    print(f"External validation: train {cfg.main_gse} HIV- ({len(ys)}), test GSE19491 ({len(yt)}), {len(genes)} shared genes")
    _, oof = ml.cross_validate(models, Xs, ys, strata=train.meta.loc[Xs.index, "site"], folds=cfg.cv_folds,
                               repeats=1, seed=cfg.random_state, verbose=False)
    tbl = ml.transfer(models, Xs, ys, Xt, yt, src_oof=oof, target=cfg.target_sensitivity,
                      n_boot=cfg.bootstrap_iterations, seed=cfg.random_state)
    # per site of the validation cohort
    fitted = {}
    rows = []
    for name, m in models.items():
        fm = ml.clone(m).fit(Xs, ys)
        p = fm.predict_proba(Xt)[:, 1]
        fitted[name] = p
        for site in test.meta["site"].unique():
            k = (test.meta.loc[Xt.index, "site"] == site).values
            if len(set(yt[k])) > 1:
                auc, lo, hi = ml.auc_ci(yt[k], p[k], 500)
                rows.append({"model": name, "site": site, "n": int(k.sum()), "AUC": auc, "AUC_low": lo, "AUC_high": hi})
    by_site = pd.DataFrame(rows)
    _save(tbl, cfg, "11_external_validation_GSE19491.csv")
    _save(by_site, cfg, "11_external_validation_by_site.csv", index=False)
    f = cfg.fig_dir / "12_external_roc.png"
    ml.plot_roc({"GSE19491 (HIV-, external)": fitted}, {"GSE19491 (HIV-, external)": yt}, path=f)
    counts = pd.crosstab(test.meta["site"], test.meta["tb_status"])
    rep.add("External validation on GSE19491 (Berry et al. 2010; HIV- only, London + South Africa)",
            f"Models trained on all {cfg.main_gse} HIV- patients; genes z-scored within each dataset "
            f"(different array versions). Test set: {dict(counts.sum())}. This checks the HIV- result only - "
            "GSE19491 has no HIV+ patients.", table=tbl[["AUC", "AUC_low", "AUC_high", "sensitivity", "specificity"]],
            figure=f)
    rep.add("External validation by site", table=by_site, index=False)
    for m in tbl.index:
        rep.number(f"external_auc_{m}", tbl.loc[m, "AUC"])
    return {"table": tbl, "by_site": by_site, "probs": fitted, "y": yt, "counts": counts}


# --------------------------------------------------------------------------------------
# Answer to the research question, written from the numbers (no hand-typed results)
# --------------------------------------------------------------------------------------
def conclusions(de_out: dict, ml_out: dict, ext_out: dict | None = None, target: float = 0.9) -> str:
    att = de_out["attenuation"]
    inter = de_out["interaction"]
    s = ml_out["summary"]
    lines = []

    # Q1: can blood expression distinguish active from latent TB?
    models = [m for m in s.index.unique() if m != "Majority class"]
    neg = s[s["group"] == "HIV-"].loc[models]
    pos = s[s["group"] == "HIV+"].loc[models]
    best = neg["AUC"].idxmax()
    lines.append(
        f"**1. Active vs latent TB.** In HIV- adults the best model ({best}) reached a cross-validated AUC of "
        f"{neg.loc[best, 'AUC']:.2f} (95% CI {neg.loc[best, 'AUC_low']:.2f}-{neg.loc[best, 'AUC_high']:.2f}); "
        f"{de_out['summary'].loc['significant genes', 'HIV-']} genes were differentially expressed (FDR < 0.05).")
    if "Sweeney 3-gene score" in neg.index and best != "Sweeney 3-gene score":
        sw = neg.loc["Sweeney 3-gene score"]
        beats = neg.loc[best, "AUC_low"] > sw["AUC_high"]
        lines.append(
            f"Against the published 3-gene score (AUC {sw['AUC']:.2f}, CI {sw['AUC_low']:.2f}-{sw['AUC_high']:.2f}) "
            + ("the learned model is clearly better (CIs do not overlap)." if beats else
               "the learned model's advantage is within the uncertainty - a 3-gene score does nearly as well."))

    # Q2: does HIV weaken the signal?
    if att["ci_high"] < 1:
        verdict = (f"Yes. On the genes that separate active from latent TB overall, the HIV+ effect is "
                   f"{att['ratio']:.2f}x the HIV- effect (95% CI {att['ci_low']:.2f}-{att['ci_high']:.2f}), i.e. "
                   f"about {1 - att['ratio']:.0%} weaker.")
    elif att["ci_low"] > 1:
        verdict = (f"No - the signal is stronger in HIV+ (ratio {att['ratio']:.2f}, CI {att['ci_low']:.2f}-"
                   f"{att['ci_high']:.2f}), contrary to the hypothesis.")
    else:
        verdict = (f"Not clearly. The ratio of HIV+ to HIV- effects is {att['ratio']:.2f} (CI {att['ci_low']:.2f}-"
                   f"{att['ci_high']:.2f}), which includes 1 (no difference).")
    lines.append(f"**2. Does HIV weaken the signal?** {verdict} The gene-by-gene TB x HIV interaction test found "
                 f"{int(inter['attenuated'].sum())} significantly attenuated and {int(inter['amplified'].sum())} "
                 f"amplified genes.")
    lines.append(
        f"Classifier AUC for {best}: HIV- {neg.loc[best, 'AUC']:.2f} vs HIV+ {pos.loc[best, 'AUC']:.2f} "
        f"(CI {pos.loc[best, 'AUC_low']:.2f}-{pos.loc[best, 'AUC_high']:.2f}). Specificity at the cut-off giving "
        f">= {target:.0%} sensitivity: {neg.loc[best, 'specificity']:.0%} (HIV-) vs "
        f"{pos.loc[best, 'specificity']:.0%} (HIV+).")

    # Transfer
    t = ml_out["transfer_neg_to_pos"].loc[best]
    lines.append(
        f"**3. Does an HIV- model work in HIV+ patients?** Trained on HIV- and tested on HIV+, {best} scored "
        f"AUC {t['AUC']:.2f} (CI {t['AUC_low']:.2f}-{t['AUC_high']:.2f}). With the threshold tuned on HIV- patients "
        f"it detected {t['sensitivity']:.0%} of active TB cases ({int(t['missed_TB'])} of {int(t['n_TB'])} missed) "
        f"at {t['specificity']:.0%} specificity; re-tuning the threshold for HIV+ patients gave "
        f"{t['specificity_group_threshold']:.0%} specificity at the target sensitivity.")

    if ext_out:
        e = ext_out["table"]
        e = e.loc[[m for m in e.index if m != "Majority class"]]
        lines.append(
            "**4. External validation (GSE19491, HIV- only).** AUCs: "
            + "; ".join(f"{m} {r['AUC']:.2f} ({r['AUC_low']:.2f}-{r['AUC_high']:.2f})" for m, r in e.iterrows())
            + ". This tests the HIV- result only: no independent HIV+ cohort was available.")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------------------
def run_all(cfg: Config | None = None, data_note: str = "") -> dict:
    cfg = cfg or Config()
    cfg.ensure_dirs()
    rep = Report(cfg.results_dir, "TB vs latent TB in HIV- and HIV+ African adults: results summary")
    out = {}
    out["load"] = stage_load(cfg, rep)
    out["select"] = stage_select(out["load"]["main"], cfg, rep)
    plt.close("all")
    out["qc"] = stage_qc(out["select"]["ds"], cfg, rep)
    plt.close("all")
    ds = out["qc"]["ds"]
    out["de"] = stage_de(ds, cfg, rep)
    plt.close("all")
    out["enrichment"] = stage_enrichment(out["de"], ds, cfg, rep)
    plt.close("all")
    out["network"] = stage_network(out["de"], ds, cfg, rep)
    plt.close("all")
    out["ml"] = stage_ml(ds, cfg, rep)
    plt.close("all")
    out["external"] = stage_external(ds, out["load"]["val"], cfg, rep)
    plt.close("all")
    out["conclusions"] = conclusions(out["de"], out["ml"], out["external"], cfg.target_sensitivity)
    rep.sections.insert(0, ("Answer to the research question", out["conclusions"]))
    out["report_path"] = rep.write(data_note)
    out["report"] = rep
    return out
