# Blood gene signatures for active vs latent TB in HIV− and HIV+ African adults

Computing Bioinformatics group project · **Matthew Nguithi (162156) · Ray Lukorito (168110) · William Githinji (110592)**

> **Research question.** Can blood gene expression distinguish active tuberculosis from latent TB infection in African adults, and does HIV co-infection weaken that signal?
>
> **Hypothesis.** Active TB shows strong up-regulation of interferon-driven immune genes in both groups, but the active-vs-latent difference (and classifier accuracy) is lower in HIV-positive patients.

## Objectives → where they are answered

| Objective | Method | Code | Notebook |
|---|---|---|---|
| 1. DE genes, active vs latent TB, separately for HIV− and HIV+, FDR-corrected; compare overlap | limma-style moderated t (empirical Bayes) adjusted for site, Benjamini–Hochberg; hypergeometric overlap test; **TB × HIV interaction model** and **attenuation ratio** with bootstrap CI | `de.py` | §3 |
| 2. Biological interpretation | Enrichr (GO BP 2023, KEGG 2021, Reactome 2022) with tested genes as background; STRING network (score ≥ 700) → NetworkX degree/betweenness hubs and communities | `enrich.py`, `network.py` | §4–5 |
| 3. ML classifiers vs baselines; HIV− → HIV+ transfer | Majority class, Sweeney 3-gene score, Kaforou 27-gene score, L1 logistic regression, random forest; 5 × 5 repeated stratified CV with in-fold gene filtering; train-HIV−/test-HIV+; leave-one-site-out; external validation on GSE19491 | `ml.py`, `signatures.py` | §6–7 |

## Data

| Dataset | Role | Samples used | Platform |
|---|---|---|---|
| **GSE37250** (Kaforou et al., 2013, *PLoS Med* 10:e1001538) | discovery | 537 total → 362 active/latent TB (Cape Town + Karonga; 180 HIV−, 182 HIV+) | Illumina HumanHT-12 V4 (GPL10558) |
| **GSE19491** (Berry et al., 2010, *Nature* 466:973) | external validation, **HIV− only** | 54 active TB + 69 latent TB from sub-series GSE19439, GSE19444 (London) and GSE19442 (South Africa) | Illumina HumanHT-12 V3 (GPL6947) |

The notebook downloads the GEO series matrices itself (no login). Probes are mapped to genes with GEOparse platform annotations.

## How to run

**Google Colab (recommended).**
1. Push this folder to the group GitHub repo.
2. In `notebooks/TB_HIV_analysis.ipynb`, set `REPO_URL` in the first code cell and the Colab badge link to that repo.
3. Open the notebook in Colab and choose *Runtime → Run all*. It takes about 10–20 minutes on the free CPU runtime; switch to a high-RAM runtime if it runs short of memory.

Outputs go to `results/`: `summary.md` (all headline numbers plus the auto-written answer to the research question), `results.json`, `figures/*.png` (12 figures) and `tables/*.csv` (full DE tables, enrichment, hub genes, CV folds, transfer and validation metrics).

**Locally / without the notebook.**
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py              # full run, real data
python scripts/run_pipeline.py --repeats 2  # quicker
python -m pytest                            # 30 tests, about 1 minute, offline
```

**If NCBI is slow or blocked:** download `GSE37250_series_matrix.txt.gz` from GEO and pass it with `--matrix GSE37250=path` (or `Config(local_matrix_files=...)` in the notebook).

## Repository layout
```
notebooks/TB_HIV_analysis.ipynb   narrative notebook, one section per stage
src/tbhiv/
  config.py      every tunable setting (FDR, |log2FC|, CV folds/repeats, k genes, target sensitivity…)
  data.py        GEO download, series-matrix parser, label parsing, log2 + quantile normalisation, probe→gene
  qc.py          low-expression filter, array QC, PCA, variance explained by TB / HIV / site
  de.py          linear models + empirical Bayes, BH FDR, overlap, TB×HIV interaction, attenuation ratio
  enrich.py      Enrichr via gseapy; offline GMT over-representation
  network.py     STRING API → NetworkX hubs/communities; co-expression fallback
  ml.py          models, repeated CV, thresholds, subgroup metrics, transfer, plots
  signatures.py  Sweeney 3-gene and Kaforou 27-gene signatures (from TBSignatureProfiler)
  pipeline.py    the stages as functions + run_all() + auto-written conclusions
  synthetic.py   GEO-format SYNTHETIC data for tests only
scripts/run_pipeline.py
tests/            parser, statistics, leakage, planted-signal recovery, end-to-end
```

## Methodological safeguards (worth mentioning in the presentation)
* **Site/batch effect.** PCA reports how much variance site explains, and every DE model includes site as a covariate. A unit test confirms that a pure site shift confounded with TB status produces no false TB genes once site is adjusted for.
* **No data leakage.** Gene filtering and hyperparameter tuning happen inside each training fold. A unit test confirms random labels give AUC ≈ 0.5.
* **Winner's curse.** The attenuation ratio uses genes selected in the pooled analysis, not in HIV−, which would inflate HIV− effects and bias the ratio towards "HIV weakens the signal".
* **Proper baselines.** The models must beat the majority class and two published signatures, not just score a high AUC.
* **Clinically meaningful thresholds.** Cut-offs give ≥ 90% sensitivity (WHO triage-test target) on training data and are applied unchanged to the test group, so missed TB cases in HIV+ patients are visible. A re-tuned HIV+-specific threshold shows how much of that a group-specific threshold recovers.

## Limitations and responsible use
Two Southern African adult cohorts only; the validation set is HIV− only (so HIV+ findings are not independently validated); case–control design overstates real-world accuracy. The data are public and de-identified. We use them only for this coursework, make no attempt to re-identify anyone, share only group-level results, and do not present any model as a diagnostic test.

## Testing note
The code was developed and tested offline on **synthetic** files in GEO format (`tbhiv.synthetic`) that copy GSE37250's sample layout, with known effects planted in them. The tests check that the pipeline recovers those effects. Synthetic numbers are not results; real results come from running the notebook on GEO data.

## References
* Kaforou M, et al. Detection of tuberculosis in HIV-infected and -uninfected African adults using whole blood RNA expression signatures: a case-control study. *PLoS Med* 2013;10(10):e1001538.
* Berry MPR, et al. An interferon-inducible neutrophil-driven blood transcriptional signature in human tuberculosis. *Nature* 2010;466:973–977.
* Sweeney TE, et al. Genome-wide expression for diagnosis of pulmonary tuberculosis: a multicohort analysis. *Lancet Respir Med* 2016;4(3):213–224.
* Smyth GK. Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Stat Appl Genet Mol Biol* 2004;3:Article 3.
* Johnson WE, et al. TBSignatureProfiler (Bioconductor), source of the curated signature gene lists.
