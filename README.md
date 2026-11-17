# What Australian Health Practitioner Tribunals Decide, and What Automated Analysis Misses

Analysis code and released labels for: **What Australian Health Practitioner Tribunals Decide, and What Automated Analysis Misses: 3,586 Decisions, 1999 to February 2026**

Hayden Farquhar MBBS MPHTM, Independent Researcher, Finley, New South Wales, Australia. ORCID [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)

Preprint: version 1, [doi:10.64898/2026.02.13.26346299](https://doi.org/10.64898/2026.02.13.26346299). This release (v2.0.0) accompanies version 2 of the preprint, which corrects version 1 and reports a revised analysis.

## Overview

The study assembled 3,586 published Australian health practitioner tribunal decisions (December 1999 to February 2026, eight jurisdictions) and asked what tribunals decide and how that has changed. Primary estimates come from a random sample of 160 decisions, stratified by period and coded by the author blind to any machine output; 60 of them were recoded by an independent second coder. A TF-IDF and logistic-regression classifier, trained on a 300-decision gold standard under a protocol fixed in advance, labels the whole corpus for exploratory analyses, and the coded sample measures how far those labels can be trusted.

This repository contains the code for every stage (corpus assembly, coding packets, gold-standard construction, classifier training, corpus-wide analysis, and validation against human coding), the written protocols, the coding guidelines, the released gold-standard labels, and the aggregate tables, statistics, and figures the code produces.

## What is and is not released

| Released | Not released |
|---|---|
| Code (`src/`), protocols and configuration (`config/`), coding guidelines (`docs/`) | Decision texts and the assembled corpus (practitioner names; obtain from the sources below) |
| Gold-standard labels for 300 decisions, keyed by neutral citation, with per-field provenance (`data/annotations/`) | Free-text notes and adjudication rationales (they can name practitioners) |
| The language-model coder's system prompt, output schema, and run summary | The model's raw responses (free text) |
| Train, validation, and test split identifiers | Coding workbooks for the coded sample and the second coder (individual codes; coder identity) |
| Ahpra registrant counts transcribed from annual reports (`data/reference/`) | Trained model files (re-create with `retrain_2026.py`) |
| Aggregate tables, statistics, and figures (`outputs/`) | Platform scripts used to recruit and pay the second coder |

`data/annotations/gold_standard_2026-11.json` is the gold standard with the `adjudication_rationales` field removed, so its hash differs from the one recorded in the manifest, which describes the full internal file. Every coded field is unchanged.

## Data sources

| Source | URL | Access |
|---|---|---|
| Australasian Legal Information Institute (AustLII) | https://www.austlii.edu.au/ | Free; observe the crawl delay in AustLII's robots.txt |
| Open Australian Legal Corpus (Butler, Hugging Face) | https://doi.org/10.57967/hf/2784 | Free; see the dataset card for its licence |
| Internet Archive Wayback Machine | https://web.archive.org/ | Free |
| Ahpra and National Boards annual reports (registrant counts) | https://www.ahpra.gov.au/Publications/Annual-reports/Annual-report-archive | Free; transcribed values and page references in `data/reference/` |
| Poisons Standard, October 2026 (F2026L01327) | https://www.legislation.gov.au/F2026L01327/latest/text | Free |

## Requirements

Python 3.12. The analysis environment is pinned in `requirements/requirements-ml-arm64.lock.txt` (built on macOS arm64, 3 November 2026); `requirements/requirements-scraping.txt` covers corpus collection.

```bash
python3.12 -m venv .venv-ml
.venv-ml/bin/pip install -r requirements/requirements-ml-arm64.lock.txt
```

The language-model coder (`src/annotation/model_second_coder.py`) needs an Anthropic API key, set as `ANTHROPIC_API_KEY` in the shell or in a `.env` file at the repository root (git-ignored), and incurs API charges. `src/ner/build_poisons_schedule_map.py` needs pandoc.

## Reproduction

Run each step from the repository root as its docstring shows: most as modules (for example `.venv-ml/bin/python -m src.analysis.registrant_rates`), and the packet builders and `check_recode` as scripts (for example `python3 src/preprocessing/build_test2_packet.py`). Corpus assembly (step 1) uses `requirements/requirements-scraping.txt`. Steps 1 to 3 need the decision texts, so they cannot be run from this repository alone; steps 4 to 6 reproduce the reported results once the corpus and the coding workbooks are in place. Each script's docstring gives its inputs and outputs.

1. **Corpus assembly** (`src/scraping/`): `run_pipeline` (AustLII), `corpus_download` and `corpus_download_nonnsw` (Open Australian Legal Corpus), `wayback_scraper` (legacy tribunals), `merge_corpus` (deduplication); then `src/preprocessing/metadata_enrichment`.
2. **Gold standard** (`src/preprocessing/`, `src/annotation/`): `sample_selection` (the 300-decision sample); `build_recode_packet` (blinded packet for the author's recode) and `check_recode`; `model_second_coder` (logged language-model second coder: `count`, `submit`, `collect`); `src/analysis/irr_recode_2026` (agreement); `build_adjudication_sheet`, `check_adjudication`, and `build_gold_standard_2026`.
3. **Supplementary training labels**: `src/annotation/expand_gold_standard` (sample of 300 further decisions), `src/preprocessing/build_supplementary_texts`, and `model_second_coder --set supplementary`.
4. **Classifier and corpus-wide labels** (`src/classification/`, `src/ner/`, `src/topic_modelling/`): `data_preparation`, `retrain_2026` (protocol in `config/retrain_protocol_2026-11.md`; selection on the validation set, test set scored once), `outcome_extraction`, `src/ner/build_poisons_schedule_map`, `src/ner/entity_extraction`, `src/topic_modelling/bertopic_analysis`, and `predict_full_corpus`.
5. **Corpus-wide analyses** (`src/analysis/`): `statistical_analysis`, `revision_analyses`, `validation_analysis`, `prescribing_analysis`, and `registrant_rates`.
6. **Coded sample and validation** (`src/preprocessing/build_test2_packet`, then `src/analysis/`): `score_test2` (classifier on the coded sample, scored once), `model_second_coder --set test2` and `score_llm_test2` (language-model coder, scored once), `human_reference_estimates` (weighted primary estimates), `figure_fraud_drift`, `validation_summary_table`, `sensitivity_author_reference`, and `second_coder_agreement_2026`.

Corpus collection is the slowest step, because AustLII's crawl delay is observed for every request. `data_preparation` rebuilds the splits in `data/annotations/splits/split_ids.json` from the gold standard.

## Protocols

| File | Covers |
|---|---|
| `config/retrain_protocol_2026-11.md` | Classifier training sets, model selection, and the single scoring of the test set |
| `config/external_test_protocol_2026-11.md` | Sampling, coding, and scoring of the 160-decision coded sample, with dated amendments |
| `config/llm_coder_protocol_2026-11.md` | The test of whether the language-model coder should replace the classifier, and the fallback |
| `config/second_coder_protocol_2026-11.md` | Recruitment, blinding, and agreement analysis for the independent second coder |

In the public copies, internal project identifiers have been removed; nothing else differs from the frozen versions.

## Outputs and the paper

| Paper | File(s) | Script |
|---|---|---|
| Figure 1 | `outputs/figures/fig22_corpus_flow_mermaid.mmd` (rendered `.png`) | Counts from steps 1 to 6 |
| Figure 2 | `outputs/figures/fig20_human_vs_classifier_prevalence.png` | `human_reference_estimates` |
| Figure 3 | `outputs/figures/fig19_registrant_rates.png` | `registrant_rates` |
| Figure 4 | `outputs/figures/fig21_fraud_drift.png` | `figure_fraud_drift` |
| Table 1 | `outputs/tables/table01_corpus_overview.csv`, `table02_profession_distribution.csv` | `statistical_analysis` |
| Table 2 | `outputs/tables/table38_coded_sample_estimates.csv`, `table35_human_reference_prevalence.csv` | `human_reference_estimates` |
| Table 3 | `outputs/tables/table37_validation_summary.csv` | `validation_summary_table` |
| Table 4 | `outputs/tables/table36_human_reference_by_period.csv` | `human_reference_estimates` |
| Table 5 | `outputs/tables/table34_registrant_rates.csv` | `registrant_rates` |
| Table 6 | `outputs/tables/table07_profession_x_outcome.csv` | `statistical_analysis` |
| Table 7 | `outputs/tables/table11_top20_medications.csv` | `statistical_analysis` |
| Supplementary tables and figures | Other files in `outputs/tables/`, `outputs/statistics/`, and `outputs/figures/` | `statistical_analysis`, `revision_analyses`, `validation_analysis`, `prescribing_analysis`, and step 6 |

## Earlier scripts

`src/preprocessing/llm_annotation.py`, `src/analysis/irr_analysis.py`, `src/analysis/llm_irr_comparison.py`, `src/classification/retrain_expanded.py`, and `notebooks/legal_bert_training.ipynb` belong to version 1 of the analysis. They are kept for transparency but are not used for any version 2 result: the version 1 gold standard and its agreement statistics were replaced, and no Legal-BERT model was trained.

## Citation

> Farquhar H. What Australian Health Practitioner Tribunals Decide, and What Automated Analysis Misses: 3,586 Decisions, 1999 to February 2026. medRxiv preprint. https://doi.org/10.64898/2026.02.13.26346299

Machine-readable citation metadata is in `CITATION.cff`.

## Licence

Code: MIT (see `LICENCE`). Released labels, protocols, guidelines, and documentation: CC BY 4.0.
