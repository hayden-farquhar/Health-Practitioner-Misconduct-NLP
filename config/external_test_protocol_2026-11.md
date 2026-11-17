# Second test set protocol (fixed 2026-11-06, before any decision was sampled)

Approved in outline by Hayden Farquhar in chat on 2026-11-06 (Part C item 5: "bigger test set,
coded by Hayden"). Written before sampling so that every choice below is fixed in advance. Any
departure is reported as a departure.

## Why

The first test set has 45 decisions. Per-class F1 ranges from 0.43 to 0.90 and the bootstrap
CIs are wide. The two significant misconduct trends (FRAUD, COMM) assume that the classifier
performs the same across the study period, which 45 decisions cannot check. The first test set
also came from a profession-stratified sample, so it does not represent the corpus.

## Models (frozen; nothing is retrained or re-selected)

The models chosen under `config/retrain_protocol_2026-11.md`, as on 2026-11-06 (sha1 prefixes):

| File | sha1 |
|---|---|
| `models/sklearn_misconduct/misconduct_clf.pkl` | 6fa13a4c3b31 |
| `models/sklearn_misconduct/disciplinary_clf.pkl` | 65ee6fce82df |
| `models/sklearn_misconduct/finding_clf.pkl` | 451e923f131e |
| `models/sklearn_misconduct/finding_le.pkl` | 99066fa715b9 |
| `models/sklearn_misconduct/vectorizer.pkl` | 74ff06d53607 |
| `models/sklearn_misconduct/vectorizer_finding.pkl` | 74ff06d53607 |
| `src/classification/outcome_extraction.py` | 0af0e9de8df6 |
| `src/analysis/statistical_analysis.py` (boundary keywords) | a78b5cc73458 (amended; see below). `boundary_analysis` source: 69c2609ae87b |

The scoring script checks these hashes and stops if any differ.

## Sample

- **Pool:** every decision in `data/extracted/all_decisions.jsonl` that has never been labelled
  by anyone, with full text of at least 500 characters and a known date. "Labelled" means it
  appears in the gold standard (300), the 2026-11 supplementary set (300), `new_sample_300_ids.json`
  or `splits/train_expanded.jsonl` (600 distinct decisions in total). Pool: 2,978 decisions.
- **Strata (decision year):** 1999–2009, 2010–2014, 2015–2019, 2020–2026. The 2010 boundary is the
  start of the National Law.
- **Allocation:** 40 per stratum, 160 in total, simple random sampling within each stratum
  (seed 20261106). Equal allocation gives the early years enough decisions to check performance
  across time. Overall estimates are weighted back to the pool (below).
- Item order is shuffled with the same seed, so strata are not coded in blocks.

## Reference standard

Hayden's own blind codes, single coder, with the same fields, definitions and rules as the
October re-code (`data/annotations/recode_2026-10/CODING_INSTRUCTIONS.md`). There is no model
pass and no adjudication. Hayden has seen aggregate results but has never seen any label or
prediction for these decisions. He must not open `data/extracted/`, `outputs/` or other
annotation files while coding. The completed workbook is frozen (copied with date and sha1)
before scoring.

## Scoring (once, after the workbook is frozen)

Metric definitions are those of `validation_analysis.classification_validation`, unchanged.

**Primary** (weighted by stratum pool size, so they estimate performance on the corpus):
- misconduct macro-F1 and micro-F1, and per-class precision, recall and F1;
- is-disciplinary accuracy;
- finding accuracy;
- outcome regex micro-F1.

95% CIs come from 10,000 bootstrap resamples within strata.

**Secondary:**
1. **By period:** the same metrics in each of the four strata (unweighted, with bootstrap CIs).
2. **Combined author-reference set:** the first 45 test decisions (author codes) plus these 160,
   unweighted per-class F1 with bootstrap CIs, n = 205.
3. **Temporal stability, for each misconduct class:**
   - Among decisions the author coded positive, a logistic regression of detection (true
     positive vs false negative) on decision year, giving an OR per decade with 95% CI.
   - Among decisions the author coded negative, the same for false positives.
   - If any class has fewer than 10 positives, that class is reported descriptively only.
4. **Trend check:** in each stratum, the classifier's predicted prevalence versus the author-coded
   prevalence, for each misconduct class.
   - The FRAUD and COMM trends are reported as robust if the author-coded prevalence moves in the
     same direction as the predicted prevalence across the strata, and recall and the false
     positive rate show no clear time dependence (the decade OR CI includes 1).
   - Otherwise the manuscript says that the trend may partly reflect changing classifier error.
   - With about 40 decisions per stratum this check has low power. A null result is reported as
     "no evidence of drift", never as proof of stability.
5. **Boundary subtypes:** among decisions the author coded BOUND, the precision and recall of
   the keyword subcategories (sexual, emotional, financial) in `boundary_analysis` against the
   author's BOUND_SEXUAL / BOUND_EMOTIONAL / BOUND_FINANCIAL codes.

**Error-adjusted prevalence:**
- The Rogan-Gladen adjustment is re-estimated using sensitivity and specificity from the weighted
  160. This sample represents the corpus and the first test set does not.
- The manuscript reports this as the primary adjusted estimate and keeps the first-test-set
  version in the supplement.

## What does not change

- No model, threshold, vectoriser or training set is changed in response to these results. If
  performance is lower than on the first test set, the paper reports that.
- The first test set's results stay in the paper as the protocol-specified evaluation.

## Amendments

- **2026-11-06, before any decision was coded or scored.** Hayden approved a correction to the
  count-based trend in `temporal_trends`: it now uses complete calendar years only (2026 is
  dropped and years with no decisions count as zero). This changed the file's sha1 from
  5a8af73653c2 to a78b5cc73458. `boundary_analysis`, the only part this protocol uses, is
  byte-identical before and after (its source sha1 is 69c2609ae87b both times). The scoring
  script checks the function hash as well as the file hash.
- **2026-11-06, after coding was returned and before the workbook was frozen or anything was
  scored.** Implementation details written into `src/analysis/score_test2.py`. These are not
  changes of substance:
  - "Moves in the same direction" in the trend check means the least-squares slope of prevalence
    across the four strata (ordered by period) has the same sign for predicted and author-coded
    prevalence.
  - Predicted prevalence is taken among decisions predicted disciplinary; author-coded prevalence
    is taken among decisions the author coded disciplinary. This matches the main trend analysis,
    which uses proportions of disciplinary decisions.
  - Sensitivity and specificity for the Rogan-Gladen adjustment use all 160 decisions, weighted,
    with the same definition as `revision_analyses.error_adjusted_prevalence`.
  - The temporal-stability logistic regressions are unweighted. Sampling depended only on year,
    which is the covariate.
  - Bootstrap: 10,000 resamples within strata for the weighted primary metrics, and plain resamples
    for each stratum and for the combined n = 205 set (seed 20261106).
- **2026-11-07, deviation: the reference standard was re-coded. Written after the re-coded
  workbook was frozen and before it was scored.**
  - What happened: the first workbook (frozen 2026-11-06, sha1 58be9274…) was scored once under
    this protocol (weighted misconduct macro-F1 0.43). A diagnostic then found that many rows did
    not describe the decision in that row's text file. Profession agreed with the logged coder on
    106/160 (October re-code 288/300), and the notes best matched their own text on 55/160
    (October 238/300). That reference was judged invalid and those results were suspended.
  - What was done: at Hayden's request, the first attempt was set aside unchanged
    (`data/annotations/test2_2026-11/_superseded_first_attempt_2026-11-06/`; its outputs are in
    `outputs/_superseded_test2_first_attempt_2026-11-06/`). The same 160 decisions, in the same order
    and with the same texts, were re-coded by Hayden on 2026-11-06 with a fresh workbook. The
    workbook adds a `text_check` field and declarations that each row was coded only from its own
    text file and without consulting the first attempt. Frozen copy:
    `P13_test2_results_SUBMITTED_2026-11-07.xlsx`, sha1 b42fd27e8e3756ee1beae2a0f0033c8ab0210a6a.
  - Checks before scoring (none compares misconduct or outcome codes with any model):
    `text_check` found in its own text 160/160; profession agrees with the logged coder 156/160;
    notes best match their own text 110/160.
  - Disclosure: when re-coding, Hayden had seen the aggregate first-attempt results (the 0.43 and
    0.47 macro-F1) but no per-decision prediction. Everything else in this protocol (models, pins,
    metrics, weights, bootstrap, secondary analyses) is unchanged and is applied once to the new
    reference. The first-attempt results are reported only as the reason for this deviation.
