# Language-model coder protocol (fixed 2026-11-06, before the model saw any test2 decision)

Requested by Hayden Farquhar in chat on 2026-11-06, after the TF-IDF classifier scored a weighted
misconduct macro-F1 of 0.43 on the second test set (`outputs/statistics/test2_results.json`).
Everything below is fixed before the coder is run on the 160 test2 decisions. Any departure is
reported as a departure.

## Why

On a representative sample coded blind by the author, the TF-IDF classifier is too weak to carry
corpus-level findings. The logged language-model coder already exists and is fixed: it was the
second coder for the gold standard and the labeller of the supplementary training set. It has
never been run on the 160 test2 decisions, so these decisions are a clean held-out test of it.

## The coder (unchanged from the gold-standard and supplementary passes)

- Script: `src/annotation/model_second_coder.py`. The only change allowed is adding a `test2`
  set (texts `data/annotations/test2_2026-11/texts`, 160 items, prefix `T2`) and, if the corpus
  run goes ahead, a `corpus` set. No change to the prompt, the schema or the request parameters.
- Model `claude-sonnet-5-5`, effort `medium`, max_tokens 16,000, structured JSON output with the
  logged schema (`data/annotations/model_pass_2026-11/output_schema.json`). The request sets no
  sampling parameters. Message Batches API.
- System prompt sha256 `9f05c736ccd257afe8bcf4665f1416d357baa25b5370ac7741ced2a6ba322866`, the
  same as both earlier passes; it was rebuilt and checked on 2026-11-06. The submit step stops if
  the hash differs.
- Every request and response is logged, and no output is edited. Refusals, errors and truncated
  outputs are counted and kept. There is no retry with a different model or prompt.
- One run only. If the batch fails as a whole (e.g. an API outage), it may be resubmitted
  unchanged, and that is recorded.

## Scoring on test2

The reference is the frozen author codes
(`P13_test2_results_SUBMITTED_2026-11-06.xlsx`, sha1 58be9274…).

- Metric definitions, weights and the bootstrap are exactly those of `src/analysis/score_test2.py`.
  Misconduct macro-F1 is over the 7 categories plus N/A. Finding uses NONE for non-disciplinary
  decisions. Outcome micro-F1 is over the 8 scored codes. Everything is weighted to the pool,
  with 10,000 within-stratum bootstrap resamples and seed 20261106.
- **Primary handling of failures:** a refused, errored or unparseable item is scored as
  "non-disciplinary, no codes", so failures count against the coder. A sensitivity analysis
  excludes them.
- **Head-to-head:** the paired bootstrap difference (coder minus TF-IDF) in weighted misconduct
  macro-F1, using the same resamples.
- The secondary analyses of the TF-IDF protocol are also run for the coder: by period, the
  temporal-stability logits, the trend check, and Rogan-Gladen. Boundary subtypes are scored
  using the coder's own `boundary_subtypes` against the author's subtype codes, which replaces the
  keyword method.

## Decision rule (fixed now)

The coder **passes** if both of the following hold on the weighted primary analysis:

1. misconduct macro-F1 ≥ 0.70 and is-disciplinary accuracy ≥ 0.90;
2. the paired difference from TF-IDF in misconduct macro-F1 has a 95% CI that excludes 0.

On a pass, the coder's codes replace the TF-IDF classifier and the regex outcome extractor for
the whole corpus, and every downstream analysis is re-run from them:

- **Corpus labels.** The coder codes all 3,586 decisions in one logged batch (the `corpus` set).
  Those outputs, unedited, are the corpus labels for every decision, including the decisions
  that also have human codes. Human codes are never substituted, so every corpus estimate comes
  from one instrument. Human codes are used only for validation.
- **Weak categories.** A misconduct category whose weighted per-class F1 is below 0.60 is
  reported as unreliable: its prevalence and trend are given only with that caveat and are not
  used for headline claims.
- **Cost check.** Before submitting the corpus batch, the cost is estimated with the script's
  `count` command and confirmed with Hayden.
- **The TF-IDF classifier** stays in the paper as a reported comparison, including its test2
  result and the first test set's overstatement.

**If the coder fails either condition:** it is reported in full, and the paper switches to the
fallback. The fallback treats the author-coded decisions (test2 n = 160, weighted, plus the
gold standard with its sampling noted) as the main prevalence evidence, and the classifier and
coder corpus results as exploratory. No further method is tried on test2, because it has now
been used.

## What this test cannot show

- Agreement with one author's codes, made under the same guidelines that form the coder's
  prompt, measures fidelity to those guidelines, not legal correctness.
- Test2 has 160 decisions, and IMPR and COMM are rare in it (11 and 2 author-coded positives),
  so their per-class estimates are imprecise whatever the result.

## Amendments

- **2026-11-07, deviation: the reference standard was re-coded. Written after the re-coded
  workbook was frozen and before it was scored.** The first scoring (weighted macro-F1 0.470;
  fail) used a reference that was later found invalid, because many rows did not describe their
  own text (see the 2026-11-07 amendment to `config/external_test_protocol_2026-11.md`). Hayden
  re-coded the same 160 decisions. The reference is now
  `P13_test2_results_SUBMITTED_2026-11-07.xlsx` (sha1 b42fd27e8e3756ee1beae2a0f0033c8ab0210a6a).
  - The coder is not re-run. Its logged outputs (`model_pass/`, batch as recorded in
    `batch_manifest.json`) were produced before either coding attempt was scored, and are scored
    once against the new reference.
  - The metrics, the failure handling, the pass rule and the consequences of a pass or a fail are
    unchanged.
  - "No further method is tried on test2" still holds: this re-scores the same coder outputs
    against a corrected reference; it is not a new method.
  - The first scoring is reported only as the reason for this deviation.
- **2026-11-07: Hayden approved the fallback, and the steps below were fixed before the logged
  run.** The coder failed the pass rule, so the paper switches to the fallback. Hayden approved
  this in chat on 2026-11-07, and also approved reframing the paper around the human-coded
  estimates.
  - Disclosure: when advising on the reframe, Claude ran a preliminary, unlogged calculation with
    the estimator described below, and reported it to Hayden. It covered the weighted prevalence of
    the seven misconduct types, cancellation and reprimand. Nothing else was looked at.
  - **Primary prevalence (human-coded).** Use the 160 re-coded test2 decisions. Estimates are
    weighted ratio estimates, with stratum weight = pool size / 40, and give proportions of the
    decisions the author coded disciplinary. They cover:
    - the share of decisions that are disciplinary;
    - the seven misconduct types;
    - the boundary subtypes, among decisions coded BOUND;
    - finding and the nine outcomes (OTHER included);
    - profession (medical, nurse/midwife, other).

    95% CIs come from 10,000 bootstrap resamples within strata (seed 20261106).
  - **What these estimates describe.** They describe the sampled pool: dated decisions of at least
    500 characters never previously labelled (2,978 of 3,586). The 300 gold-standard and 300
    supplementary decisions were sampled by profession and are not used for prevalence.
  - **Human-coded change over time.** For each misconduct type, give the unweighted prevalence
    among disciplinary decisions in each period stratum, with bootstrap CIs. Add a logistic
    regression of the type on decision year among author-coded disciplinary decisions, giving an
    OR per decade with 95% CI. It is unweighted because sampling depended only on year. With
    about 27 disciplinary decisions per stratum, a null result is reported as "no evidence of
    change", never as stability.
  - **Corpus-wide classifier analyses.** These are kept and labelled exploratory: counts, Mann-Kendall
    trends, associations, jurisdiction and profession comparisons, prescribing, and rates per
    registrant. Each is reported next to the classifier's weighted test2 performance for the
    categories it uses. The following are reported as unreliable:
    - categories with test2 per-class F1 < 0.60 (COMM, IMPR);
    - the financial boundary subtype (keyword precision 0.21);
    - the FRAUD trend (detection drifts over time).

    Corpus counts and the TOTAL trend depend only on is-disciplinary classification (test2
    accuracy 0.977), and are reported with that accuracy. The Rogan-Gladen estimates from test2
    are a sensitivity analysis, not the primary estimates.
