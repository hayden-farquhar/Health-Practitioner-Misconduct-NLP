# Retrain protocol (fixed 2026-11-05, before any model was trained)

Approved by Hayden Farquhar in chat on 2026-11-05. Written before training so that every choice
below is fixed in advance. Any departure is reported as a departure.

## Why

The February 2026 classifier was chosen from seven training-set variants by its test-set score
(`outputs/statistics/filtering_comparison.json`: the shipped `rare_plus_nd` variant had the
second-lowest validation macro-F1, 0.635, and the highest test macro-F1, 0.682). The finding
classifier's "hybrid" vectoriser was chosen the same way. The reported test metrics were
therefore optimistic. This protocol replaces that process.

## Labels

- Gold standard: `data/annotations/gold_standard_2026-11.json` (author's blind re-code, logged
  model pass, author's adjudication, one dated amendment; see its manifest).
- Split membership is kept exactly as in February (train 210, validation 45, test 45), read from
  `data/annotations/splits/split_ids.json`, which is written from the February split files before
  any label changes. No re-stratification.
- Supplementary training decisions: the 300 decisions in `data/annotations/new_sample_300.jsonl`
  (disjoint from the gold 300). They are re-labelled by the same logged model pass used as second
  coder (`src/annotation/model_second_coder.py`: same model, effort, system prompt and output
  schema; Batches API). Their February labels (`new_batch_01-12.json`, unlogged agent runs under
  the old rules) are not used. A decision the model refuses or fails to code is left out and
  counted.

## Training sets (two, fixed)

- A: the 210 gold training decisions.
- B: A plus every successfully coded supplementary decision.

The February filter variants (rare_only, disc_only, rare_plus_nd and the rest) are dropped.

## Models

As February: TF-IDF (max_features 10000, ngram (1,2), sublinear_tf, min_df 2, max_df 0.95,
strip_accents unicode) fitted on the training set; logistic regression (lbfgs, max_iter 2000,
class_weight balanced, random_state 42); C grid [0.5, 1, 5, 10, 50, 100].

- Misconduct: one-vs-rest over the 8 labels (7 categories plus N/A).
- Finding: multiclass over PM, UPC_SAT, UPC_GEN, NO_FINDING, NONE (non-disciplinary).
- Is-disciplinary: binary.

## Selection (validation set only)

1. For each training set and task, C is the grid value with the highest validation score
   (misconduct: macro-F1 over the 8 labels; finding: macro-F1; is-disciplinary: binary F1).
   Ties go to the smaller C.
2. The misconduct training set is the one with the higher validation misconduct macro-F1.
   Is-disciplinary uses the same training set and vectoriser (the evaluation code applies one
   vectoriser to both).
3. The finding training set (and so its vectoriser) is chosen separately, by validation finding
   macro-F1.
4. Ties between training sets go to A (fewer machine-labelled decisions).
5. No refit on train plus validation.

The training script never loads test labels.

## Test (scored once)

The chosen models are scored once on the 45 test decisions by
`src/analysis/validation_analysis.py`, with its existing metric definitions (misconduct
macro-F1 over 8 labels; outcome regex over the 8 standard outcomes, OTHER excluded as in
February). The result is reported whatever it is. No further model is chosen after the test
score has been seen.

## Then

Corpus predictions, Phase 4 analyses, Phase 5 validation and the revision analyses are re-run
from the chosen models. The February test-set selection is disclosed in the medRxiv v2
correction note and in Methods.
