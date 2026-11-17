# Independent second-coder protocol (fixed 2026-11-07, before recruitment)

Requested by Hayden Farquhar on 2026-11-07, after the manuscript evaluation identified the
single-coder reference as the main remaining weakness. Hayden chose Prolific as the
recruitment channel (it is the only one available to him), a 60-decision subset, and a law-or-
health postgraduate filter. Everything below is fixed before any coder is recruited. Any
departure is reported as a departure.

## Purpose and use

The aim is to measure agreement between the author's frozen re-code of the coded sample
(`data/annotations/test2_2026-11/P13_test2_results_SUBMITTED_2026-11-07.xlsx`, sha1
b42fd27e…) and one independent, blinded human coder.

- The second coder's codes are used **only to report reliability**: per-field agreement, and
  a sensitivity estimate of prevalence on the subset.
- They do **not** replace, adjust or adjudicate the author's codes. The primary estimates in
  the manuscript are unchanged whatever the agreement turns out to be.
- No result is re-selected or re-scored on the basis of the second coder's codes.

## Subset

- **60 of the 160 coded decisions:** 15 per period stratum (1999–2009, 2010–2014, 2015–2019,
  2020–2026), drawn by simple random sampling within strata, seed 20261109. They are shuffled
  and split into two sessions of 30.
- Every period is represented in each session. Documents are relabelled D01–D60, so the
  author's item numbers are not shown.
- Rare categories may have too few coded positives in 60 decisions for kappa to be
  informative. For any field with fewer than 5 positives from either coder, agreement is
  reported descriptively, without kappa.

## Coder recruitment (Prolific)

Filters:
- approval rate ≥ 99%;
- ≥ 100 previous submissions;
- highest education a graduate or doctoral degree;
- degree subject Law, Medicine/Health, Nursing, Pharmacology, Dentistry or Psychology;
- fluent English;
- desktop only.

The process:
1. **Screener:** 5 places. Each candidate reads the coding guide and codes **4 practice
   decisions**. These come from the gold-standard training set and are never part of the 160.
2. **Selection:** one coder is chosen on **objective, procedural criteria only**:
   - a complete, well-formed export;
   - the name check matches each practice text;
   - disciplinary status correct on all 4 practice decisions (each is unambiguous: one stay
     application, three disciplinary decisions with stated orders);
   - the orders that are explicitly stated (reprimand on the three disciplinary decisions;
     conditions on one) are recorded;
   - at least one note with decision-specific content;
   - a median time per decision of at least 4 minutes.

   If several candidates meet every criterion, the earliest complete submission is chosen.
   Selection never uses agreement with the author's codes on judgement-call fields: that
   would select for coders who resemble the author.
3. **Coding:** the selected coder codes the 60 decisions in two sessions (30 each), with at
   least an overnight gap between them. Session 2 is open only to that coder.
4. **Fallback:** if no candidate meets every criterion, the screener may be re-run once with
   5 new places. If that also fails, the result is reported and no second coder is used.

**Payment:** about £12 per hour.
- Screener: £12 for an estimated 60 minutes.
- Each session: £48 for an estimated 4 hours.
- A bonus may be paid where a coder's logged time substantially exceeds the estimate, so that
  the effective rate stays fair.
- Planning cost, including Prolific fees (multiplier about ×1.40, measured on a previous study): about
  £220.

## Instruments and blinding

- The coder uses the same fields, definitions and rules as the author:
  - the annotation guidelines, sections 2–9;
  - the field rules of the test2 coding instructions;
  - the same text files.

  All of these are rendered in an in-browser workbook that saves automatically and exports
  a CSV. Only the framing text differs: the author-facing references to protocols,
  classifiers and project files are removed. Each such difference is logged in
  `data/annotations/second_coder_2026-11/DIFF_NOTES.md`.
- The coder never sees the author's codes, any machine output, the manuscript, or any
  result.
- The URL line of each text is removed, and the coder is told not to look decisions up
  online.
- Rules given to the coder:
  - work alone;
  - no AI tools of any kind;
  - code only from the text shown;
  - use the same browser and device throughout.
- The coder attests to these at export.

## Provenance

- Each returned CSV is frozen on receipt (dated, read-only copy with sha256) before any
  comparison.
- Claude may stage drafts, build the workbook, freeze files and compute agreement from
  returned data only.
- Claude must never create, edit or impute a coder's codes.
- Nothing is published on Prolific without Hayden's explicit confirmation.
- Coders are compensated contributors, not research participants. No personal data beyond
  the pseudonymous Prolific ID is collected, and IDs are not published. The coder may choose
  named acknowledgement or anonymity.

## Analysis (fixed now)

Unit: the 60 decisions.

1. **Cohen's kappa with 95% CI** (2,000 bootstrap resamples, seed 20261109), plus raw
   agreement, for:
   - disciplinary status;
   - finding;
   - each of the 7 misconduct types;
   - each of the 3 boundary subtypes (among decisions both coders coded BOUND);
   - each of the 9 outcomes;
   - profession.

   As in the author's own coding, a field ticked on a non-disciplinary decision is treated
   as absent.
2. **Prevalence on the subset** of each misconduct type among decisions each coder coded
   disciplinary, for both coders. These are unweighted, with bootstrap CIs. The paired
   difference is reported as a sensitivity analysis of the primary estimates.
3. **Disagreements** are listed by field and classified descriptively (definitional,
   reading, or scope of the decision). They are not adjudicated, and the author's codes are
   not changed.

## Reporting

The manuscript reports:
- the recruitment channel and filters;
- the selection criteria (including the rule against selection on judgement-call agreement);
- payment;
- blinding;
- the attestation;
- the freeze hashes;
- kappa by field;
- the subset prevalence comparison.

It reports whatever the result is. If recruitment fails, it says so.
