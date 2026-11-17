# Gold-standard re-code: coding instructions (October 2026)

## What this is and why

You are re-coding all 300 decisions in the gold-standard sample from scratch. The original "human" annotations were partly produced by AI coding agents, so this pass replaces them. Your codes become the new gold standard. The classifier is retrained on them, and every agreement statistic is recomputed from them.

For the paper to describe this as the author's own blinded coding, two things must hold:

1. **You make every decision yourself.** Do not use Claude, ChatGPT, Label Studio pre-annotations, scripts or any other assistance to suggest or fill codes.
2. **You stay blind to earlier labels.** While coding, do not open:
   - anything else in `data/annotations/`, other than this folder;
   - `data/extracted/` (the predictions and enriched corpus files);
   - `outputs/`;
   - the manuscript's results tables.

   Each text file in `texts/` contains only the decision text and neutral metadata (citation, tribunal, date, URL), so nothing in this folder shows an earlier label.

If either rule is broken for any item, say so in that row's `notes` and on the Protocol sheet. A disclosed exception is fine. An undisclosed one is the problem this re-code exists to fix.

## Files

| File | Use |
|---|---|
| `P13_recode_results.xlsx` | The results file. Fill the **Protocol** sheet first, then the **Coding** sheet. |
| `texts/HPM-001.txt` … `HPM-300.txt` | One decision per file, in the same order as the Coding sheet. |
| `../../../manuscript/supplementary/annotation_guidelines.md` | The full definitions (sections 3–8) and worked examples (section 9). |
| `../../../manuscript/supplementary/annotation_quickstart.md` | One-page summary of the codes. |

The row order was randomised (seed 20261017). Don't re-sort the Coding sheet. Grey columns are fixed. Fill only the white columns. Save often; you can stop and resume at any row.

## Method (per decision)

Read the catchwords, then the orders, then the reasons for decision if you still need them. The quickstart's "60-second method" usually works. If a decision needs more than about 5 minutes, code your best judgement, set confidence to Low and write a short note.

## Fields

### is_disciplinary (Y / N)
- **Y:** a proceeding that decides whether the practitioner's conduct, performance or health warrants findings or sanctions under the National Law or its predecessor Acts. This includes appeals against such decisions and health or impairment proceedings (guidelines 8.5).
- **N:** a decision that is not that kind of proceeding. Examples are registration appeals, reviews of immediate action, and purely procedural rulings (stays, adjournments, costs-only, suppression orders). It also covers a matter that turns out not to concern a registered health practitioner.

  If the case is borderline, choose one and explain in `notes`.

**If N:** set `finding` = NA and leave every misconduct, boundary-subtype and outcome column blank.

### finding (one)
`PM` (professional misconduct) · `UPC_SAT` (unsatisfactory professional conduct or performance) · `UPC_GEN` (unprofessional conduct) · `NO_FINDING` · `NA` (only when is_disciplinary = N).

If the decision makes more than one finding, code the most serious. Health or impairment proceedings with no conduct finding are `NO_FINDING` (guidelines 8.5).

### Misconduct type: CLIN, PRESC, BOUND, IMPR, FRAUD, COMM, PROF (choose Y for all that apply)
Code the conduct the decision deals with. Use the definitions in guidelines section 3. Three rules from the published methods:
- **BOUND** is reserved for breaches of the practitioner–patient relationship. Criminal conduct outside practice is **PROF**.
- Pharmacist dispensing or supply misconduct is **PRESC**, not CLIN.
- Breach of registration conditions is **PROF**.

If you disagree with one of these rules for a specific case, code what you think is right and say so in `notes`.

### Boundary subtype: BOUND_SEXUAL, BOUND_EMOTIONAL, BOUND_FINANCIAL (Y; only when BOUND = Y)
Tick each kind of boundary breach the decision actually describes. Statutory text, quoted case law and boilerplate do not count. These codes validate the paper's keyword-based subcategorisation, so code what happened, not what is mentioned.

### Outcomes: CANCEL, SUSPEND, CONDITIONS, REPRIMAND, FINE, CAUTION, UNDERTAKING, NO_ADVERSE, OTHER (Y for all ordered)
Code the orders made in this decision (guidelines section 5):
- For an appeal, code the appeal decision's orders.
- Disqualification of an unregistered person, costs-only orders, referrals and non-standard orders are `OTHER`.
- A dismissal on all grounds is `NO_ADVERSE`.
- For interim orders, code the order and write "interim" in `notes`.

### profession (one)
Take it from the decision text. Choose `unknown` if the text does not say. Don't infer it from the practitioner's name or title alone, unless the decision itself relies on that.

### confidence
`High`: clear. `Medium`: some judgement needed. `Low`: genuinely uncertain.

### notes
Optional. Use it for: appeal, consent orders, interim orders, multiple practitioners, truncated text, any departure from the rules above, and any breach of the blinding rules.

### date_coded
The date you coded the row (YYYY-MM-DD).

## When you finish

1. Complete the Protocol sheet (coder, dates, both declarations).
2. Run the completeness check from the project root:
   ```
   python3 src/preprocessing/check_recode.py
   ```
   It lists any blank or inconsistent rows. It does not judge your codes.
3. Tell Claude the file is complete. The next steps are scripted:
   - Recompute agreement with the genuinely API-produced model annotations only.
   - Build a separate adjudication sheet, where you review disagreements with your own code visible. That step is recorded as unblinded.
   - Retrain the classifier and re-run the analyses.
