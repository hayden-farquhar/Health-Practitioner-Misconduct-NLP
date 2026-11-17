# Annotation Guidelines: Health Practitioner Misconduct Decisions

**Version:** 1.0
**Date:** 2026-02-12
**Project:** NLP Analysis of Australian Health Practitioner Tribunal Decisions

---

## 1. Overview

This document provides coding instructions for annotating Australian health practitioner disciplinary tribunal decisions. Each decision is coded for:

1. **Misconduct type(s)** — multi-label, from a two-level taxonomy (Level-1 required, Level-2 optional)
2. **Finding** — single-label (the most serious finding made)
3. **Outcome(s)** — multi-label (all sanctions/orders made)
4. **Profession** — single-label (the practitioner's registered health profession)
5. **Confidence** — per-label rating (high / medium / low)

Refer to `config/taxonomy.json` for the full taxonomy with definitions and examples.

---

## 2. General Principles

### 2.1 Unit of Analysis
Each **decision document** is one annotation unit. A single decision may involve multiple charges, multiple findings, and multiple sanctions.

### 2.2 Multi-label Coding
A decision may be coded with **multiple misconduct categories**. For example, a doctor who prescribes opioids excessively while in a sexual relationship with a patient would be coded as both PRESC (Prescribing misconduct) and BOUND (Boundary violations).

### 2.3 Coding from Text
Code based on **what is described in the decision**, not inferences beyond the text. If the tribunal does not discuss a particular type of misconduct, do not code it even if it seems likely.

### 2.4 Prioritise the Findings/Orders Sections
The **Findings** and **Orders** sections of the decision contain the most reliable information for coding. Where the evidence section describes allegations that were ultimately not sustained, do not code the unsustained allegations.

### 2.5 When in Doubt
- If genuinely uncertain, code the category and mark confidence as **low**.
- Record any reasoning in the **Notes** field.
- Do **not** leave a decision uncoded unless it is genuinely impossible to determine any misconduct type (e.g., procedural-only decision with no substantive findings).

---

## 3. Misconduct Type Coding

### 3.1 Level-1 Categories (Required)

| Code | Category | Key Indicators |
|------|----------|----------------|
| CLIN | Clinical competence | Diagnostic error, treatment failure, inadequate assessment, poor record-keeping, infection control |
| PRESC | Prescribing/dispensing misconduct | Drug prescribing or dispensing issues, medication management, self-prescribing, supply irregularities |
| BOUND | Boundary violations | Sexual, emotional, financial, or social boundary breaches with patients |
| IMPR | Impairment | Substance use, mental health, physical health affecting practice |
| FRAUD | Dishonesty and fraud | Billing fraud, falsified records, misrepresented qualifications |
| COMM | Communication failures | Consent failures, inadequate disclosure, mandatory notification failures |
| PROF | Professional conduct | Breach of conditions, advertising, insurance, supervision, other conduct |

### 3.2 Level-2 Subcategories (Optional but Encouraged)

Level-2 codes provide more specific classification within each Level-1 category. Code the most specific applicable subcategory where possible. See `config/taxonomy.json` for full definitions.

### 3.3 Decision Rules for Misconduct Coding

**Rule 1: Code all applicable categories.**
If a decision involves both prescribing misconduct and clinical incompetence, code both PRESC and CLIN.

**Rule 2: Distinguish root cause from consequence.**
If a practitioner's substance use (IMPR) led to clinical errors (CLIN), code **both** categories. The impairment is the root cause; the clinical errors are the manifestation.

**Rule 3: Prescribing vs. Clinical competence.**
- If the core issue is *what* was prescribed (wrong drug, wrong dose, controlled substance abuse) -- PRESC
- If the core issue is *clinical assessment/diagnosis* that led to prescribing -- CLIN
- If both -- code both

**Rule 4: Boundary violations require a patient relationship.**
A boundary violation requires a practitioner-patient relationship (current or recent). Misconduct involving non-patients (e.g., colleagues, general public) is coded under PROF unless it involves a former patient within the relevant cooling-off period.

**Rule 5: Dishonesty vs. poor practice.**
- Deliberate falsification of records -- FRAUD
- Inadequate or incomplete records due to poor practice -- CLIN_RECORD
- The key distinguishing factor is **intent**: was the record failure deliberate?

**Rule 6: Breach of conditions.**
Breach of conditions on registration is coded as PROF_CONDITIONS, regardless of what the original conditions were imposed for. However, also code the underlying misconduct if it is described (e.g., if the practitioner breached conditions by prescribing outside scope, code both PROF_CONDITIONS and PRESC).

---

## 4. Finding Coding

### 4.1 Categories

| Code | Finding | Description |
|------|---------|-------------|
| PM | Professional misconduct | Most serious; conduct substantially below expected standard |
| UPC_SAT | Unsatisfactory professional conduct | Below expected standard but not amounting to professional misconduct |
| UPC_GEN | Unprofessional conduct | Lesser standard than expected by public or professional peers |
| NO_FINDING | No finding | Complaint dismissed, charges not established |

### 4.2 Decision Rules

**Rule 1: Code the most serious finding.**
If a decision makes both a finding of professional misconduct (on some charges) and unsatisfactory professional conduct (on others), code **PM** as the primary finding.

**Rule 2: Jurisdictional terminology varies.**
- NSW uses "unsatisfactory professional conduct" and "professional misconduct"
- Victoria uses "unprofessional conduct" and "professional misconduct"
- Queensland uses "professional misconduct" and "unsatisfactory professional conduct"
- Other jurisdictions may use slightly different terminology
- Map to the closest matching category based on the statutory definition cited in the decision.

**Rule 3: Consent orders.**
If a matter is resolved by consent without a formal finding, code based on:
- If the consent orders include an acknowledgment of professional misconduct -- PM
- If no finding is explicitly made but sanctions are imposed -- code as UPC_GEN (default)
- If the matter is truly dismissed by consent -- NO_FINDING

---

## 5. Outcome Coding

### 5.1 Categories (in order of severity)

| Rank | Code | Outcome |
|------|------|---------|
| 1 | CANCEL | Cancellation of registration |
| 2 | SUSPEND | Suspension of registration |
| 3 | CONDITIONS | Conditions on registration |
| 4 | REPRIMAND | Reprimand or censure |
| 5 | FINE | Monetary penalty |
| 6 | CAUTION | Formal caution |
| 7 | UNDERTAKING | Undertaking given by practitioner |
| 8 | OTHER | Other (costs only, referral, non-standard) |
| 9 | NO_ADVERSE | No adverse finding / complaint dismissed |

### 5.2 Decision Rules

**Rule 1: Code all outcomes ordered.**
A decision that cancels registration AND imposes a fine is coded as both CANCEL and FINE. The **primary outcome** is the most severe (CANCEL in this case).

**Rule 2: Suspension with conditions.**
If a practitioner is suspended and then has conditions imposed upon return, code both SUSPEND and CONDITIONS.

**Rule 3: Non-review period.**
If registration is cancelled with a specified non-review period (e.g., "not to apply for review before 3 years"), this is still coded as CANCEL. Note the non-review period in the Notes field.

**Rule 4: Voluntary surrender.**
If a practitioner voluntarily surrenders registration during proceedings, code as CANCEL (the practical effect is the same).

**Rule 5: Costs orders.**
Costs orders alone (without substantive sanctions) are coded as OTHER. If costs are ordered alongside a substantive sanction, code the substantive sanction(s) only.

---

## 6. Profession Coding

### 6.1 Categories
Code the practitioner's registered profession using the AHPRA profession list:

- medical_practitioner (includes GPs, specialists, surgeons)
- nurse (includes registered nurse, enrolled nurse, nurse practitioner)
- midwife
- dentist (includes dental hygienist, dental therapist, dental prosthetist, oral health therapist)
- pharmacist
- psychologist
- physiotherapist
- chiropractor
- osteopath
- optometrist
- podiatrist
- paramedic
- occupational_therapist
- chinese_medicine_practitioner
- medical_radiation_practitioner
- aboriginal_health_practitioner

### 6.2 Decision Rules

**Rule 1: Dual registration.**
If a practitioner holds dual registration (e.g., nurse and midwife), code the profession that the disciplinary action primarily relates to.

**Rule 2: Specialist titles.**
Code specialists by their base profession (e.g., a psychiatrist is coded as medical_practitioner, a specialist pharmacist is coded as pharmacist).

---

## 7. Confidence Rating

For each coded label, assign a confidence rating:

| Rating | Definition | When to Use |
|--------|------------|-------------|
| **High** | Clear and unambiguous from the text | The decision explicitly describes this type of misconduct/outcome |
| **Medium** | Probable but some ambiguity | The misconduct type is implied or the categorisation involves judgement |
| **Low** | Uncertain but possible | The text is ambiguous, the decision is unclear, or the category is borderline |

---

## 8. Edge Cases

### 8.1 Appeals
- Code the **appeal decision's outcome**, not the original tribunal decision.
- Note in the Notes field: "Appeal decision — original outcome was [X]."
- If the appeal upholds the original decision, code the same outcomes.

### 8.2 Interim/Interlocutory Orders
- If the decision is purely about interim orders (e.g., interim suspension pending hearing), note this in the Notes field.
- Code the interim outcome (e.g., SUSPEND) but mark it as interim in Notes.

### 8.3 Multiple Practitioners
- If a single decision involves multiple practitioners, code for the **primary practitioner** (typically the respondent named first or most prominently).
- Note in Notes: "Decision involves multiple practitioners."

### 8.4 Procedural Decisions
- Decisions that are purely procedural (e.g., adjournment, stay of proceedings, costs) without substantive findings should be coded as NO_FINDING with outcome OTHER.
- Note in Notes: "Procedural decision only."

### 8.5 Impairment Proceedings (Non-Disciplinary)
- Some decisions concern impairment without misconduct findings (e.g., health review under National Law).
- Code finding as NO_FINDING (since no misconduct finding is made).
- Code misconduct as IMPR (since the underlying issue is impairment).
- Code outcome based on orders made (typically CONDITIONS).

---

## 9. Worked Examples

### Example 1: Prescribing Misconduct with Impairment

> *The Medical Board of Australia brought proceedings against Dr X, alleging that Dr X prescribed excessive quantities of oxycodone to multiple patients without adequate clinical justification. The tribunal also found that Dr X had been self-prescribing benzodiazepines and had a substance use disorder. Dr X's registration was cancelled.*

**Coding:**
- Misconduct: **PRESC** (PRESC_OPIOID, PRESC_SELF), **IMPR** (IMPR_SUBSTANCE) — Confidence: High
- Finding: **PM** (Professional misconduct) — Confidence: High
- Outcome: **CANCEL** — Confidence: High
- Profession: medical_practitioner

### Example 2: Boundary Violation

> *The HCCC commenced proceedings against a psychologist who entered into a sexual relationship with a patient during the course of treatment. The practitioner admitted the conduct. The tribunal found professional misconduct and ordered a 12-month suspension followed by conditions requiring supervision.*

**Coding:**
- Misconduct: **BOUND** (BOUND_SEXUAL) — Confidence: High
- Finding: **PM** — Confidence: High
- Outcome: **SUSPEND**, **CONDITIONS** — Confidence: High
- Profession: psychologist

### Example 3: Clinical Competence with Communication Failure

> *VCAT heard a complaint regarding a general practitioner who failed to diagnose a cardiac condition despite the patient presenting with classic symptoms over multiple visits. The practitioner did not explain the risks of the patient's condition or obtain consent for the chosen treatment plan. The tribunal found unsatisfactory professional conduct and imposed conditions on registration.*

**Coding:**
- Misconduct: **CLIN** (CLIN_DIAG), **COMM** (COMM_CONSENT, COMM_DISCLOSURE) — Confidence: High
- Finding: **UPC_SAT** — Confidence: High
- Outcome: **CONDITIONS** — Confidence: High
- Profession: medical_practitioner

### Example 4: Medicare Fraud

> *The Medical Board brought charges against a medical practitioner who billed Medicare for consultations that did not occur and altered patient records to support the fraudulent claims. The tribunal cancelled the practitioner's registration.*

**Coding:**
- Misconduct: **FRAUD** (FRAUD_BILLING, FRAUD_RECORDS) — Confidence: High
- Finding: **PM** — Confidence: High
- Outcome: **CANCEL** — Confidence: High
- Profession: medical_practitioner

### Example 5: Dismissed Complaint

> *The HCCC brought a complaint against a nurse alleging inadequate care of an elderly patient. After hearing evidence, the tribunal found that the nurse's care met the expected standard and dismissed the complaint.*

**Coding:**
- Misconduct: **CLIN** — Confidence: Low (alleged but not sustained; still code as the decision discusses clinical care)
- Finding: **NO_FINDING** — Confidence: High
- Outcome: **NO_ADVERSE** — Confidence: High
- Profession: nurse
- Notes: "Complaint dismissed — clinical competence alleged but not established."

### Example 6: Breach of Conditions

> *A chiropractor was found to have breached conditions on their registration by failing to attend required supervision sessions and practising outside the permitted scope. The original conditions had been imposed following a finding of clinical incompetence. The tribunal suspended the practitioner's registration for 6 months.*

**Coding:**
- Misconduct: **PROF** (PROF_CONDITIONS) — Confidence: High
- Finding: **PM** — Confidence: High (breach of conditions typically constitutes professional misconduct)
- Outcome: **SUSPEND** — Confidence: High
- Profession: chiropractor

---

## 10. Annotation Workflow

1. **Read** the full decision (or at minimum the Background, Findings, and Orders sections).
2. **Identify** the practitioner's profession.
3. **Code** the misconduct type(s) at Level-1 (and Level-2 if determinable).
4. **Code** the finding (most serious finding if multiple).
5. **Code** all outcomes ordered.
6. **Assign** confidence ratings.
7. **Add notes** for any edge cases, ambiguities, or decisions requiring discussion.
8. **Review** your coding against the decision rules before submitting.

---

## 11. Quality Assurance

- A random subset of 100 decisions will be independently coded by a second annotator for inter-rater reliability.
- Target: Cohen's kappa > 0.7 per Level-1 misconduct category.
- Disagreements will be resolved through discussion and consensus coding.
- Annotation guidelines will be updated if systematic disagreements reveal ambiguities.
