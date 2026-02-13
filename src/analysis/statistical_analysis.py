"""Phase 4: Statistical analysis of the enriched corpus.

Loads the enriched corpus and runs all statistical analyses for the
manuscript, producing publication-quality tables (CSV) and figures (PNG).

Usage:
    .venv-ml/bin/python -m src.analysis.statistical_analysis
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "extracted" / "enriched_corpus.jsonl"
ENTITIES_PATH = PROJECT_ROOT / "data" / "extracted" / "entities.jsonl"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"

# Ensure output dirs
for d in (FIGURES_DIR, TABLES_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("colorblind")
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.1,
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    },
)

MISCONDUCT_LABELS = {
    "CLIN": "Clinical competence",
    "PRESC": "Prescribing",
    "BOUND": "Boundary violations",
    "IMPR": "Impairment",
    "FRAUD": "Dishonesty/fraud",
    "COMM": "Communication",
    "PROF": "Professional conduct",
}

OUTCOME_LABELS = {
    "CANCEL": "Cancellation",
    "SUSPEND": "Suspension",
    "CONDITIONS": "Conditions",
    "REPRIMAND": "Reprimand",
    "FINE": "Fine",
    "CAUTION": "Caution",
    "UNDERTAKING": "Undertaking",
    "NO_ADVERSE": "No adverse finding",
}

FINDING_LABELS = {
    "PM": "Professional misconduct",
    "UPC_SAT": "Unsatisfactory professional conduct",
    "UPC_GEN": "Unprofessional conduct",
    "NO_FINDING": "No finding",
    "NONE": "No finding",
}

PROFESSION_LABELS = {
    "medical_practitioner": "Medical practitioner",
    "nurse": "Nurse",
    "psychologist": "Psychologist",
    "pharmacist": "Pharmacist",
    "dentist": "Dentist",
    "chiropractor": "Chiropractor",
    "physiotherapist": "Physiotherapist",
    "chinese_medicine_practitioner": "Chinese medicine",
    "osteopath": "Osteopath",
    "paramedic": "Paramedic",
    "podiatrist": "Podiatrist",
    "occupational_therapist": "Occupational therapist",
    "midwife": "Midwife",
    "optometrist": "Optometrist",
}

# Severity order for outcomes (most → least severe)
OUTCOME_ORDER = [
    "CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
    "FINE", "CAUTION", "UNDERTAKING", "NO_ADVERSE",
]

MISCONDUCT_ORDER = ["CLIN", "PRESC", "BOUND", "PROF", "FRAUD", "IMPR", "COMM"]

# Accumulator for statistical test results
chi_square_results = []
trend_results = []
summary_stats = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cramers_v(contingency_table: np.ndarray) -> float:
    """Cramér's V effect size from a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0


def safe_chi2(table: pd.DataFrame, test_name: str) -> dict:
    """Run chi-square on a crosstab, record results, return dict."""
    ct = table.values
    # Remove rows/cols that sum to 0
    ct = ct[ct.sum(axis=1) > 0][:, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"test": test_name, "skipped": True, "reason": "insufficient dimensions"}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = cramers_v(ct)
    result = {
        "test": test_name,
        "chi2": round(chi2, 2),
        "df": int(dof),
        "p_value": float(f"{p:.6g}"),
        "cramers_v": round(v, 3),
        "n": int(ct.sum()),
    }
    chi_square_results.append(result)
    return result


def format_profession(val):
    """Map raw profession to display label."""
    if pd.isna(val) or val is None:
        return "Unknown"
    return PROFESSION_LABELS.get(val, val.replace("_", " ").title())


def save_fig(fig, name: str):
    """Save figure and close."""
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {name}")


def save_table(df: pd.DataFrame, name: str):
    """Save table as CSV."""
    df.to_csv(TABLES_DIR / name)
    print(f"    Saved {name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus() -> pd.DataFrame:
    """Load enriched corpus into DataFrame with derived fields."""
    records = []
    with open(CORPUS_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Parse year
    df["year"] = pd.to_numeric(
        df["date"].str[:4], errors="coerce"
    ).astype("Int64")

    # Profession display
    df["profession_label"] = df["profession"].apply(format_profession)

    # Count of misconduct types (excluding N/A)
    df["misconduct_types"] = df["predicted_misconduct_L1"].apply(
        lambda x: [m for m in x if m != "N/A"]
    )
    df["n_misconduct_types"] = df["misconduct_types"].apply(len)
    df["has_multiple_misconduct"] = df["n_misconduct_types"] > 1

    # COVID periods
    df["covid_period"] = pd.cut(
        df["year"].astype(float),
        bins=[0, 2019.5, 2021.5, 3000],
        labels=["Pre-COVID (<2020)", "COVID (2020–2021)", "Post-COVID (2022+)"],
    )

    # Merge NO_FINDING and NONE for finding analysis
    df["finding_clean"] = df["predicted_finding"].replace("NONE", "NO_FINDING")

    # Boolean columns for each misconduct type
    for mc in MISCONDUCT_ORDER:
        df[f"mc_{mc}"] = df["misconduct_types"].apply(lambda x, m=mc: m in x)

    # Boolean columns for each outcome
    for oc in OUTCOME_ORDER:
        df[f"oc_{oc}"] = df["predicted_outcomes"].apply(lambda x, o=oc: o in x)

    # Most severe outcome
    def most_severe(outcomes):
        for o in OUTCOME_ORDER:
            if o in outcomes:
                return o
        return "NONE"
    df["most_severe_outcome"] = df["predicted_outcomes"].apply(most_severe)

    print(f"  Loaded {len(df)} records, {df['predicted_is_disciplinary'].sum()} disciplinary")
    return df


# ---------------------------------------------------------------------------
# 4.1 Descriptive Statistics
# ---------------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame, disc: pd.DataFrame):
    """Tables 1–6, Figures 1–2."""
    print("\n--- 4.1 Descriptive Statistics ---")

    # Table 1: Corpus overview
    t1_rows = [
        ("Total decisions", len(df)),
        ("Disciplinary decisions", len(disc)),
        ("Non-disciplinary decisions", len(df) - len(disc)),
        ("", ""),
        ("By source:", ""),
    ]
    for src, cnt in df["source"].value_counts().items():
        t1_rows.append((f"  {src}", cnt))
    t1_rows.append(("", ""))
    t1_rows.append(("By jurisdiction:", ""))
    for jur, cnt in df["jurisdiction"].value_counts().items():
        t1_rows.append((f"  {jur}", cnt))
    t1_rows.append(("", ""))
    t1_rows.append((f"Year range", f"{int(df['year'].min())}–{int(df['year'].max())}"))
    t1_rows.append(("Decisions with date", int(df["year"].notna().sum())))

    t1 = pd.DataFrame(t1_rows, columns=["Metric", "Value"])
    save_table(t1, "table01_corpus_overview.csv")

    # Record summary stats
    summary_stats["total_decisions"] = len(df)
    summary_stats["disciplinary_decisions"] = len(disc)
    summary_stats["non_disciplinary_decisions"] = len(df) - len(disc)
    summary_stats["pct_disciplinary"] = round(len(disc) / len(df) * 100, 1)
    summary_stats["year_range"] = f"{int(df['year'].min())}–{int(df['year'].max())}"
    summary_stats["jurisdictions"] = df["jurisdiction"].value_counts().to_dict()
    summary_stats["sources"] = df["source"].value_counts().to_dict()

    # Table 2: Decisions by profession
    prof_counts = disc["profession_label"].value_counts()
    t2 = pd.DataFrame({
        "Profession": prof_counts.index,
        "N": prof_counts.values,
        "Pct": (prof_counts.values / len(disc) * 100).round(1),
    })
    save_table(t2, "table02_profession_distribution.csv")
    summary_stats["profession_distribution"] = {
        row["Profession"]: int(row["N"]) for _, row in t2.iterrows()
    }

    # Table 3: Decisions by year
    year_counts = disc[disc["year"].notna()].groupby("year").size()
    t3 = pd.DataFrame({"Year": year_counts.index, "N": year_counts.values})
    save_table(t3, "table03_decisions_by_year.csv")

    # Table 4: Misconduct type prevalence
    mc_counts = {}
    for mc in MISCONDUCT_ORDER:
        mc_counts[mc] = int(disc[f"mc_{mc}"].sum())
    t4 = pd.DataFrame([
        {
            "Code": mc,
            "Misconduct type": MISCONDUCT_LABELS[mc],
            "N": mc_counts[mc],
            "Pct": round(mc_counts[mc] / len(disc) * 100, 1),
        }
        for mc in MISCONDUCT_ORDER
    ])
    save_table(t4, "table04_misconduct_prevalence.csv")
    summary_stats["misconduct_prevalence"] = mc_counts

    # Table 5: Outcome distribution
    oc_counts = {}
    for oc in OUTCOME_ORDER:
        oc_counts[oc] = int(disc[f"oc_{oc}"].sum())
    t5 = pd.DataFrame([
        {
            "Code": oc,
            "Outcome": OUTCOME_LABELS[oc],
            "N": oc_counts[oc],
            "Pct": round(oc_counts[oc] / len(disc) * 100, 1),
        }
        for oc in OUTCOME_ORDER
    ])
    save_table(t5, "table05_outcome_distribution.csv")
    summary_stats["outcome_distribution"] = oc_counts

    # Table 6: Finding distribution
    finding_counts = disc["finding_clean"].value_counts()
    finding_order = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]
    t6 = pd.DataFrame([
        {
            "Code": f,
            "Finding": FINDING_LABELS.get(f, f),
            "N": int(finding_counts.get(f, 0)),
            "Pct": round(finding_counts.get(f, 0) / len(disc) * 100, 1),
        }
        for f in finding_order
    ])
    save_table(t6, "table06_finding_distribution.csv")
    summary_stats["finding_distribution"] = {
        f: int(finding_counts.get(f, 0)) for f in finding_order
    }

    # Figure 1: Stacked bar — decisions per year by jurisdiction
    year_juris = disc[disc["year"].notna()].groupby(
        ["year", "jurisdiction"]
    ).size().unstack(fill_value=0)
    # Order jurisdictions by total count
    juris_order = year_juris.sum().sort_values(ascending=False).index.tolist()
    year_juris = year_juris[juris_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    year_juris.plot.bar(stacked=True, ax=ax, color=PALETTE[:len(juris_order)], width=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of decisions")
    ax.set_title("Disciplinary decisions per year by jurisdiction")
    ax.legend(title="Jurisdiction", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Thin x-tick labels
    labels = [str(int(float(t.get_text()))) for t in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    save_fig(fig, "fig01_decisions_year_jurisdiction.png")

    # Figure 2: Horizontal bar — misconduct type prevalence
    fig, ax = plt.subplots(figsize=(8, 4))
    mc_sorted = sorted(mc_counts.items(), key=lambda x: x[1])
    labels = [MISCONDUCT_LABELS[k] for k, _ in mc_sorted]
    vals = [v for _, v in mc_sorted]
    bars = ax.barh(labels, vals, color=PALETTE[0])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of disciplinary decisions")
    ax.set_title("Misconduct type prevalence (multi-label)")
    save_fig(fig, "fig02_misconduct_prevalence.png")


# ---------------------------------------------------------------------------
# 4.2 Profession Comparison
# ---------------------------------------------------------------------------
def profession_comparison(disc: pd.DataFrame):
    """Tables 7–8, Figures 3–4. RQ2."""
    print("\n--- 4.2 Profession Comparison ---")

    # Top 10 professions for cross-tabs
    top_profs = disc["profession_label"].value_counts().head(10).index.tolist()
    sub = disc[disc["profession_label"].isin(top_profs)].copy()

    # Table 7: Profession x Outcome
    rows = []
    for prof in top_profs:
        pf = sub[sub["profession_label"] == prof]
        row = {"Profession": prof, "N": len(pf)}
        for oc in OUTCOME_ORDER:
            n = int(pf[f"oc_{oc}"].sum())
            row[OUTCOME_LABELS[oc]] = n
            row[f"{OUTCOME_LABELS[oc]} %"] = round(n / len(pf) * 100, 1) if len(pf) else 0
        rows.append(row)
    t7 = pd.DataFrame(rows)
    save_table(t7, "table07_profession_x_outcome.csv")

    # Chi-square: profession vs each outcome
    for oc in OUTCOME_ORDER:
        ct = pd.crosstab(sub["profession_label"], sub[f"oc_{oc}"])
        safe_chi2(ct, f"profession_vs_{oc}")

    # Cramér's V overall: profession x most severe outcome
    ct_overall = pd.crosstab(sub["profession_label"], sub["most_severe_outcome"])
    safe_chi2(ct_overall, "profession_vs_most_severe_outcome")

    # Table 8: Profession x Finding
    rows = []
    for prof in top_profs:
        pf = sub[sub["profession_label"] == prof]
        row = {"Profession": prof, "N": len(pf)}
        for f in ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]:
            n = int((pf["finding_clean"] == f).sum())
            row[FINDING_LABELS[f]] = n
            row[f"{FINDING_LABELS[f]} %"] = round(n / len(pf) * 100, 1) if len(pf) else 0
        rows.append(row)
    t8 = pd.DataFrame(rows)
    save_table(t8, "table08_profession_x_finding.csv")

    ct_finding = pd.crosstab(sub["profession_label"], sub["finding_clean"])
    safe_chi2(ct_finding, "profession_vs_finding")

    # Figure 3: Heatmap — profession x outcome (normalised by profession)
    ct_pct = pd.crosstab(
        sub["profession_label"], sub["most_severe_outcome"], normalize="index"
    ) * 100
    # Reorder columns
    col_order = [c for c in OUTCOME_ORDER if c in ct_pct.columns]
    ct_pct = ct_pct[col_order]
    ct_pct.columns = [OUTCOME_LABELS.get(c, c) for c in col_order]
    # Reorder rows by total count
    row_order = sub["profession_label"].value_counts().index.tolist()
    ct_pct = ct_pct.reindex([r for r in row_order if r in ct_pct.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct_pct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% of profession"})
    ax.set_title("Most severe outcome by profession (%)")
    ax.set_ylabel("")
    save_fig(fig, "fig03_profession_outcome_heatmap.png")

    # Figure 4: Grouped bar — cancellation/suspension rates by profession
    rates = []
    for prof in top_profs:
        pf = sub[sub["profession_label"] == prof]
        n = len(pf)
        rates.append({
            "Profession": prof,
            "Cancellation %": round(pf["oc_CANCEL"].sum() / n * 100, 1),
            "Suspension %": round(pf["oc_SUSPEND"].sum() / n * 100, 1),
        })
    rates_df = pd.DataFrame(rates).set_index("Profession")
    # Sort by cancellation rate
    rates_df = rates_df.sort_values("Cancellation %", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    rates_df.plot.barh(ax=ax, color=[PALETTE[3], PALETTE[1]])
    ax.set_xlabel("% of disciplinary decisions")
    ax.set_title("Cancellation and suspension rates by profession")
    ax.legend(loc="lower right")
    save_fig(fig, "fig04_profession_cancel_suspend.png")


# ---------------------------------------------------------------------------
# 4.3 Misconduct Type Analysis
# ---------------------------------------------------------------------------
def misconduct_type_analysis(disc: pd.DataFrame):
    """Tables 9–10, Figures 5–6. RQ1."""
    print("\n--- 4.3 Misconduct Type Analysis ---")

    # Top 8 professions
    top_profs = disc["profession_label"].value_counts().head(8).index.tolist()
    sub = disc[disc["profession_label"].isin(top_profs)].copy()

    # Table 9: Misconduct type frequency overall and by profession
    rows = []
    for mc in MISCONDUCT_ORDER:
        row = {
            "Code": mc,
            "Misconduct type": MISCONDUCT_LABELS[mc],
            "Overall N": int(disc[f"mc_{mc}"].sum()),
            "Overall %": round(disc[f"mc_{mc}"].sum() / len(disc) * 100, 1),
        }
        for prof in top_profs:
            pf = sub[sub["profession_label"] == prof]
            n = int(pf[f"mc_{mc}"].sum())
            row[prof] = n
        rows.append(row)
    t9 = pd.DataFrame(rows)
    save_table(t9, "table09_misconduct_by_profession.csv")

    # Chi-square: profession vs each misconduct type
    for mc in MISCONDUCT_ORDER:
        ct = pd.crosstab(sub["profession_label"], sub[f"mc_{mc}"])
        safe_chi2(ct, f"profession_vs_mc_{mc}")

    # Table 10: Co-occurrence matrix
    mc_cols = [f"mc_{mc}" for mc in MISCONDUCT_ORDER]
    cooccur = disc[mc_cols].astype(int).T.dot(disc[mc_cols].astype(int))
    cooccur.index = [MISCONDUCT_LABELS[mc] for mc in MISCONDUCT_ORDER]
    cooccur.columns = [MISCONDUCT_LABELS[mc] for mc in MISCONDUCT_ORDER]
    save_table(cooccur, "table10_misconduct_cooccurrence.csv")

    summary_stats["misconduct_cooccurrence_diagonal"] = {
        mc: int(cooccur.iloc[i, i]) for i, mc in enumerate(MISCONDUCT_ORDER)
    }
    summary_stats["avg_misconduct_types"] = round(disc["n_misconduct_types"].mean(), 2)
    summary_stats["pct_multiple_misconduct"] = round(
        disc["has_multiple_misconduct"].mean() * 100, 1
    )

    # Figure 5: Heatmap — misconduct co-occurrence
    # Normalise: off-diagonal as % of row total (how often X co-occurs with Y)
    diag = np.diag(cooccur.values).copy()
    cooccur_pct = cooccur.copy().astype(float)
    for i in range(len(MISCONDUCT_ORDER)):
        if diag[i] > 0:
            cooccur_pct.iloc[i, :] = cooccur_pct.iloc[i, :] / diag[i] * 100
    # Mask the diagonal for display
    diag_mask = np.eye(len(MISCONDUCT_ORDER), dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cooccur_pct, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% co-occurrence"},
                mask=diag_mask)
    ax.set_title("Misconduct type co-occurrence (% of row type)")
    save_fig(fig, "fig05_misconduct_cooccurrence.png")

    # Figure 6: Stacked bar — misconduct type distribution by profession
    mc_by_prof = []
    for prof in top_profs:
        pf = sub[sub["profession_label"] == prof]
        total_mc = sum(pf[f"mc_{mc}"].sum() for mc in MISCONDUCT_ORDER)
        row = {"Profession": prof}
        for mc in MISCONDUCT_ORDER:
            n = int(pf[f"mc_{mc}"].sum())
            row[MISCONDUCT_LABELS[mc]] = round(n / total_mc * 100, 1) if total_mc else 0
        mc_by_prof.append(row)
    mc_prof_df = pd.DataFrame(mc_by_prof).set_index("Profession")

    fig, ax = plt.subplots(figsize=(10, 6))
    mc_prof_df.plot.barh(stacked=True, ax=ax, color=PALETTE[:len(MISCONDUCT_ORDER)])
    ax.set_xlabel("% of misconduct labels")
    ax.set_title("Misconduct type distribution by profession")
    ax.legend(title="Misconduct type", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    save_fig(fig, "fig06_misconduct_by_profession.png")


# ---------------------------------------------------------------------------
# 4.4 Prescribing Analysis Extension
# ---------------------------------------------------------------------------
def prescribing_analysis(disc: pd.DataFrame, full_df: pd.DataFrame):
    """Tables 11–12, Figure 7. RQ3."""
    print("\n--- 4.4 Prescribing Analysis ---")

    presc = disc[disc["mc_PRESC"]].copy()
    print(f"    {len(presc)} prescribing decisions")

    # Load entities for medication details
    entities = {}
    if ENTITIES_PATH.exists():
        with open(ENTITIES_PATH) as f:
            for line in f:
                rec = json.loads(line)
                entities[rec["id"]] = rec.get("entities", {})

    # Medication frequency
    med_counter = Counter()
    for _, row in presc.iterrows():
        meds = row.get("medications", [])
        if isinstance(meds, list):
            med_counter.update([m.lower() for m in meds])

    # Table 11: Top 20 medications
    top_meds = med_counter.most_common(20)
    t11 = pd.DataFrame(top_meds, columns=["Medication", "N"])
    t11["% of PRESC decisions"] = (t11["N"] / len(presc) * 100).round(1)

    # Add schedule classification from entities
    schedule_map = {}
    for doc_id, ents in entities.items():
        for sched in ents.get("SCHEDULE", []):
            normalised = sched.get("normalised", sched.get("text", ""))
            if normalised:
                # Map medications mentioned near this schedule
                for med in ents.get("MEDICATION", []):
                    schedule_map[med["text"].lower()] = normalised

    t11["Schedule"] = t11["Medication"].map(schedule_map).fillna("")
    save_table(t11, "table11_top20_medications.csv")

    # Opioid vs non-opioid analysis
    opioids = {
        "oxycodone", "oxycontin", "endone", "targin",
        "morphine", "ms contin", "kapanol",
        "fentanyl", "durogesic", "actiq",
        "methadone", "physeptone", "biodone",
        "hydromorphone", "dilaudid", "jurnista",
        "codeine", "panadeine forte",
        "tramadol",
        "buprenorphine", "suboxone", "subutex", "temgesic",
        "pethidine", "meperidine",
        "tapentadol", "palexia",
    }

    presc["has_opioid"] = presc["medications"].apply(
        lambda meds: any(m.lower() in opioids for m in meds) if isinstance(meds, list) else False
    )

    opioid = presc[presc["has_opioid"]]
    non_opioid = presc[~presc["has_opioid"]]

    # Table 12: Opioid vs non-opioid outcomes
    rows = []
    for label, subset in [("Opioid", opioid), ("Non-opioid", non_opioid)]:
        row = {"Group": label, "N": len(subset)}
        for oc in OUTCOME_ORDER:
            n = int(subset[f"oc_{oc}"].sum())
            row[f"{OUTCOME_LABELS[oc]} N"] = n
            row[f"{OUTCOME_LABELS[oc]} %"] = round(n / len(subset) * 100, 1) if len(subset) else 0
        rows.append(row)
    t12 = pd.DataFrame(rows)
    save_table(t12, "table12_opioid_vs_nonopioid.csv")

    summary_stats["prescribing"] = {
        "total_presc": len(presc),
        "pct_opioid": round(len(opioid) / len(presc) * 100, 1) if len(presc) else 0,
        "top_medication": top_meds[0][0] if top_meds else "",
        "top_medication_pct": round(top_meds[0][1] / len(presc) * 100, 1) if top_meds else 0,
    }

    # Chi-square: opioid vs cancellation/suspension
    for oc in ["CANCEL", "SUSPEND"]:
        ct = pd.crosstab(presc["has_opioid"], presc[f"oc_{oc}"])
        safe_chi2(ct, f"opioid_vs_{oc}")

    # Figure 7: Top 20 medications (publication quality)
    if top_meds:
        fig, ax = plt.subplots(figsize=(9, 7))
        meds = [m for m, _ in reversed(top_meds)]
        counts = [c for _, c in reversed(top_meds)]
        bars = ax.barh(meds, counts, color=PALETTE[0])
        for bar, v in zip(bars, counts):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=8)
        ax.set_xlabel("Number of prescribing misconduct decisions")
        ax.set_title("Top 20 medications in prescribing misconduct decisions")
        save_fig(fig, "fig07_top20_medications.png")


# ---------------------------------------------------------------------------
# 4.5 Boundary Violations Analysis
# ---------------------------------------------------------------------------
def boundary_analysis(disc: pd.DataFrame):
    """Tables 13–14. RQ3 boundary focus."""
    print("\n--- 4.5 Boundary Violations Analysis ---")

    bound = disc[disc["mc_BOUND"]].copy()
    non_bound = disc[~disc["mc_BOUND"]].copy()
    print(f"    {len(bound)} boundary violation decisions")

    # Table 13: Boundary violation sub-categorisation via keyword search
    sexual_kw = r"\bsexual\b|\bintercourse\b|\bindecent\b|\bgrop\b|\bmolest\b|\bsexualis"
    emotional_kw = r"\bemotional\b|\bdual relationship\b|\bpersonal relationship\b|\bintimate"
    financial_kw = r"\bfinancial\b|\bmoney\b|\bloan\b|\bborrow\b|\bgift\b|\bbusiness relationship"

    bound["sub_sexual"] = bound["full_text"].str.contains(sexual_kw, case=False, na=False)
    bound["sub_emotional"] = bound["full_text"].str.contains(emotional_kw, case=False, na=False)
    bound["sub_financial"] = bound["full_text"].str.contains(financial_kw, case=False, na=False)
    bound["sub_other"] = ~(bound["sub_sexual"] | bound["sub_emotional"] | bound["sub_financial"])

    t13 = pd.DataFrame([
        {"Subcategory": "Sexual boundary violation", "N": int(bound["sub_sexual"].sum()),
         "%": round(bound["sub_sexual"].mean() * 100, 1)},
        {"Subcategory": "Emotional boundary violation", "N": int(bound["sub_emotional"].sum()),
         "%": round(bound["sub_emotional"].mean() * 100, 1)},
        {"Subcategory": "Financial boundary violation", "N": int(bound["sub_financial"].sum()),
         "%": round(bound["sub_financial"].mean() * 100, 1)},
        {"Subcategory": "Other/unclassified", "N": int(bound["sub_other"].sum()),
         "%": round(bound["sub_other"].mean() * 100, 1)},
    ])
    save_table(t13, "table13_boundary_subcategories.csv")

    summary_stats["boundary"] = {
        "total_bound": len(bound),
        "pct_sexual": round(bound["sub_sexual"].mean() * 100, 1),
    }

    # Table 14: Outcome severity — BOUND vs other
    rows = []
    for label, subset in [("Boundary violations", bound), ("Other misconduct", non_bound)]:
        row = {"Group": label, "N": len(subset)}
        for oc in OUTCOME_ORDER:
            n = int(subset[f"oc_{oc}"].sum())
            row[f"{OUTCOME_LABELS[oc]} N"] = n
            row[f"{OUTCOME_LABELS[oc]} %"] = round(n / len(subset) * 100, 1) if len(subset) else 0
        rows.append(row)
    t14 = pd.DataFrame(rows)
    save_table(t14, "table14_boundary_vs_other_outcomes.csv")

    # Chi-square: BOUND vs non-BOUND for severe outcomes
    for oc in ["CANCEL", "SUSPEND"]:
        ct = pd.crosstab(disc["mc_BOUND"], disc[f"oc_{oc}"])
        safe_chi2(ct, f"boundary_vs_{oc}")

    # Fisher's exact for cancellation (2x2)
    ct_cancel = pd.crosstab(disc["mc_BOUND"], disc["oc_CANCEL"])
    if ct_cancel.shape == (2, 2):
        odds, p = stats.fisher_exact(ct_cancel.values)
        result = {
            "test": "fisher_boundary_vs_CANCEL",
            "odds_ratio": round(odds, 3),
            "p_value": float(f"{p:.6g}"),
        }
        chi_square_results.append(result)
        print(f"    Fisher's exact (BOUND vs CANCEL): OR={odds:.3f}, p={p:.4g}")


# ---------------------------------------------------------------------------
# 4.6 Temporal Trends
# ---------------------------------------------------------------------------
def temporal_trends(disc: pd.DataFrame):
    """Tables 15, Figures 8–9. RQ4."""
    print("\n--- 4.6 Temporal Trends ---")

    disc_year = disc[disc["year"].notna()].copy()
    disc_year["year_int"] = disc_year["year"].astype(int)

    # Table 15: Misconduct type counts by year (wide)
    year_range = range(int(disc_year["year_int"].min()), int(disc_year["year_int"].max()) + 1)
    rows = []
    for yr in year_range:
        yr_sub = disc_year[disc_year["year_int"] == yr]
        row = {"Year": yr, "Total": len(yr_sub)}
        for mc in MISCONDUCT_ORDER:
            row[MISCONDUCT_LABELS[mc]] = int(yr_sub[f"mc_{mc}"].sum())
        rows.append(row)
    t15 = pd.DataFrame(rows)
    save_table(t15, "table15_misconduct_by_year.csv")

    # Mann-Kendall trend tests
    # Need at least 8 years of data for meaningful test
    annual_totals = disc_year.groupby("year_int").size()
    for mc in MISCONDUCT_ORDER:
        annual = disc_year.groupby("year_int")[f"mc_{mc}"].sum()
        # Normalise by annual total decisions
        annual_pct = (annual / annual_totals * 100).dropna()
        if len(annual_pct) >= 8:
            try:
                result = mk.original_test(annual_pct.values)
                trend_results.append({
                    "misconduct_type": mc,
                    "label": MISCONDUCT_LABELS[mc],
                    "trend": result.trend,
                    "p_value": float(f"{result.p:.6g}"),
                    "tau": round(result.Tau, 3),
                    "sens_slope": round(result.slope, 4),
                    "n_years": len(annual_pct),
                })
                print(f"    {mc}: trend={result.trend}, tau={result.Tau:.3f}, p={result.p:.4g}")
            except Exception as e:
                print(f"    {mc}: Mann-Kendall failed — {e}")

    # Also test total decisions trend
    if len(annual_totals) >= 8:
        result = mk.original_test(annual_totals.values)
        trend_results.append({
            "misconduct_type": "TOTAL",
            "label": "Total decisions",
            "trend": result.trend,
            "p_value": float(f"{result.p:.6g}"),
            "tau": round(result.Tau, 3),
            "sens_slope": round(result.slope, 4),
            "n_years": len(annual_totals),
        })
        print(f"    TOTAL: trend={result.trend}, tau={result.Tau:.3f}, p={result.p:.4g}")

    # Figure 8: Line chart — misconduct type trends normalised
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mc in enumerate(MISCONDUCT_ORDER):
        annual = disc_year.groupby("year_int")[f"mc_{mc}"].sum()
        annual_pct = annual / annual_totals * 100
        ax.plot(annual_pct.index, annual_pct.values,
                label=MISCONDUCT_LABELS[mc], color=PALETTE[i], marker=".", markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of annual disciplinary decisions")
    ax.set_title("Misconduct type trends over time (normalised)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    save_fig(fig, "fig08_misconduct_trends.png")

    # Figure 9: Total decisions per year with trend line
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual_totals.index, annual_totals.values, color=PALETTE[0], alpha=0.7)
    # Trend line
    z = np.polyfit(annual_totals.index.astype(float), annual_totals.values.astype(float), 1)
    p_trend = np.poly1d(z)
    ax.plot(annual_totals.index, p_trend(annual_totals.index),
            color=PALETTE[3], linestyle="--", linewidth=2, label="Linear trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of disciplinary decisions")
    ax.set_title("Annual disciplinary decisions with trend")
    ax.legend()
    save_fig(fig, "fig09_annual_decisions_trend.png")


# ---------------------------------------------------------------------------
# 4.7 Jurisdiction Comparison
# ---------------------------------------------------------------------------
def jurisdiction_comparison(disc: pd.DataFrame):
    """Tables 16–17, Figures 10–11. RQ5."""
    print("\n--- 4.7 Jurisdiction Comparison ---")

    jurisdictions = disc["jurisdiction"].value_counts().index.tolist()

    # Table 16: Jurisdiction x Outcome
    rows = []
    for jur in jurisdictions:
        jf = disc[disc["jurisdiction"] == jur]
        row = {"Jurisdiction": jur, "N": len(jf)}
        for oc in OUTCOME_ORDER:
            n = int(jf[f"oc_{oc}"].sum())
            row[f"{OUTCOME_LABELS[oc]} N"] = n
            row[f"{OUTCOME_LABELS[oc]} %"] = round(n / len(jf) * 100, 1) if len(jf) else 0
        rows.append(row)
    t16 = pd.DataFrame(rows)
    save_table(t16, "table16_jurisdiction_x_outcome.csv")

    ct_jur_oc = pd.crosstab(disc["jurisdiction"], disc["most_severe_outcome"])
    safe_chi2(ct_jur_oc, "jurisdiction_vs_most_severe_outcome")

    # Table 17: Jurisdiction x Misconduct type
    rows = []
    for jur in jurisdictions:
        jf = disc[disc["jurisdiction"] == jur]
        row = {"Jurisdiction": jur, "N": len(jf)}
        for mc in MISCONDUCT_ORDER:
            n = int(jf[f"mc_{mc}"].sum())
            row[f"{MISCONDUCT_LABELS[mc]} N"] = n
            row[f"{MISCONDUCT_LABELS[mc]} %"] = round(n / len(jf) * 100, 1) if len(jf) else 0
        rows.append(row)
    t17 = pd.DataFrame(rows)
    save_table(t17, "table17_jurisdiction_x_misconduct.csv")

    ct_jur_mc = pd.DataFrame()
    for mc in MISCONDUCT_ORDER:
        ct = pd.crosstab(disc["jurisdiction"], disc[f"mc_{mc}"])
        safe_chi2(ct, f"jurisdiction_vs_mc_{mc}")

    # Pairwise jurisdiction comparisons for outcome (top 5 jurisdictions)
    top_jurs = disc["jurisdiction"].value_counts().head(5).index.tolist()
    from itertools import combinations
    for j1, j2 in combinations(top_jurs, 2):
        pair = disc[disc["jurisdiction"].isin([j1, j2])]
        ct = pd.crosstab(pair["jurisdiction"], pair["most_severe_outcome"])
        safe_chi2(ct, f"jurisdiction_pairwise_{j1}_vs_{j2}")

    # Figure 10: Grouped bar — outcome rates by jurisdiction
    # Normalise by jurisdiction total
    jur_oc_pct = pd.crosstab(
        disc["jurisdiction"], disc["most_severe_outcome"], normalize="index"
    ) * 100
    col_order = [c for c in OUTCOME_ORDER if c in jur_oc_pct.columns]
    jur_oc_pct = jur_oc_pct[col_order]
    jur_oc_pct.columns = [OUTCOME_LABELS.get(c, c) for c in col_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    jur_oc_pct.plot.bar(ax=ax, color=PALETTE[:len(col_order)], width=0.8)
    ax.set_xlabel("Jurisdiction")
    ax.set_ylabel("% of decisions")
    ax.set_title("Most severe outcome by jurisdiction")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    save_fig(fig, "fig10_jurisdiction_outcome.png")

    # Figure 11: Heatmap — jurisdiction x misconduct type
    jur_mc_pct = pd.DataFrame()
    for mc in MISCONDUCT_ORDER:
        jur_mc_pct[MISCONDUCT_LABELS[mc]] = disc.groupby("jurisdiction")[f"mc_{mc}"].mean() * 100
    # Order by total decisions
    jur_mc_pct = jur_mc_pct.reindex(jurisdictions)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(jur_mc_pct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% of jurisdiction decisions"})
    ax.set_title("Misconduct type prevalence by jurisdiction (%)")
    ax.set_ylabel("")
    save_fig(fig, "fig11_jurisdiction_misconduct_heatmap.png")


# ---------------------------------------------------------------------------
# 4.8 COVID-19 Analysis
# ---------------------------------------------------------------------------
def covid_analysis(disc: pd.DataFrame):
    """Table 18, Figure 12. RQ8."""
    print("\n--- 4.8 COVID-19 Analysis ---")

    disc_covid = disc[disc["covid_period"].notna()].copy()
    periods = ["Pre-COVID (<2020)", "COVID (2020–2021)", "Post-COVID (2022+)"]

    # Table 18: Decision counts and misconduct by period
    rows = []
    for period in periods:
        pf = disc_covid[disc_covid["covid_period"] == period]
        row = {"Period": period, "N": len(pf)}
        for mc in MISCONDUCT_ORDER:
            n = int(pf[f"mc_{mc}"].sum())
            row[f"{MISCONDUCT_LABELS[mc]} N"] = n
            row[f"{MISCONDUCT_LABELS[mc]} %"] = round(n / len(pf) * 100, 1) if len(pf) else 0
        # Also add outcome columns
        for oc in OUTCOME_ORDER:
            n = int(pf[f"oc_{oc}"].sum())
            row[f"{OUTCOME_LABELS[oc]} N"] = n
            row[f"{OUTCOME_LABELS[oc]} %"] = round(n / len(pf) * 100, 1) if len(pf) else 0
        rows.append(row)
    t18 = pd.DataFrame(rows)
    save_table(t18, "table18_covid_periods.csv")

    # Chi-square: period vs misconduct type distribution
    for mc in MISCONDUCT_ORDER:
        ct = pd.crosstab(disc_covid["covid_period"], disc_covid[f"mc_{mc}"])
        safe_chi2(ct, f"covid_vs_mc_{mc}")

    # Chi-square: period vs outcome distribution
    ct_period_oc = pd.crosstab(disc_covid["covid_period"], disc_covid["most_severe_outcome"])
    safe_chi2(ct_period_oc, "covid_vs_most_severe_outcome")

    summary_stats["covid"] = {
        period: int((disc_covid["covid_period"] == period).sum())
        for period in periods
    }

    # Figure 12: Grouped bar — misconduct type % by COVID period
    mc_by_period = []
    for period in periods:
        pf = disc_covid[disc_covid["covid_period"] == period]
        row = {"Period": period}
        for mc in MISCONDUCT_ORDER:
            row[MISCONDUCT_LABELS[mc]] = round(pf[f"mc_{mc}"].mean() * 100, 1) if len(pf) else 0
        mc_by_period.append(row)
    mc_period_df = pd.DataFrame(mc_by_period).set_index("Period")

    fig, ax = plt.subplots(figsize=(12, 6))
    mc_period_df.plot.bar(ax=ax, color=PALETTE[:len(MISCONDUCT_ORDER)], width=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("% of decisions in period")
    ax.set_title("Misconduct type prevalence by COVID period")
    ax.legend(title="Misconduct type", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    save_fig(fig, "fig12_covid_misconduct.png")


# ---------------------------------------------------------------------------
# Save statistical results
# ---------------------------------------------------------------------------
def save_statistical_results():
    """Write machine-readable JSON outputs."""
    print("\n--- Saving statistical results ---")

    with open(STATS_DIR / "chi_square_results.json", "w") as f:
        json.dump(chi_square_results, f, indent=2)
    print(f"    Saved chi_square_results.json ({len(chi_square_results)} tests)")

    with open(STATS_DIR / "trend_tests.json", "w") as f:
        json.dump(trend_results, f, indent=2)
    print(f"    Saved trend_tests.json ({len(trend_results)} tests)")

    with open(STATS_DIR / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"    Saved summary_statistics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 4: Statistical Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading enriched corpus...")
    df = load_corpus()

    # Filter to disciplinary decisions
    disc = df[df["predicted_is_disciplinary"]].copy()
    print(f"  Disciplinary subset: {len(disc)} decisions")

    # Run all analyses
    descriptive_statistics(df, disc)
    profession_comparison(disc)
    misconduct_type_analysis(disc)
    prescribing_analysis(disc, df)
    boundary_analysis(disc)
    temporal_trends(disc)
    jurisdiction_comparison(disc)
    covid_analysis(disc)

    # Save results
    save_statistical_results()

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Tables:     {len(list(TABLES_DIR.glob('table*.csv')))} saved to {TABLES_DIR}")
    print(f"  Figures:    {len(list(FIGURES_DIR.glob('fig*.png')))} saved to {FIGURES_DIR}")
    print(f"  Statistics: {len(list(STATS_DIR.glob('*.json')))} saved to {STATS_DIR}")
    print(f"  Chi-square tests: {len(chi_square_results)}")
    print(f"  Trend tests: {len(trend_results)}")


if __name__ == "__main__":
    main()
