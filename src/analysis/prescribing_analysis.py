"""Prescribing misconduct analysis pipeline.

Filters to PRESC-classified decisions, applies NER to extract medications
and schedules, and computes cross-tabulations by profession, jurisdiction,
and year.

Usage:
    python -m src.analysis.prescribing_analysis
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
PREDICTIONS_PATH = DATA_DIR / "extracted" / "misconduct_predictions_sklearn.jsonl"
ENTITIES_PATH = DATA_DIR / "extracted" / "entities.jsonl"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def load_data() -> tuple[dict, dict, dict]:
    """Load corpus, predictions, and entities."""
    # Corpus metadata
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            record = json.loads(line)
            corpus[record["id"]] = record

    # Predictions
    predictions = {}
    with open(PREDICTIONS_PATH) as f:
        for line in f:
            record = json.loads(line)
            predictions[record["id"]] = record

    # Entities
    entities = {}
    with open(ENTITIES_PATH) as f:
        for line in f:
            record = json.loads(line)
            entities[record["id"]] = record["entities"]

    return corpus, predictions, entities


def get_presc_decisions(
    corpus: dict, predictions: dict, entities: dict,
) -> pd.DataFrame:
    """Filter to PRESC-classified decisions and combine data."""
    rows = []
    for doc_id, pred in predictions.items():
        if "PRESC" not in pred.get("predicted_misconduct_L1", []):
            continue
        if doc_id not in corpus:
            continue

        record = corpus[doc_id]
        ents = entities.get(doc_id, {})

        medications = [e["text"].lower() for e in ents.get("MEDICATION", [])]
        schedules = [e.get("normalised", e["text"]) for e in ents.get("SCHEDULE", [])]

        # Extract year from date
        date_str = record.get("date", "")
        year = None
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
            except ValueError:
                pass

        rows.append({
            "id": doc_id,
            "profession": record.get("profession", "unknown"),
            "jurisdiction": record.get("jurisdiction", "unknown"),
            "tribunal": record.get("tribunal", "unknown"),
            "year": year,
            "medications": medications,
            "n_medications": len(medications),
            "schedules": schedules,
            "has_s8": "Schedule 8" in schedules or "controlled substance" in schedules or "drug of dependence" in schedules,
            "has_s4": "Schedule 4" in schedules or "Schedule 4 Appendix D" in schedules,
        })

    return pd.DataFrame(rows)


def medication_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute medication frequency across prescribing decisions."""
    counter = Counter()
    for meds in df["medications"]:
        counter.update(meds)

    freq_df = pd.DataFrame(
        counter.most_common(),
        columns=["medication", "n_decisions"],
    )
    freq_df["pct_of_presc"] = (freq_df["n_decisions"] / len(df) * 100).round(1)
    return freq_df


def schedule_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute schedule reference distribution."""
    counter = Counter()
    for schedules in df["schedules"]:
        counter.update(schedules)

    sched_df = pd.DataFrame(
        counter.most_common(),
        columns=["schedule_type", "n_decisions"],
    )
    sched_df["pct_of_presc"] = (sched_df["n_decisions"] / len(df) * 100).round(1)
    return sched_df


def opioid_subset_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse opioid-specific prescribing decisions."""
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

    df = df.copy()
    df["has_opioid"] = df["medications"].apply(
        lambda meds: any(m in opioids for m in meds)
    )

    opioid_df = df[df["has_opioid"]]
    return opioid_df


def cross_tabulate(df: pd.DataFrame) -> None:
    """Create cross-tabulation tables and save."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Profession x has_s8
    if len(df) > 0:
        prof_schedule = pd.crosstab(
            df["profession"],
            df["has_s8"],
            margins=True,
        )
        prof_schedule.to_csv(TABLES_DIR / "presc_profession_x_s8.csv")
        print(f"  Saved presc_profession_x_s8.csv")

    # Jurisdiction distribution
    juris_counts = df["jurisdiction"].value_counts()
    juris_counts.to_csv(TABLES_DIR / "presc_by_jurisdiction.csv")
    print(f"  Saved presc_by_jurisdiction.csv")

    # Year trend
    year_counts = df[df["year"].notna()].groupby("year").size()
    year_counts.to_csv(TABLES_DIR / "presc_by_year.csv")
    print(f"  Saved presc_by_year.csv")

    # Profession distribution
    prof_counts = df["profession"].value_counts()
    prof_counts.to_csv(TABLES_DIR / "presc_by_profession.csv")
    print(f"  Saved presc_by_profession.csv")


def generate_figures(df: pd.DataFrame, med_freq: pd.DataFrame) -> None:
    """Generate prescribing analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Top 20 medications
    if len(med_freq) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = med_freq.head(20)
        ax.barh(top20["medication"][::-1], top20["n_decisions"][::-1])
        ax.set_xlabel("Number of Decisions")
        ax.set_title("Top 20 Medications in Prescribing Misconduct Decisions")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "presc_top20_medications.png", dpi=150)
        plt.close()
        print(f"  Saved presc_top20_medications.png")

    # Year trend
    year_data = df[df["year"].notna()].copy()
    if len(year_data) > 0:
        year_counts = year_data.groupby("year").size()
        fig, ax = plt.subplots(figsize=(10, 5))
        year_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Decisions")
        ax.set_title("Prescribing Misconduct Decisions by Year")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "presc_by_year.png", dpi=150)
        plt.close()
        print(f"  Saved presc_by_year.png")

    # Profession distribution
    if len(df) > 0:
        prof_counts = df["profession"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        prof_counts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of Decisions")
        ax.set_title("Prescribing Misconduct by Profession (Top 10)")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "presc_by_profession.png", dpi=150)
        plt.close()
        print(f"  Saved presc_by_profession.png")


def main():
    print("=== Prescribing Analysis Pipeline ===\n")

    # Load data
    print("Loading data...")
    corpus, predictions, entities = load_data()
    print(f"  Corpus: {len(corpus)}, Predictions: {len(predictions)}, Entities: {len(entities)}")

    # Filter to PRESC decisions
    print("\nFiltering to PRESC-classified decisions...")
    df = get_presc_decisions(corpus, predictions, entities)
    print(f"  {len(df)} prescribing misconduct decisions")

    if len(df) == 0:
        print("  No prescribing decisions found. Exiting.")
        return

    # Medication frequency
    print("\nMedication frequency analysis...")
    med_freq = medication_frequency_table(df)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    med_freq.to_csv(TABLES_DIR / "medication_frequency.csv", index=False)
    print(f"  {len(med_freq)} unique medications found")
    print(f"  Top 10:")
    for _, row in med_freq.head(10).iterrows():
        print(f"    {row['medication']}: {row['n_decisions']} ({row['pct_of_presc']}%)")

    # Schedule distribution
    print("\nSchedule distribution...")
    sched_dist = schedule_distribution(df)
    sched_dist.to_csv(TABLES_DIR / "schedule_distribution.csv", index=False)
    for _, row in sched_dist.iterrows():
        print(f"  {row['schedule_type']}: {row['n_decisions']} ({row['pct_of_presc']}%)")

    # Opioid subset
    print("\nOpioid subset analysis...")
    opioid_df = opioid_subset_analysis(df)
    print(f"  Decisions with opioid mention: {len(opioid_df)}/{len(df)} ({len(opioid_df)/len(df)*100:.1f}%)")

    # Cross-tabulations
    print("\nCross-tabulations...")
    cross_tabulate(df)

    # Figures
    print("\nGenerating figures...")
    generate_figures(df, med_freq)

    print("\nDone.")


if __name__ == "__main__":
    main()
