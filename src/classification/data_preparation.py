"""Prepare gold-standard data for classification experiments.

Loads the 300-decision gold standard and full corpus, joins labels with
full text, creates stratified train/val/test splits, binarises multi-label
fields, and saves splits to data/annotations/splits/.

Usage:
    python -m src.classification.data_preparation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
# Nov 2026 gold standard (re-code + logged model pass + adjudication); see its manifest.
GOLD_STANDARD_PATH = DATA_DIR / "annotations" / "gold_standard_2026-11.json"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
SPLITS_DIR = DATA_DIR / "annotations" / "splits"
# Feb 2026 split membership, kept fixed when labels change (config/retrain_protocol_2026-11.md)
SPLIT_IDS_PATH = SPLITS_DIR / "split_ids.json"
TAXONOMY_PATH = CONFIG_DIR / "taxonomy.json"

# Label lists (aligned with taxonomy.json)
MISCONDUCT_L1_CODES = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF", "N/A"]
FINDING_CODES = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]
OUTCOME_CODES = [
    "CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
    "FINE", "CAUTION", "UNDERTAKING", "NO_ADVERSE", "OTHER", "N/A",
]

RANDOM_SEED = 42


def load_gold_standard() -> list[dict]:
    """Load the 300-decision gold standard annotations."""
    with open(GOLD_STANDARD_PATH) as f:
        return json.load(f)


def load_corpus() -> dict[str, dict]:
    """Load full corpus as {id: record} dict."""
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            record = json.loads(line)
            corpus[record["id"]] = record
    return corpus


def join_labels_with_text(gold: list[dict], corpus: dict[str, dict]) -> pd.DataFrame:
    """Join gold-standard labels with full_text from the corpus.

    Returns a DataFrame with columns:
        id, full_text, is_disciplinary, misconduct_L1, finding, outcomes, profession
    """
    rows = []
    missing = []
    for entry in gold:
        doc_id = entry["id"]
        if doc_id not in corpus:
            missing.append(doc_id)
            continue
        rows.append({
            "id": doc_id,
            "full_text": corpus[doc_id]["full_text"],
            "is_disciplinary": entry["is_disciplinary"],
            "misconduct_L1": entry["misconduct_L1"],
            "finding": entry.get("finding"),
            "outcomes": entry["outcomes"],
            "profession": entry.get("profession", "unknown"),
        })

    if missing:
        print(f"WARNING: {len(missing)} gold-standard IDs not found in corpus: {missing[:5]}...")

    return pd.DataFrame(rows)


def binarise_multilabel(series: pd.Series, label_set: list[str]) -> pd.DataFrame:
    """Convert a Series of label lists into a binary matrix."""
    matrix = pd.DataFrame(0, index=series.index, columns=label_set)
    for idx, labels in series.items():
        if labels is None:
            continue
        for label in labels:
            if label in label_set:
                matrix.at[idx, label] = 1
    return matrix


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split: 70/15/15, stratified by is_disciplinary.

    Returns (train, val, test) DataFrames. If split_ids.json exists, its membership is reused
    instead of re-stratifying, so a label change cannot move decisions between splits.
    """
    if SPLIT_IDS_PATH.exists():
        ids = json.loads(SPLIT_IDS_PATH.read_text())
        by_id = df.set_index("id", drop=False)
        if sorted(sum(ids.values(), [])) != sorted(df["id"]):
            sys.exit("split_ids.json does not cover exactly the gold-standard ids")
        train, val, test = (by_id.loc[ids[k]].reset_index(drop=True) for k in ("train", "val", "test"))
        print(f"Split sizes (fixed ids) — train: {len(train)}, val: {len(val)}, test: {len(test)}")
        for name, part in [("Train", train), ("Val", val), ("Test", test)]:
            print(f"  {name} disciplinary: {part['is_disciplinary'].sum()}/{len(part)}")
        return train, val, test

    # Stratify by is_disciplinary to maintain class balance
    strat_col = df["is_disciplinary"].astype(int)

    train, temp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=strat_col,
    )
    # Split temp into val (50%) and test (50%) — i.e. 15% each of total
    strat_temp = temp["is_disciplinary"].astype(int)
    val, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_SEED, stratify=strat_temp,
    )

    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"  Train disciplinary: {train['is_disciplinary'].sum()}/{len(train)}")
    print(f"  Val   disciplinary: {val['is_disciplinary'].sum()}/{len(val)}")
    print(f"  Test  disciplinary: {test['is_disciplinary'].sum()}/{len(test)}")

    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save train/val/test splits as JSONL files."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = SPLITS_DIR / f"{name}.jsonl"
        records = split_df.to_dict(orient="records")
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        print(f"Saved {len(records)} records to {path}")


def print_label_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Print label distribution for a split."""
    print(f"\n--- {split_name} label distribution ---")

    # Misconduct L1
    misconduct_counts = {}
    for labels in df["misconduct_L1"]:
        if labels:
            for l in labels:
                misconduct_counts[l] = misconduct_counts.get(l, 0) + 1
    print(f"  Misconduct L1: {dict(sorted(misconduct_counts.items(), key=lambda x: -x[1]))}")

    # Finding
    finding_counts = df["finding"].value_counts().to_dict()
    print(f"  Finding: {finding_counts}")

    # Outcomes
    outcome_counts = {}
    for labels in df["outcomes"]:
        if labels:
            for l in labels:
                outcome_counts[l] = outcome_counts.get(l, 0) + 1
    print(f"  Outcomes: {dict(sorted(outcome_counts.items(), key=lambda x: -x[1]))}")


def main():
    print("=== Data Preparation for Classification ===\n")

    # Load data
    print("Loading gold standard...")
    gold = load_gold_standard()
    print(f"  {len(gold)} annotations loaded")

    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus)} decisions loaded")

    # Join
    print("\nJoining labels with text...")
    df = join_labels_with_text(gold, corpus)
    print(f"  {len(df)} records joined")

    # Verify text lengths
    text_lengths = df["full_text"].str.len()
    print(f"  Text lengths: min={text_lengths.min()}, median={text_lengths.median():.0f}, max={text_lengths.max()}")

    # Create splits
    print("\nCreating train/val/test splits...")
    train, val, test = create_splits(df)

    # Print distributions
    print_label_distribution(train, "Train")
    print_label_distribution(val, "Val")
    print_label_distribution(test, "Test")

    # Save
    print("\nSaving splits...")
    save_splits(train, val, test)

    # Also save the binarised label matrices for convenience
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        misconduct_bin = binarise_multilabel(split_df["misconduct_L1"], MISCONDUCT_L1_CODES)
        outcome_bin = binarise_multilabel(split_df["outcomes"], OUTCOME_CODES)

        misconduct_bin.to_csv(SPLITS_DIR / f"{name}_misconduct_binary.csv", index=False)
        outcome_bin.to_csv(SPLITS_DIR / f"{name}_outcome_binary.csv", index=False)

    print(f"\nBinary label matrices saved to {SPLITS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
