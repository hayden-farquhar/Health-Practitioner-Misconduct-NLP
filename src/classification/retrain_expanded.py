"""Merge new LLM annotations with original gold standard and retrain classifiers.

Loads 12 batch annotation files (new_batch_01..12.json), merges with original
210 training decisions, retrains TF-IDF+LR classifiers, and compares performance
against the original 210-decision baseline.

Val (45) and test (45) sets remain unchanged for fair comparison.

Usage:
    .venv-ml/bin/python -m src.classification.retrain_expanded
"""

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "annotations" / "splits"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
MODEL_DIR = PROJECT_ROOT / "models" / "sklearn_expanded"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "statistics"

MISCONDUCT_L1_CODES = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF", "N/A"]
FINDING_CODES = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]


def load_split(name: str) -> pd.DataFrame:
    """Load a JSONL split file into a DataFrame."""
    path = SPLITS_DIR / f"{name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_corpus_texts() -> dict:
    """Load full_text for all decisions from corpus, keyed by ID."""
    texts = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            texts[d["id"]] = d["full_text"]
    return texts


def load_new_annotations() -> list:
    """Load and merge all new batch annotation files."""
    all_annotations = []
    missing = []

    for i in range(1, 13):
        batch_file = ANNOTATIONS_DIR / f"new_batch_{i:02d}.json"
        if batch_file.exists():
            with open(batch_file) as f:
                batch = json.load(f)
            all_annotations.extend(batch)
            print(f"  Loaded {batch_file.name}: {len(batch)} decisions")
        else:
            missing.append(batch_file.name)
            print(f"  MISSING: {batch_file.name}")

    if missing:
        print(f"\n  WARNING: {len(missing)} batch files missing: {missing}")

    return all_annotations


def validate_annotations(annotations: list) -> list:
    """Validate and clean annotations, reporting issues."""
    valid = []
    issues = []

    valid_misconduct = set(MISCONDUCT_L1_CODES)
    valid_findings = set(FINDING_CODES + ["N/A"])
    valid_outcomes = {"CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "CAUTION",
                      "NO_ADVERSE", "UNDERTAKING", "FINE", "N/A"}

    for ann in annotations:
        problems = []

        # Required fields
        if "id" not in ann:
            problems.append("missing id")
            continue

        if "is_disciplinary" not in ann:
            problems.append("missing is_disciplinary")

        # Validate misconduct codes
        if "misconduct_L1" in ann:
            invalid_codes = [c for c in ann["misconduct_L1"] if c not in valid_misconduct]
            if invalid_codes:
                problems.append(f"invalid misconduct codes: {invalid_codes}")
                ann["misconduct_L1"] = [c for c in ann["misconduct_L1"] if c in valid_misconduct]

        # Validate finding
        if "finding" in ann and ann["finding"] not in valid_findings:
            problems.append(f"invalid finding: {ann['finding']}")

        # Validate outcomes
        if "outcomes" in ann:
            invalid_outcomes = [o for o in ann["outcomes"] if o not in valid_outcomes]
            if invalid_outcomes:
                problems.append(f"invalid outcomes: {invalid_outcomes}")
                ann["outcomes"] = [o for o in ann["outcomes"] if o in valid_outcomes]

        if problems:
            issues.append((ann.get("id", "unknown"), problems))

        valid.append(ann)

    if issues:
        print(f"\n  Validation issues in {len(issues)} annotations:")
        for aid, probs in issues[:10]:
            print(f"    {aid}: {probs}")

    return valid


def binarise_multilabel(series: pd.Series, label_set: list[str]) -> np.ndarray:
    """Convert a Series of label lists into a binary numpy matrix."""
    matrix = np.zeros((len(series), len(label_set)), dtype=int)
    for i, labels in enumerate(series):
        if labels is None:
            continue
        if isinstance(labels, str):
            labels = json.loads(labels.replace("'", '"'))
        for label in labels:
            if label in label_set:
                matrix[i, label_set.index(label)] = 1
    return matrix


def train_and_evaluate(
    X_train, y_train_misc, y_train_finding, y_train_disc,
    X_val, y_val_misc, y_val_finding, y_val_disc,
    X_test, y_test_misc, y_test_finding, y_test_disc,
    label: str = "",
) -> dict:
    """Train all three classifiers and return performance metrics."""
    results = {"label": label}

    # 1. Multi-label misconduct
    print(f"\n{'='*60}")
    print(f"MISCONDUCT CLASSIFIER ({label})")
    print(f"{'='*60}")

    best_f1 = 0
    best_C = 1.0
    best_model = None

    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs", random_state=42,
                class_weight="balanced",
            )
        )
        clf.fit(X_train, y_train_misc)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val_misc, y_val_pred, average="macro", zero_division=0)
        print(f"  C={C:<6} val macro-F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model = clf

    print(f"\nBest C={best_C} (val macro-F1={best_f1:.4f})")

    y_test_pred = best_model.predict(X_test)
    report = classification_report(
        y_test_misc, y_test_pred,
        target_names=MISCONDUCT_L1_CODES,
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        y_test_misc, y_test_pred,
        target_names=MISCONDUCT_L1_CODES,
        zero_division=0,
    ))

    hl = hamming_loss(y_test_misc, y_test_pred)
    print(f"Hamming loss: {hl:.4f}")

    results["misconduct"] = {
        "best_C": best_C,
        "val_macro_f1": round(best_f1, 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "test_micro_f1": round(report["micro avg"]["f1-score"], 4),
        "hamming_loss": round(hl, 4),
        "per_class": {code: round(report[code]["f1-score"], 4) for code in MISCONDUCT_L1_CODES if code in report},
    }
    results["_misconduct_clf"] = best_model

    # 2. Finding classifier
    print(f"\n{'='*60}")
    print(f"FINDING CLASSIFIER ({label})")
    print(f"{'='*60}")

    le = LabelEncoder()
    y_tr_f = le.fit_transform(y_train_finding.fillna("NONE"))
    y_va_f = le.transform(y_val_finding.fillna("NONE"))
    y_te_f = le.transform(y_test_finding.fillna("NONE"))

    best_f1 = 0
    best_C = 1.0
    best_model_f = None

    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_tr_f)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_va_f, y_val_pred, average="macro", zero_division=0)
        print(f"  C={C:<6} val macro-F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model_f = clf

    print(f"\nBest C={best_C} (val macro-F1={best_f1:.4f})")

    y_test_pred_f = best_model_f.predict(X_test)
    from sklearn.metrics import accuracy_score
    finding_acc = accuracy_score(y_te_f, y_test_pred_f)
    print(f"Test accuracy: {finding_acc:.4f}")
    print(classification_report(
        y_te_f, y_test_pred_f,
        target_names=le.classes_,
        zero_division=0,
    ))

    results["finding"] = {
        "best_C": best_C,
        "val_macro_f1": round(best_f1, 4),
        "test_accuracy": round(finding_acc, 4),
    }
    results["_finding_clf"] = best_model_f
    results["_finding_le"] = le

    # 3. Disciplinary classifier
    print(f"\n{'='*60}")
    print(f"DISCIPLINARY CLASSIFIER ({label})")
    print(f"{'='*60}")

    y_tr_d = y_train_disc.astype(int).values
    y_va_d = y_val_disc.astype(int).values
    y_te_d = y_test_disc.astype(int).values

    best_f1 = 0
    best_C = 1.0
    best_model_d = None

    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_tr_d)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_va_d, y_val_pred, average="binary", zero_division=0)
        print(f"  C={C:<5} val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model_d = clf

    print(f"\nBest C={best_C} (val F1={best_f1:.4f})")

    y_test_pred_d = best_model_d.predict(X_test)
    disc_acc = accuracy_score(y_te_d, y_test_pred_d)
    print(f"Test accuracy: {disc_acc:.4f}")
    print(classification_report(
        y_te_d, y_test_pred_d,
        target_names=["non_disciplinary", "disciplinary"],
        zero_division=0,
    ))

    results["disciplinary"] = {
        "best_C": best_C,
        "val_f1": round(best_f1, 4),
        "test_accuracy": round(disc_acc, 4),
    }
    results["_disciplinary_clf"] = best_model_d

    return results


def predict_full_corpus(vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf):
    """Apply expanded classifiers to full corpus."""
    print(f"\n{'='*60}")
    print("FULL CORPUS PREDICTION (EXPANDED MODEL)")
    print(f"{'='*60}")

    records = []
    with open(CORPUS_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    texts = [r["full_text"] for r in records]
    X = vectorizer.transform(texts)

    misconduct_pred = misconduct_clf.predict(X)
    finding_pred = finding_le.inverse_transform(finding_clf.predict(X))
    disciplinary_pred = disciplinary_clf.predict(X)

    output_path = DATA_DIR / "extracted" / "misconduct_predictions_expanded.jsonl"
    predictions = []
    for i, record in enumerate(records):
        pred_labels = [MISCONDUCT_L1_CODES[j] for j in range(len(MISCONDUCT_L1_CODES)) if misconduct_pred[i, j] == 1]
        predictions.append({
            "id": record["id"],
            "predicted_misconduct_L1": pred_labels,
            "predicted_finding": finding_pred[i],
            "predicted_is_disciplinary": bool(disciplinary_pred[i]),
        })

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions for {len(predictions)} decisions saved to {output_path}")
    disc_count = sum(p["predicted_is_disciplinary"] for p in predictions)
    print(f"  Disciplinary: {disc_count} ({disc_count/len(predictions)*100:.1f}%)")

    misconduct_counts = defaultdict(int)
    for p in predictions:
        for label in p["predicted_misconduct_L1"]:
            misconduct_counts[label] += 1
    print("  Misconduct L1 distribution:")
    for code in MISCONDUCT_L1_CODES:
        count = misconduct_counts.get(code, 0)
        print(f"    {code}: {count} ({count / len(predictions) * 100:.1f}%)")


def main():
    print("=" * 70)
    print("EXPANDED GOLD STANDARD: MERGE, RETRAIN, AND COMPARE")
    print("=" * 70)

    # ── Step 1: Load original splits ──
    print("\n[1] Loading original splits...")
    train_orig = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"  Original train: {len(train_orig)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Step 2: Load and merge new annotations ──
    print("\n[2] Loading new LLM annotations...")
    new_annotations = load_new_annotations()
    print(f"  Total new annotations: {len(new_annotations)}")

    if len(new_annotations) == 0:
        print("ERROR: No new annotations found. Run annotation agents first.")
        sys.exit(1)

    # Validate
    print("\n[3] Validating annotations...")
    new_annotations = validate_annotations(new_annotations)

    # Check for duplicates with original training set
    orig_ids = set(train_orig["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    new_annotations = [a for a in new_annotations if a["id"] not in (orig_ids | val_ids | test_ids)]
    print(f"  After removing duplicates with existing splits: {len(new_annotations)}")

    # ── Step 3: Build expanded training set ──
    print("\n[4] Building expanded training set...")
    corpus_texts = load_corpus_texts()

    new_records = []
    missing_text = 0
    for ann in new_annotations:
        if ann["id"] not in corpus_texts:
            missing_text += 1
            continue
        new_records.append({
            "id": ann["id"],
            "full_text": corpus_texts[ann["id"]],
            "is_disciplinary": ann.get("is_disciplinary", False),
            "misconduct_L1": ann.get("misconduct_L1", ["N/A"]),
            "finding": ann.get("finding", "N/A"),
            "outcomes": ann.get("outcomes", ["N/A"]),
            "profession": ann.get("profession", "unknown"),
        })

    if missing_text:
        print(f"  WARNING: {missing_text} decisions had no full_text in corpus")

    new_df = pd.DataFrame(new_records)
    train_expanded = pd.concat([train_orig, new_df], ignore_index=True)
    print(f"  Expanded train: {len(train_expanded)} ({len(train_orig)} original + {len(new_df)} new)")

    # Distribution summary
    print(f"\n  Expanded training set distribution:")
    disc_count = train_expanded["is_disciplinary"].sum()
    print(f"    Disciplinary: {disc_count} ({disc_count/len(train_expanded)*100:.1f}%)")
    all_misc = [code for labels in train_expanded["misconduct_L1"] for code in (labels if isinstance(labels, list) else [labels])]
    misc_counts = Counter(all_misc)
    for code in MISCONDUCT_L1_CODES:
        c = misc_counts.get(code, 0)
        print(f"    {code}: {c} ({c/len(train_expanded)*100:.1f}%)")

    # Save expanded train split
    expanded_split_path = SPLITS_DIR / "train_expanded.jsonl"
    with open(expanded_split_path, "w") as f:
        for _, row in train_expanded.iterrows():
            record = row.to_dict()
            # Ensure lists are proper lists not numpy arrays
            for key in ["misconduct_L1", "outcomes"]:
                if hasattr(record[key], "tolist"):
                    record[key] = record[key].tolist()
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  Saved expanded train to {expanded_split_path}")

    # ── Step 4: Train with ORIGINAL data (baseline) ──
    print("\n\n" + "#" * 70)
    print("# BASELINE: ORIGINAL TRAINING SET (n=210)")
    print("#" * 70)

    vec_orig = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents="unicode",
    )
    X_train_orig = vec_orig.fit_transform(train_orig["full_text"])
    X_val_orig = vec_orig.transform(val_df["full_text"])
    X_test_orig = vec_orig.transform(test_df["full_text"])
    print(f"  Vocab size: {len(vec_orig.vocabulary_)}")

    y_train_misc_orig = binarise_multilabel(train_orig["misconduct_L1"], MISCONDUCT_L1_CODES)
    y_val_misc = binarise_multilabel(val_df["misconduct_L1"], MISCONDUCT_L1_CODES)
    y_test_misc = binarise_multilabel(test_df["misconduct_L1"], MISCONDUCT_L1_CODES)

    results_orig = train_and_evaluate(
        X_train_orig, y_train_misc_orig, train_orig["finding"], train_orig["is_disciplinary"],
        X_val_orig, y_val_misc, val_df["finding"], val_df["is_disciplinary"],
        X_test_orig, y_test_misc, test_df["finding"], test_df["is_disciplinary"],
        label="ORIGINAL (n=210)",
    )

    # ── Step 5: Train with EXPANDED data ──
    print("\n\n" + "#" * 70)
    print(f"# EXPANDED: EXPANDED TRAINING SET (n={len(train_expanded)})")
    print("#" * 70)

    vec_exp = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents="unicode",
    )
    X_train_exp = vec_exp.fit_transform(train_expanded["full_text"])
    X_val_exp = vec_exp.transform(val_df["full_text"])
    X_test_exp = vec_exp.transform(test_df["full_text"])
    print(f"  Vocab size: {len(vec_exp.vocabulary_)}")

    y_train_misc_exp = binarise_multilabel(train_expanded["misconduct_L1"], MISCONDUCT_L1_CODES)

    results_exp = train_and_evaluate(
        X_train_exp, y_train_misc_exp, train_expanded["finding"], train_expanded["is_disciplinary"],
        X_val_exp, y_val_misc, val_df["finding"], val_df["is_disciplinary"],
        X_test_exp, y_test_misc, test_df["finding"], test_df["is_disciplinary"],
        label=f"EXPANDED (n={len(train_expanded)})",
    )

    # ── Step 6: Compare results ──
    print("\n\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs EXPANDED")
    print("=" * 70)

    comparison = {
        "original": {
            "n_train": len(train_orig),
            "misconduct_test_macro_f1": results_orig["misconduct"]["test_macro_f1"],
            "misconduct_test_micro_f1": results_orig["misconduct"]["test_micro_f1"],
            "misconduct_hamming_loss": results_orig["misconduct"]["hamming_loss"],
            "misconduct_per_class": results_orig["misconduct"]["per_class"],
            "finding_test_accuracy": results_orig["finding"]["test_accuracy"],
            "disciplinary_test_accuracy": results_orig["disciplinary"]["test_accuracy"],
        },
        "expanded": {
            "n_train": len(train_expanded),
            "n_new": len(new_df),
            "misconduct_test_macro_f1": results_exp["misconduct"]["test_macro_f1"],
            "misconduct_test_micro_f1": results_exp["misconduct"]["test_micro_f1"],
            "misconduct_hamming_loss": results_exp["misconduct"]["hamming_loss"],
            "misconduct_per_class": results_exp["misconduct"]["per_class"],
            "finding_test_accuracy": results_exp["finding"]["test_accuracy"],
            "disciplinary_test_accuracy": results_exp["disciplinary"]["test_accuracy"],
        },
    }

    # Delta
    delta_macro = comparison["expanded"]["misconduct_test_macro_f1"] - comparison["original"]["misconduct_test_macro_f1"]
    delta_micro = comparison["expanded"]["misconduct_test_micro_f1"] - comparison["original"]["misconduct_test_micro_f1"]
    delta_finding = comparison["expanded"]["finding_test_accuracy"] - comparison["original"]["finding_test_accuracy"]
    delta_disc = comparison["expanded"]["disciplinary_test_accuracy"] - comparison["original"]["disciplinary_test_accuracy"]

    comparison["delta"] = {
        "misconduct_macro_f1": round(delta_macro, 4),
        "misconduct_micro_f1": round(delta_micro, 4),
        "finding_accuracy": round(delta_finding, 4),
        "disciplinary_accuracy": round(delta_disc, 4),
    }

    print(f"\n{'Metric':<35} {'Original':>10} {'Expanded':>10} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Train size':<35} {len(train_orig):>10} {len(train_expanded):>10} {'+' + str(len(new_df)):>10}")
    print(f"{'Misconduct macro-F1':<35} {comparison['original']['misconduct_test_macro_f1']:>10.4f} {comparison['expanded']['misconduct_test_macro_f1']:>10.4f} {delta_macro:>+10.4f}")
    print(f"{'Misconduct micro-F1':<35} {comparison['original']['misconduct_test_micro_f1']:>10.4f} {comparison['expanded']['misconduct_test_micro_f1']:>10.4f} {delta_micro:>+10.4f}")
    print(f"{'Misconduct hamming loss':<35} {comparison['original']['misconduct_hamming_loss']:>10.4f} {comparison['expanded']['misconduct_hamming_loss']:>10.4f} {comparison['expanded']['misconduct_hamming_loss'] - comparison['original']['misconduct_hamming_loss']:>+10.4f}")
    print(f"{'Finding accuracy':<35} {comparison['original']['finding_test_accuracy']:>10.4f} {comparison['expanded']['finding_test_accuracy']:>10.4f} {delta_finding:>+10.4f}")
    print(f"{'Disciplinary accuracy':<35} {comparison['original']['disciplinary_test_accuracy']:>10.4f} {comparison['expanded']['disciplinary_test_accuracy']:>10.4f} {delta_disc:>+10.4f}")

    print(f"\nPer-class misconduct F1:")
    print(f"{'Class':<10} {'Original':>10} {'Expanded':>10} {'Delta':>10}")
    print("-" * 45)
    for code in MISCONDUCT_L1_CODES:
        orig_f1 = comparison["original"]["misconduct_per_class"].get(code, 0)
        exp_f1 = comparison["expanded"]["misconduct_per_class"].get(code, 0)
        delta = exp_f1 - orig_f1
        print(f"{code:<10} {orig_f1:>10.4f} {exp_f1:>10.4f} {delta:>+10.4f}")

    # Save comparison
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "expanded_training_comparison.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {results_path}")

    # ── Step 7: Save expanded models and re-predict corpus ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vec_exp, f)
    with open(MODEL_DIR / "misconduct_clf.pkl", "wb") as f:
        pickle.dump(results_exp["_misconduct_clf"], f)
    with open(MODEL_DIR / "finding_clf.pkl", "wb") as f:
        pickle.dump(results_exp["_finding_clf"], f)
    with open(MODEL_DIR / "finding_le.pkl", "wb") as f:
        pickle.dump(results_exp["_finding_le"], f)
    with open(MODEL_DIR / "disciplinary_clf.pkl", "wb") as f:
        pickle.dump(results_exp["_disciplinary_clf"], f)
    print(f"\nExpanded models saved to {MODEL_DIR}")

    # Predict full corpus with expanded model
    predict_full_corpus(
        vec_exp,
        results_exp["_misconduct_clf"],
        results_exp["_finding_clf"],
        results_exp["_finding_le"],
        results_exp["_disciplinary_clf"],
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
