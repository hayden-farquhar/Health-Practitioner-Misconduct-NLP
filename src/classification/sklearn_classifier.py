"""sklearn baseline classifiers for misconduct, finding, and is_disciplinary.

Uses TF-IDF + LogisticRegression for:
1. Multi-label misconduct classification (OneVsRest)
2. Single-label finding classification
3. Binary is_disciplinary classification

Trains on 210 decisions, validates on 45, tests on 45.
Predicts on full corpus (3,586 decisions).

Usage:
    python -m src.classification.sklearn_classifier
"""

import json
import pickle
import sys
from collections import defaultdict
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
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "annotations" / "splits"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
MODEL_DIR = PROJECT_ROOT / "models" / "sklearn_misconduct"

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


def binarise_multilabel(series: pd.Series, label_set: list[str]) -> np.ndarray:
    """Convert a Series of label lists into a binary numpy matrix."""
    matrix = np.zeros((len(series), len(label_set)), dtype=int)
    for i, labels in enumerate(series):
        if labels is None:
            continue
        # Handle string representation of lists from JSONL
        if isinstance(labels, str):
            labels = json.loads(labels.replace("'", '"'))
        for label in labels:
            if label in label_set:
                matrix[i, label_set.index(label)] = 1
    return matrix


def train_misconduct_classifier(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple:
    """Train and evaluate multi-label misconduct classifier."""
    print("\n" + "=" * 60)
    print("MULTI-LABEL MISCONDUCT CLASSIFIER (L1)")
    print("=" * 60)

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
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"  C={C:<6} val macro-F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model = clf

    print(f"\nBest C={best_C} (val macro-F1={best_f1:.4f})")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    print("\nTest set results:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=MISCONDUCT_L1_CODES,
        zero_division=0,
    ))

    hl = hamming_loss(y_test, y_test_pred)
    print(f"Hamming loss: {hl:.4f}")

    return best_model, best_C


def train_finding_classifier(
    X_train: np.ndarray, y_train_finding: pd.Series,
    X_val: np.ndarray, y_val_finding: pd.Series,
    X_test: np.ndarray, y_test_finding: pd.Series,
) -> tuple:
    """Train and evaluate single-label finding classifier."""
    print("\n" + "=" * 60)
    print("SINGLE-LABEL FINDING CLASSIFIER")
    print("=" * 60)

    # Encode labels
    le = LabelEncoder()
    # Fill None/nan with "NONE"
    y_train = le.fit_transform(y_train_finding.fillna("NONE"))
    y_val = le.transform(y_val_finding.fillna("NONE"))
    y_test = le.transform(y_test_finding.fillna("NONE"))

    best_f1 = 0
    best_C = 1.0
    best_model = None

    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"  C={C:<6} val macro-F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model = clf

    print(f"\nBest C={best_C} (val macro-F1={best_f1:.4f})")

    y_test_pred = best_model.predict(X_test)
    print("\nTest set results:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    return best_model, le


def train_disciplinary_classifier(
    X_train: np.ndarray, y_train_disc: pd.Series,
    X_val: np.ndarray, y_val_disc: pd.Series,
    X_test: np.ndarray, y_test_disc: pd.Series,
) -> object:
    """Train and evaluate binary is_disciplinary classifier."""
    print("\n" + "=" * 60)
    print("BINARY IS_DISCIPLINARY CLASSIFIER")
    print("=" * 60)

    y_train = y_train_disc.astype(int).values
    y_val = y_val_disc.astype(int).values
    y_test = y_test_disc.astype(int).values

    best_f1 = 0
    best_C = 1.0
    best_model = None

    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
        print(f"  C={C:<5} val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model = clf

    print(f"\nBest C={best_C} (val F1={best_f1:.4f})")

    y_test_pred = best_model.predict(X_test)
    print("\nTest set results:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=["non_disciplinary", "disciplinary"],
        zero_division=0,
    ))

    return best_model


def predict_full_corpus(
    vectorizer: TfidfVectorizer,
    misconduct_clf,
    finding_clf,
    finding_le: LabelEncoder,
    disciplinary_clf,
    corpus_path: Path,
    output_path: Path,
) -> None:
    """Apply classifiers to full corpus and save predictions."""
    print("\n" + "=" * 60)
    print("FULL CORPUS PREDICTION")
    print("=" * 60)

    # Load corpus
    records = []
    with open(corpus_path) as f:
        for line in f:
            records.append(json.loads(line))

    texts = [r["full_text"] for r in records]
    X = vectorizer.transform(texts)

    # Predict
    misconduct_pred = misconduct_clf.predict(X)
    finding_pred = finding_le.inverse_transform(finding_clf.predict(X))
    disciplinary_pred = disciplinary_clf.predict(X)

    # Save predictions
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

    # Summary
    print(f"\nPredictions for {len(predictions)} decisions saved to {output_path}")
    print(f"  Disciplinary: {sum(p['predicted_is_disciplinary'] for p in predictions)}")

    misconduct_counts = defaultdict(int)
    for p in predictions:
        for label in p["predicted_misconduct_L1"]:
            misconduct_counts[label] += 1
    print("  Misconduct L1 distribution:")
    for code in MISCONDUCT_L1_CODES:
        count = misconduct_counts.get(code, 0)
        print(f"    {code}: {count} ({count / len(predictions) * 100:.1f}%)")

    finding_counts = defaultdict(int)
    for p in predictions:
        finding_counts[p["predicted_finding"]] += 1
    print("  Finding distribution:")
    for finding, count in sorted(finding_counts.items(), key=lambda x: -x[1]):
        print(f"    {finding}: {count} ({count / len(predictions) * 100:.1f}%)")


def main():
    print("=== sklearn Classification Pipeline ===\n")

    # Load splits
    print("Loading splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # TF-IDF vectorisation
    # With only 210 training examples, limit features to avoid curse of dimensionality
    print("\nFitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_df["full_text"])
    X_val = vectorizer.transform(val_df["full_text"])
    X_test = vectorizer.transform(test_df["full_text"])
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix shape: {X_train.shape}")

    # Binarise misconduct labels
    y_train_misconduct = binarise_multilabel(train_df["misconduct_L1"], MISCONDUCT_L1_CODES)
    y_val_misconduct = binarise_multilabel(val_df["misconduct_L1"], MISCONDUCT_L1_CODES)
    y_test_misconduct = binarise_multilabel(test_df["misconduct_L1"], MISCONDUCT_L1_CODES)

    # Train classifiers
    misconduct_clf, best_C = train_misconduct_classifier(
        X_train, y_train_misconduct,
        X_val, y_val_misconduct,
        X_test, y_test_misconduct,
    )

    finding_clf, finding_le = train_finding_classifier(
        X_train, train_df["finding"],
        X_val, val_df["finding"],
        X_test, test_df["finding"],
    )

    disciplinary_clf = train_disciplinary_classifier(
        X_train, train_df["is_disciplinary"],
        X_val, val_df["is_disciplinary"],
        X_test, test_df["is_disciplinary"],
    )

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_DIR / "misconduct_clf.pkl", "wb") as f:
        pickle.dump(misconduct_clf, f)
    with open(MODEL_DIR / "finding_clf.pkl", "wb") as f:
        pickle.dump(finding_clf, f)
    with open(MODEL_DIR / "finding_le.pkl", "wb") as f:
        pickle.dump(finding_le, f)
    with open(MODEL_DIR / "disciplinary_clf.pkl", "wb") as f:
        pickle.dump(disciplinary_clf, f)
    print(f"\nModels saved to {MODEL_DIR}")

    # Predict on full corpus
    output_path = DATA_DIR / "extracted" / "misconduct_predictions_sklearn.jsonl"
    predict_full_corpus(
        vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
        CORPUS_PATH, output_path,
    )


if __name__ == "__main__":
    main()
