"""Phase 5: Validation and sensitivity analysis.

Evaluates classifier performance on the held-out test set, conducts
error analysis, and runs sensitivity analyses (appeal exclusion,
temporal split, threshold tuning, per-profession evaluation).

Usage:
    .venv-ml/bin/python -m src.analysis.validation_analysis
"""

import json
import pickle
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "annotations" / "splits"
GOLD_PATH = DATA_DIR / "annotations" / "gold_standard_2026-11.json"
CORPUS_PATH = DATA_DIR / "extracted" / "enriched_corpus.jsonl"
MODEL_DIR = PROJECT_ROOT / "models" / "sklearn_misconduct"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"

for d in (FIGURES_DIR, TABLES_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISCONDUCT_CODES = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF", "N/A"]
MISCONDUCT_LABELS = {
    "CLIN": "Clinical competence",
    "PRESC": "Prescribing",
    "BOUND": "Boundary violations",
    "IMPR": "Impairment",
    "FRAUD": "Dishonesty/fraud",
    "COMM": "Communication",
    "PROF": "Professional conduct",
    "N/A": "N/A",
}
FINDING_CODES = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]
OUTCOME_CODES = [
    "CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
    "FINE", "CAUTION", "UNDERTAKING", "NO_ADVERSE",
]

PALETTE = sns.color_palette("colorblind")
sns.set_theme(
    style="whitegrid", context="paper", font_scale=1.1,
    rc={"figure.dpi": 300, "savefig.dpi": 300, "font.family": "sans-serif",
        "figure.facecolor": "white", "savefig.facecolor": "white"},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def binarise(series: pd.Series, label_set: list[str]) -> np.ndarray:
    """Convert Series of label lists → binary matrix."""
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


def save_fig(fig, name: str):
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {name}")


def save_table(df, name: str):
    df.to_csv(TABLES_DIR / name)
    print(f"    Saved {name}")


def load_split(name: str) -> pd.DataFrame:
    records = []
    with open(SPLITS_DIR / f"{name}.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_models():
    """Load saved sklearn models (supports hybrid vectorizer setup)."""
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_DIR / "misconduct_clf.pkl", "rb") as f:
        misconduct_clf = pickle.load(f)
    with open(MODEL_DIR / "finding_clf.pkl", "rb") as f:
        finding_clf = pickle.load(f)
    with open(MODEL_DIR / "finding_le.pkl", "rb") as f:
        finding_le = pickle.load(f)
    with open(MODEL_DIR / "disciplinary_clf.pkl", "rb") as f:
        disciplinary_clf = pickle.load(f)
    # Load separate finding vectorizer if available (hybrid model)
    finding_vectorizer_path = MODEL_DIR / "vectorizer_finding.pkl"
    if finding_vectorizer_path.exists():
        with open(finding_vectorizer_path, "rb") as f:
            vectorizer_finding = pickle.load(f)
    else:
        vectorizer_finding = vectorizer
    return vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf, vectorizer_finding


# ---------------------------------------------------------------------------
# 5.1 Classification Validation
# ---------------------------------------------------------------------------
def classification_validation(
    test_df: pd.DataFrame,
    vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
    val_df: pd.DataFrame = None, vectorizer_finding=None,
) -> dict:
    """Full classification reports and confusion matrices on test set."""
    print("\n" + "=" * 60)
    print("5.1 CLASSIFICATION VALIDATION")
    print("=" * 60)

    if vectorizer_finding is None:
        vectorizer_finding = vectorizer

    results = {}

    X_test = vectorizer.transform(test_df["full_text"])
    X_test_finding = vectorizer_finding.transform(test_df["full_text"])

    # --- 5.1.1 Is-disciplinary ---
    print("\n--- Is-Disciplinary ---")
    y_true_disc = test_df["is_disciplinary"].astype(int).values
    y_pred_disc = disciplinary_clf.predict(X_test)

    disc_report = classification_report(
        y_true_disc, y_pred_disc,
        target_names=["Non-disciplinary", "Disciplinary"],
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_disc, y_pred_disc,
        target_names=["Non-disciplinary", "Disciplinary"],
        zero_division=0,
    ))

    cm_disc = confusion_matrix(y_true_disc, y_pred_disc)
    results["is_disciplinary"] = {
        "accuracy": round(accuracy_score(y_true_disc, y_pred_disc), 4),
        "f1_disciplinary": round(disc_report["Disciplinary"]["f1-score"], 4),
        "f1_non_disciplinary": round(disc_report["Non-disciplinary"]["f1-score"], 4),
        "confusion_matrix": cm_disc.tolist(),
        "n_test": len(y_true_disc),
    }

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_disc, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-disc.", "Disciplinary"],
                yticklabels=["Non-disc.", "Disciplinary"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Is-Disciplinary Confusion Matrix (Test Set)")
    save_fig(fig, "fig13_cm_disciplinary.png")

    # --- 5.1.2 Multi-label misconduct ---
    print("\n--- Multi-Label Misconduct (L1) ---")
    y_true_mc = binarise(test_df["misconduct_L1"], MISCONDUCT_CODES)
    y_pred_mc = misconduct_clf.predict(X_test)

    mc_report = classification_report(
        y_true_mc, y_pred_mc,
        target_names=MISCONDUCT_CODES,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_mc, y_pred_mc,
        target_names=MISCONDUCT_CODES,
        zero_division=0,
    ))

    hl = hamming_loss(y_true_mc, y_pred_mc)
    macro_f1 = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_mc, y_pred_mc, average="micro", zero_division=0)
    print(f"  Hamming loss: {hl:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Micro F1: {micro_f1:.4f}")

    results["misconduct"] = {
        "hamming_loss": round(hl, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": {
            code: {
                "precision": round(mc_report[code]["precision"], 4),
                "recall": round(mc_report[code]["recall"], 4),
                "f1": round(mc_report[code]["f1-score"], 4),
                "support": int(mc_report[code]["support"]),
            }
            for code in MISCONDUCT_CODES
        },
    }

    # Per-class confusion matrices for top misconduct types
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, code in enumerate(MISCONDUCT_CODES):
        ax = axes[idx // 4, idx % 4]
        cm = confusion_matrix(y_true_mc[:, idx], y_pred_mc[:, idx])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        f1 = mc_report[code]["f1-score"]
        ax.set_title(f"{code} (F1={f1:.2f})", fontsize=10)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
    plt.suptitle("Misconduct L1 Confusion Matrices (Test Set)", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "fig14_cm_misconduct.png")

    # --- 5.1.3 Finding ---
    print("\n--- Finding ---")
    y_true_find = test_df["finding"].fillna("NONE").values
    y_pred_find_enc = finding_clf.predict(X_test_finding)
    y_pred_find = finding_le.inverse_transform(y_pred_find_enc)

    # Align labels
    all_labels = sorted(set(y_true_find) | set(y_pred_find))
    find_report = classification_report(
        y_true_find, y_pred_find,
        labels=all_labels,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_find, y_pred_find,
        labels=all_labels,
        zero_division=0,
    ))

    cm_find = confusion_matrix(y_true_find, y_pred_find, labels=all_labels)
    results["finding"] = {
        "accuracy": round(accuracy_score(y_true_find, y_pred_find), 4),
        "macro_f1": round(f1_score(y_true_find, y_pred_find, average="macro",
                                    labels=all_labels, zero_division=0), 4),
        "per_class": {
            label: {
                "precision": round(find_report[label]["precision"], 4),
                "recall": round(find_report[label]["recall"], 4),
                "f1": round(find_report[label]["f1-score"], 4),
                "support": int(find_report[label]["support"]),
            }
            for label in all_labels if label in find_report
        },
        "confusion_matrix": cm_find.tolist(),
        "labels": all_labels,
    }

    # Finding confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_find, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=all_labels, yticklabels=all_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Finding Confusion Matrix (Test Set)")
    save_fig(fig, "fig15_cm_finding.png")

    # --- 5.1.4 Outcome extraction (regex) ---
    print("\n--- Outcome Extraction (Regex) ---")
    # Load enriched corpus to get predicted_outcomes for test IDs
    enriched = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            enriched[rec["id"]] = rec

    y_true_oc = binarise(test_df["outcomes"], OUTCOME_CODES)
    # Get predicted outcomes from enriched corpus
    pred_outcomes = []
    for _, row in test_df.iterrows():
        erec = enriched.get(row["id"], {})
        pred_outcomes.append(erec.get("predicted_outcomes", []))
    y_pred_oc = binarise(pd.Series(pred_outcomes), OUTCOME_CODES)

    oc_report = classification_report(
        y_true_oc, y_pred_oc,
        target_names=OUTCOME_CODES,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_oc, y_pred_oc,
        target_names=OUTCOME_CODES,
        zero_division=0,
    ))

    oc_micro = f1_score(y_true_oc, y_pred_oc, average="micro", zero_division=0)
    oc_macro = f1_score(y_true_oc, y_pred_oc, average="macro", zero_division=0)
    print(f"  Micro F1: {oc_micro:.4f}")
    print(f"  Macro F1: {oc_macro:.4f}")

    results["outcome_extraction"] = {
        "micro_f1": round(oc_micro, 4),
        "macro_f1": round(oc_macro, 4),
        "per_class": {
            code: {
                "precision": round(oc_report[code]["precision"], 4),
                "recall": round(oc_report[code]["recall"], 4),
                "f1": round(oc_report[code]["f1-score"], 4),
                "support": int(oc_report[code]["support"]),
            }
            for code in OUTCOME_CODES
        },
    }

    # --- Summary table ---
    summary_rows = [
        {"Task": "Is-disciplinary (binary)", "Metric": "Accuracy",
         "Value": results["is_disciplinary"]["accuracy"]},
        {"Task": "Is-disciplinary (binary)", "Metric": "F1 (disciplinary)",
         "Value": results["is_disciplinary"]["f1_disciplinary"]},
        {"Task": "Misconduct L1 (multi-label)", "Metric": "Macro F1",
         "Value": results["misconduct"]["macro_f1"]},
        {"Task": "Misconduct L1 (multi-label)", "Metric": "Micro F1",
         "Value": results["misconduct"]["micro_f1"]},
        {"Task": "Misconduct L1 (multi-label)", "Metric": "Hamming loss",
         "Value": results["misconduct"]["hamming_loss"]},
        {"Task": "Finding (single-label)", "Metric": "Accuracy",
         "Value": results["finding"]["accuracy"]},
        {"Task": "Finding (single-label)", "Metric": "Macro F1",
         "Value": results["finding"]["macro_f1"]},
        {"Task": "Outcome extraction (regex)", "Metric": "Micro F1",
         "Value": results["outcome_extraction"]["micro_f1"]},
        {"Task": "Outcome extraction (regex)", "Metric": "Macro F1",
         "Value": results["outcome_extraction"]["macro_f1"]},
    ]
    # Add per-class misconduct F1
    for code in MISCONDUCT_CODES:
        if code == "N/A":
            continue
        summary_rows.append({
            "Task": f"Misconduct: {code}",
            "Metric": "F1",
            "Value": results["misconduct"]["per_class"][code]["f1"],
        })

    t_summary = pd.DataFrame(summary_rows)
    save_table(t_summary, "table19_classification_performance.csv")

    return results


# ---------------------------------------------------------------------------
# 5.2 Error Analysis
# ---------------------------------------------------------------------------
def error_analysis(
    test_df: pd.DataFrame,
    vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
    vectorizer_finding=None,
) -> dict:
    """Identify and categorise misclassified decisions."""
    print("\n" + "=" * 60)
    print("5.2 ERROR ANALYSIS")
    print("=" * 60)

    if vectorizer_finding is None:
        vectorizer_finding = vectorizer

    X_test = vectorizer.transform(test_df["full_text"])
    X_test_finding = vectorizer_finding.transform(test_df["full_text"])

    # Get predictions
    y_pred_disc = disciplinary_clf.predict(X_test)
    y_pred_mc = misconduct_clf.predict(X_test)
    y_pred_find = finding_le.inverse_transform(finding_clf.predict(X_test_finding))

    # Load enriched corpus for outcome predictions
    enriched = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            enriched[rec["id"]] = rec

    y_true_mc = binarise(test_df["misconduct_L1"], MISCONDUCT_CODES)

    errors = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        doc_id = row["id"]
        err = {"id": doc_id, "error_types": []}

        # Is-disciplinary errors
        true_disc = int(row["is_disciplinary"])
        pred_disc = int(y_pred_disc[i])
        if true_disc != pred_disc:
            err["error_types"].append("is_disciplinary")
            err["disc_true"] = bool(true_disc)
            err["disc_pred"] = bool(pred_disc)

        # Misconduct errors
        true_mc = set(row["misconduct_L1"]) if row["misconduct_L1"] else set()
        pred_mc = set(
            MISCONDUCT_CODES[j]
            for j in range(len(MISCONDUCT_CODES))
            if y_pred_mc[i, j] == 1
        )
        if true_mc != pred_mc:
            err["error_types"].append("misconduct")
            err["mc_true"] = sorted(true_mc)
            err["mc_pred"] = sorted(pred_mc)
            err["mc_false_pos"] = sorted(pred_mc - true_mc)
            err["mc_false_neg"] = sorted(true_mc - pred_mc)

        # Finding errors
        true_find = row["finding"] if pd.notna(row["finding"]) else "NONE"
        pred_find = y_pred_find[i]
        if true_find != pred_find:
            err["error_types"].append("finding")
            err["find_true"] = true_find
            err["find_pred"] = pred_find

        # Outcome errors
        true_oc = set(row["outcomes"]) if row["outcomes"] else set()
        erec = enriched.get(doc_id, {})
        pred_oc = set(erec.get("predicted_outcomes", []))
        true_oc_filtered = true_oc & set(OUTCOME_CODES)
        pred_oc_filtered = pred_oc & set(OUTCOME_CODES)
        if true_oc_filtered != pred_oc_filtered:
            err["error_types"].append("outcome")
            err["oc_true"] = sorted(true_oc_filtered)
            err["oc_pred"] = sorted(pred_oc_filtered)
            err["oc_false_pos"] = sorted(pred_oc_filtered - true_oc_filtered)
            err["oc_false_neg"] = sorted(true_oc_filtered - pred_oc_filtered)

        # Add metadata
        err["profession"] = row.get("profession")
        err["text_length"] = len(row.get("full_text", ""))

        if err["error_types"]:
            errors.append(err)

    print(f"\n  {len(errors)}/{len(test_df)} test decisions have at least one error")

    # Categorise error patterns
    error_type_counts = Counter()
    mc_fp_counts = Counter()
    mc_fn_counts = Counter()
    oc_fp_counts = Counter()
    oc_fn_counts = Counter()

    for err in errors:
        for et in err["error_types"]:
            error_type_counts[et] += 1
        for fp in err.get("mc_false_pos", []):
            mc_fp_counts[fp] += 1
        for fn in err.get("mc_false_neg", []):
            mc_fn_counts[fn] += 1
        for fp in err.get("oc_false_pos", []):
            oc_fp_counts[fp] += 1
        for fn in err.get("oc_false_neg", []):
            oc_fn_counts[fn] += 1

    print("\n  Error type frequency:")
    for et, cnt in error_type_counts.most_common():
        print(f"    {et}: {cnt}/{len(test_df)} ({cnt/len(test_df)*100:.0f}%)")

    print("\n  Misconduct false positives (over-predicted):")
    for code, cnt in mc_fp_counts.most_common():
        print(f"    {code}: {cnt}")

    print("\n  Misconduct false negatives (missed):")
    for code, cnt in mc_fn_counts.most_common():
        print(f"    {code}: {cnt}")

    print("\n  Outcome false positives:")
    for code, cnt in oc_fp_counts.most_common():
        print(f"    {code}: {cnt}")

    print("\n  Outcome false negatives:")
    for code, cnt in oc_fn_counts.most_common():
        print(f"    {code}: {cnt}")

    # Error analysis by text length
    error_ids = {e["id"] for e in errors}
    test_df_copy = test_df.copy()
    test_df_copy["has_error"] = test_df_copy["id"].isin(error_ids)
    test_df_copy["text_length"] = test_df_copy["full_text"].str.len()

    err_len = test_df_copy[test_df_copy["has_error"]]["text_length"]
    ok_len = test_df_copy[~test_df_copy["has_error"]]["text_length"]
    if len(err_len) > 0 and len(ok_len) > 0:
        u_stat, u_p = stats.mannwhitneyu(err_len, ok_len, alternative="two-sided")
        print(f"\n  Text length: errors median={err_len.median():.0f}, "
              f"correct median={ok_len.median():.0f} (Mann-Whitney p={u_p:.4f})")

    # Table: error details
    error_rows = []
    for err in errors:
        error_rows.append({
            "id": err["id"],
            "profession": err.get("profession", ""),
            "error_types": ", ".join(err["error_types"]),
            "mc_true": ", ".join(err.get("mc_true", [])),
            "mc_pred": ", ".join(err.get("mc_pred", [])),
            "mc_false_pos": ", ".join(err.get("mc_false_pos", [])),
            "mc_false_neg": ", ".join(err.get("mc_false_neg", [])),
            "find_true": err.get("find_true", ""),
            "find_pred": err.get("find_pred", ""),
            "oc_true": ", ".join(err.get("oc_true", [])),
            "oc_pred": ", ".join(err.get("oc_pred", [])),
            "text_length": err.get("text_length", 0),
        })
    t_errors = pd.DataFrame(error_rows)
    save_table(t_errors, "table20_error_analysis.csv")

    # Summary
    error_summary = {
        "total_test": len(test_df),
        "total_with_errors": len(errors),
        "error_rate": round(len(errors) / len(test_df) * 100, 1),
        "error_type_counts": dict(error_type_counts),
        "misconduct_false_positives": dict(mc_fp_counts),
        "misconduct_false_negatives": dict(mc_fn_counts),
        "outcome_false_positives": dict(oc_fp_counts),
        "outcome_false_negatives": dict(oc_fn_counts),
    }

    return error_summary


# ---------------------------------------------------------------------------
# 5.3 Sensitivity Analyses
# ---------------------------------------------------------------------------
def sensitivity_analyses(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
    vectorizer_finding=None,
) -> dict:
    """Run sensitivity analyses on the full corpus."""
    print("\n" + "=" * 60)
    print("5.3 SENSITIVITY ANALYSES")
    print("=" * 60)

    sensitivity_results = {}

    # Load enriched corpus
    records = []
    with open(CORPUS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    corpus_df = pd.DataFrame(records)
    corpus_df["year"] = pd.to_numeric(corpus_df["date"].str[:4], errors="coerce")

    disc = corpus_df[corpus_df["predicted_is_disciplinary"]].copy()

    # --- 5.3.1 Exclude appeals ---
    print("\n--- 5.3.1 Exclude Appeals ---")
    # Identify appeals by keywords in text or citation
    appeal_pattern = re.compile(
        r"\b(?:appeal|appellant|appellants?)\b.*\b(?:decision|tribunal|orders?)\b",
        re.IGNORECASE,
    )
    # Also check citation patterns like "NSWCATAP" (appeal panel)
    citation_appeal = corpus_df["tribunal"].str.contains("AP$|appeal", case=False, na=False)
    text_appeal = corpus_df["full_text"].str[:5000].str.contains(
        r"\bappeal\b", case=False, na=False
    )
    corpus_df["is_appeal"] = citation_appeal | text_appeal
    n_appeals = corpus_df["is_appeal"].sum()
    print(f"    Identified {n_appeals} likely appeal decisions ({n_appeals/len(corpus_df)*100:.1f}%)")

    # Compare key statistics with and without appeals
    disc_no_appeal = corpus_df[
        corpus_df["predicted_is_disciplinary"] & ~corpus_df["is_appeal"]
    ].copy()

    # Misconduct distribution comparison
    mc_with = {}
    mc_without = {}
    for mc in ["CLIN", "PRESC", "BOUND", "PROF", "FRAUD", "IMPR", "COMM"]:
        col = corpus_df["predicted_misconduct_L1"].apply(lambda x: mc in x)
        mc_with[mc] = round(col[corpus_df["predicted_is_disciplinary"]].mean() * 100, 1)
        mc_without[mc] = round(col[corpus_df["predicted_is_disciplinary"] & ~corpus_df["is_appeal"]].mean() * 100, 1)

    appeal_rows = []
    for mc in mc_with:
        appeal_rows.append({
            "Misconduct type": MISCONDUCT_LABELS.get(mc, mc),
            "With appeals (%)": mc_with[mc],
            "Without appeals (%)": mc_without[mc],
            "Difference (pp)": round(mc_without[mc] - mc_with[mc], 1),
        })
    t_appeals = pd.DataFrame(appeal_rows)
    save_table(t_appeals, "table21_sensitivity_appeals.csv")

    sensitivity_results["appeals"] = {
        "n_appeals": int(n_appeals),
        "pct_appeals": round(n_appeals / len(corpus_df) * 100, 1),
        "n_disc_with": int(disc["predicted_is_disciplinary"].sum()) if "predicted_is_disciplinary" in disc.columns else len(disc),
        "n_disc_without": len(disc_no_appeal),
        "misconduct_pct_with": mc_with,
        "misconduct_pct_without": mc_without,
    }
    print(f"    Disciplinary decisions: {len(disc)} → {len(disc_no_appeal)} after excluding appeals")

    # --- 5.3.2 Pre-2015 vs Post-2015 temporal split ---
    print("\n--- 5.3.2 Pre-2015 vs Post-2015 Split ---")
    disc_with_year = disc[disc["year"].notna()].copy()
    pre_2015 = disc_with_year[disc_with_year["year"] < 2015]
    post_2015 = disc_with_year[disc_with_year["year"] >= 2015]

    temporal_rows = []
    for mc in ["CLIN", "PRESC", "BOUND", "PROF", "FRAUD", "IMPR", "COMM"]:
        col_pre = pre_2015["predicted_misconduct_L1"].apply(lambda x: mc in x)
        col_post = post_2015["predicted_misconduct_L1"].apply(lambda x: mc in x)
        pct_pre = round(col_pre.mean() * 100, 1)
        pct_post = round(col_post.mean() * 100, 1)

        # Chi-square for pre vs post
        ct = pd.DataFrame({
            "present": [int(col_pre.sum()), int(col_post.sum())],
            "absent": [int((~col_pre).sum()), int((~col_post).sum())],
        }, index=["Pre-2015", "Post-2015"])
        if ct.values.min() >= 0 and ct.values.sum() > 0:
            chi2, p, _, _ = stats.chi2_contingency(ct.values)
        else:
            chi2, p = 0, 1.0

        temporal_rows.append({
            "Misconduct type": MISCONDUCT_LABELS.get(mc, mc),
            "Pre-2015 N": int(col_pre.sum()),
            "Pre-2015 %": pct_pre,
            "Post-2015 N": int(col_post.sum()),
            "Post-2015 %": pct_post,
            "Chi2": round(chi2, 2),
            "p-value": float(f"{p:.4g}"),
        })

    t_temporal = pd.DataFrame(temporal_rows)
    save_table(t_temporal, "table22_sensitivity_temporal_split.csv")

    # Outcome comparison
    outcome_temporal = []
    for oc in OUTCOME_CODES:
        col_pre = pre_2015["predicted_outcomes"].apply(lambda x: oc in x)
        col_post = post_2015["predicted_outcomes"].apply(lambda x: oc in x)
        outcome_temporal.append({
            "Outcome": oc,
            "Pre-2015 %": round(col_pre.mean() * 100, 1),
            "Post-2015 %": round(col_post.mean() * 100, 1),
        })
    t_oc_temporal = pd.DataFrame(outcome_temporal)
    save_table(t_oc_temporal, "table23_sensitivity_outcome_temporal.csv")

    sensitivity_results["temporal_split"] = {
        "n_pre_2015": len(pre_2015),
        "n_post_2015": len(post_2015),
        "misconduct_comparison": temporal_rows,
    }
    print(f"    Pre-2015: {len(pre_2015)}, Post-2015: {len(post_2015)}")

    # --- 5.3.3 Alternative classification thresholds ---
    print("\n--- 5.3.3 Alternative Classification Thresholds ---")
    # Get decision function scores for misconduct
    X_test = vectorizer.transform(test_df["full_text"])
    y_true_mc = binarise(test_df["misconduct_L1"], MISCONDUCT_CODES)

    # Get probability scores
    try:
        y_scores = misconduct_clf.decision_function(X_test)
    except AttributeError:
        y_scores = misconduct_clf.predict_proba(X_test)

    # Test different thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    threshold_rows = []
    for thresh in thresholds:
        y_pred_t = (y_scores >= thresh).astype(int)
        macro = f1_score(y_true_mc, y_pred_t, average="macro", zero_division=0)
        micro = f1_score(y_true_mc, y_pred_t, average="micro", zero_division=0)
        hl = hamming_loss(y_true_mc, y_pred_t)
        threshold_rows.append({
            "Threshold": thresh,
            "Macro F1": round(macro, 4),
            "Micro F1": round(micro, 4),
            "Hamming Loss": round(hl, 4),
        })
        print(f"    threshold={thresh:.1f}: macro-F1={macro:.4f}, micro-F1={micro:.4f}, HL={hl:.4f}")

    t_thresholds = pd.DataFrame(threshold_rows)
    save_table(t_thresholds, "table24_sensitivity_thresholds.csv")

    sensitivity_results["thresholds"] = threshold_rows

    # Figure: threshold effect
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot([r["Threshold"] for r in threshold_rows],
             [r["Macro F1"] for r in threshold_rows],
             "o-", color=PALETTE[0], label="Macro F1")
    ax1.plot([r["Threshold"] for r in threshold_rows],
             [r["Micro F1"] for r in threshold_rows],
             "s-", color=PALETTE[1], label="Micro F1")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Classification threshold sensitivity (Test set)")
    ax2 = ax1.twinx()
    ax2.plot([r["Threshold"] for r in threshold_rows],
             [r["Hamming Loss"] for r in threshold_rows],
             "^--", color=PALETTE[2], label="Hamming Loss")
    ax2.set_ylabel("Hamming Loss")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    save_fig(fig, "fig16_threshold_sensitivity.png")

    # --- 5.3.4 Per-profession evaluation ---
    print("\n--- 5.3.4 Per-Profession Evaluation ---")
    # Evaluate misconduct classifier performance by profession on test set
    test_profs = test_df["profession"].value_counts()
    prof_rows = []
    for prof, count in test_profs.items():
        if pd.isna(prof) or count < 3:
            continue
        mask = test_df["profession"] == prof
        idx = mask.values
        if idx.sum() == 0:
            continue

        y_true_p = y_true_mc[idx]
        y_pred_p = misconduct_clf.predict(X_test[idx])
        macro = f1_score(y_true_p, y_pred_p, average="macro", zero_division=0)
        micro = f1_score(y_true_p, y_pred_p, average="micro", zero_division=0)
        prof_rows.append({
            "Profession": prof,
            "N (test)": int(count),
            "Macro F1": round(macro, 4),
            "Micro F1": round(micro, 4),
        })
        print(f"    {prof} (n={count}): macro-F1={macro:.4f}, micro-F1={micro:.4f}")

    if prof_rows:
        t_prof = pd.DataFrame(prof_rows)
        save_table(t_prof, "table25_sensitivity_profession.csv")
        sensitivity_results["per_profession"] = prof_rows

    # --- 5.3.5 Corpus-level stability: bootstrap confidence intervals ---
    print("\n--- 5.3.5 Bootstrap Confidence Intervals (Full Corpus) ---")
    # Bootstrap the key prevalence estimates from Phase 4
    n_boot = 1000
    rng = np.random.default_rng(42)

    boot_stats = {}
    for mc in ["CLIN", "PRESC", "BOUND", "PROF", "FRAUD", "IMPR", "COMM"]:
        col = disc["predicted_misconduct_L1"].apply(lambda x: mc in x).values
        boot_pcts = []
        for _ in range(n_boot):
            sample = rng.choice(col, size=len(col), replace=True)
            boot_pcts.append(sample.mean() * 100)
        ci_lo = np.percentile(boot_pcts, 2.5)
        ci_hi = np.percentile(boot_pcts, 97.5)
        point = col.mean() * 100
        boot_stats[mc] = {
            "point_estimate": round(point, 1),
            "ci_95_lower": round(ci_lo, 1),
            "ci_95_upper": round(ci_hi, 1),
        }
        print(f"    {mc}: {point:.1f}% (95% CI: {ci_lo:.1f}–{ci_hi:.1f}%)")

    t_boot = pd.DataFrame([
        {
            "Misconduct type": MISCONDUCT_LABELS.get(mc, mc),
            "Prevalence %": v["point_estimate"],
            "95% CI lower": v["ci_95_lower"],
            "95% CI upper": v["ci_95_upper"],
        }
        for mc, v in boot_stats.items()
    ])
    save_table(t_boot, "table26_bootstrap_ci.csv")

    sensitivity_results["bootstrap_ci"] = boot_stats

    return sensitivity_results


# ---------------------------------------------------------------------------
# Save all results
# ---------------------------------------------------------------------------
def save_results(classification_results, error_summary, sensitivity_results):
    """Save machine-readable validation results."""
    print("\n--- Saving validation results ---")

    all_results = {
        "classification": classification_results,
        "error_analysis": error_summary,
        "sensitivity": sensitivity_results,
    }

    with open(STATS_DIR / "validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"    Saved validation_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 5: Validation & Sensitivity Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load models
    print("\nLoading sklearn models...")
    vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf, vectorizer_finding = load_models()
    print(f"  Vectorizer vocab: {len(vectorizer.vocabulary_)}")
    print(f"  Finding vectorizer vocab: {len(vectorizer_finding.vocabulary_)}")

    # 5.1 Classification validation
    classification_results = classification_validation(
        test_df, vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
        val_df=val_df, vectorizer_finding=vectorizer_finding,
    )

    # 5.2 Error analysis
    error_summary = error_analysis(
        test_df, vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
        vectorizer_finding=vectorizer_finding,
    )

    # 5.3 Sensitivity analyses
    sensitivity_results = sensitivity_analyses(
        train_df, val_df, test_df,
        vectorizer, misconduct_clf, finding_clf, finding_le, disciplinary_clf,
        vectorizer_finding=vectorizer_finding,
    )

    # Save all results
    save_results(classification_results, error_summary, sensitivity_results)

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    n_tables = len(list(TABLES_DIR.glob("table[12][0-9]*.csv")))
    n_figs = len(list(FIGURES_DIR.glob("fig1[3-9]*.png")))
    print(f"  New tables: {n_tables}")
    print(f"  New figures: {n_figs}")
    print(f"  Key results:")
    print(f"    Is-disciplinary accuracy: {classification_results['is_disciplinary']['accuracy']}")
    print(f"    Misconduct macro-F1: {classification_results['misconduct']['macro_f1']}")
    print(f"    Finding accuracy: {classification_results['finding']['accuracy']}")
    print(f"    Outcome micro-F1: {classification_results['outcome_extraction']['micro_f1']}")
    print(f"    Test errors: {error_summary['total_with_errors']}/{error_summary['total_test']}")


if __name__ == "__main__":
    main()
