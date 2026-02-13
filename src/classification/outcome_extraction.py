"""Validate and improve regex-based outcome extraction against gold standard.

Compares the existing regex-based outcome extraction (from austlii_extractor.py)
against gold-standard human annotations. Reports precision/recall/F1 per outcome
category. Applies improved extraction to the full corpus.

Usage:
    python -m src.classification.outcome_extraction
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOLD_STANDARD_PATH = DATA_DIR / "annotations" / "gold_standard.json"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
OUTPUT_PATH = DATA_DIR / "extracted" / "outcome_predictions.jsonl"

# Mapping from regex outcome names to gold-standard taxonomy codes
REGEX_TO_GOLD = {
    "cancellation": "CANCEL",
    "suspension": "SUSPEND",
    "conditions": "CONDITIONS",
    "reprimand": "REPRIMAND",
    "caution": "CAUTION",
    "fine": "FINE",
    "undertaking": "UNDERTAKING",
    "no_adverse_finding": "NO_ADVERSE",
}

GOLD_TO_REGEX = {v: k for k, v in REGEX_TO_GOLD.items()}

# Improved outcome patterns (refined from austlii_extractor.py originals)
OUTCOME_PATTERNS = {
    "CANCEL": re.compile(
        r"\b(?:cancel(?:led|lation)|registration\s+(?:is|be)\s+cancelled)\b",
        re.IGNORECASE,
    ),
    "SUSPEND": re.compile(
        r"\b(?:suspend(?:ed)?\s+(?:the\s+)?(?:respondent'?s?\s+)?(?:registration|for\s+a\s+period)|"
        r"suspension\s+of\s+(?:the\s+)?(?:respondent'?s?\s+)?registration|"
        r"registration\s+(?:is|be)\s+suspended|"
        r"period\s+of\s+suspension)\b",
        re.IGNORECASE,
    ),
    "CONDITIONS": re.compile(
        r"\b(?:conditions?\s+(?:imposed|placed|be\s+imposed|are\s+imposed)|"
        r"(?:impose|place|imposing)\s+(?:the\s+following\s+)?conditions?|"
        r"subject\s+to\s+(?:the\s+following\s+)?conditions?|"
        r"registration\s+(?:is|be)\s+subject\s+to\s+conditions?|"
        r"practise\s+subject\s+to)\b",
        re.IGNORECASE,
    ),
    "REPRIMAND": re.compile(
        r"\b(?:reprimand(?:ed)?|censure[d]?)\b",
        re.IGNORECASE,
    ),
    "CAUTION": re.compile(
        r"\b(?:(?:is|be)\s+cautioned|"
        r"(?:issue[ds]?\s+(?:a\s+)?|receive[ds]?\s+(?:a\s+)?)caution|"
        r"formal\s+caution)\b",
        re.IGNORECASE,
    ),
    "FINE": re.compile(
        r"\b(?:fine[d]?\s+(?:the\s+sum\s+of\s+)?\$|pay\s+a\s+(?:fine|penalty)\s+of|"
        r"penalty\s+of\s+\$)\b",
        re.IGNORECASE,
    ),
    "UNDERTAKING": re.compile(
        # Narrow pattern — "undertaking" is too common in general legal text
        r"\b(?:(?:give[sn]?|enter(?:ed|s)?\s+into)\s+(?:a[n]?\s+)?undertaking\s+(?:to|not)|"
        r"accept(?:s|ed)?\s+(?:the\s+)?(?:respondent'?s?\s+)?undertaking)\b",
        re.IGNORECASE,
    ),
    "NO_ADVERSE": re.compile(
        r"\b(?:no\s+adverse\s+finding|complaint[s]?\s+(?:is\s+|are\s+)?dismissed|"
        r"application\s+(?:is\s+)?dismissed|"
        r"charges?\s+(?:is\s+|are\s+)?(?:not\s+(?:established|proven|made\s+out)|dismissed)|"
        r"(?:allegation|complaint)[s]?\s+(?:is\s+|are\s+)?not\s+(?:established|proven|made\s+out))\b",
        re.IGNORECASE,
    ),
}


ORDERS_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:ORDERS?|THE\s+ORDERS?|DETERMINATION|THE\s+DETERMINATION|"
    r"DECISION|THE\s+DECISION)\s*\n",
    re.IGNORECASE | re.MULTILINE,
)


def extract_outcomes_improved(text: str) -> list[str]:
    """Extract outcome categories from decision text using improved regex patterns.

    First tries to locate the orders/determination section. Falls back to
    the last 5000 characters if no section heading is found.
    """
    # Try to find the orders section
    matches = list(ORDERS_SECTION_RE.finditer(text))
    if matches:
        # Use text from the last orders heading onward
        search_text = text[matches[-1].start():]
    else:
        # Fallback: use last 5000 chars
        search_text = text[-5000:]

    outcomes = []
    for code, pattern in OUTCOME_PATTERNS.items():
        if pattern.search(search_text):
            outcomes.append(code)

    return outcomes


def evaluate_extraction(gold: list[dict], corpus: dict[str, dict]) -> dict:
    """Evaluate outcome extraction against gold standard.

    Returns per-category and aggregate metrics.
    """
    # Outcome codes to evaluate (excluding N/A and OTHER which regex can't detect)
    eval_codes = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
                  "CAUTION", "FINE", "UNDERTAKING", "NO_ADVERSE"]

    # Per-category counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    total_correct = 0
    total_predicted = 0
    total_gold = 0
    exact_match = 0

    for entry in gold:
        doc_id = entry["id"]
        if doc_id not in corpus:
            continue

        text = corpus[doc_id]["full_text"]
        predicted = set(extract_outcomes_improved(text))
        raw_outcomes = entry.get("outcomes") or []
        gold_outcomes = set(raw_outcomes)

        # Filter to evaluable codes only
        gold_eval = gold_outcomes & set(eval_codes)
        pred_eval = predicted & set(eval_codes)

        if gold_eval == pred_eval:
            exact_match += 1

        for code in eval_codes:
            in_gold = code in gold_eval
            in_pred = code in pred_eval

            if in_gold and in_pred:
                tp[code] += 1
            elif in_pred and not in_gold:
                fp[code] += 1
            elif in_gold and not in_pred:
                fn[code] += 1

    # Compute per-category metrics
    results = {}
    for code in eval_codes:
        precision = tp[code] / (tp[code] + fp[code]) if (tp[code] + fp[code]) > 0 else 0.0
        recall = tp[code] / (tp[code] + fn[code]) if (tp[code] + fn[code]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp[code] + fn[code]

        results[code] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[code],
            "fp": fp[code],
            "fn": fn[code],
            "support": support,
        }

    # Macro averages
    codes_with_support = [c for c in eval_codes if results[c]["support"] > 0]
    macro_precision = np.mean([results[c]["precision"] for c in codes_with_support])
    macro_recall = np.mean([results[c]["recall"] for c in codes_with_support])
    macro_f1 = np.mean([results[c]["f1"] for c in codes_with_support])

    # Micro averages
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    n_evaluated = sum(1 for e in gold if e["id"] in corpus)
    results["_macro"] = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    results["_micro"] = {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1}
    results["_exact_match"] = exact_match / n_evaluated if n_evaluated > 0 else 0.0
    results["_n_evaluated"] = n_evaluated

    return results


def print_evaluation(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'Category':<14} {'Prec':>6} {'Recall':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Support':>7}")
    print("-" * 60)

    eval_codes = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
                  "CAUTION", "FINE", "UNDERTAKING", "NO_ADVERSE"]

    for code in eval_codes:
        r = results[code]
        print(f"{code:<14} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['support']:>7}")

    print("-" * 60)
    m = results["_macro"]
    print(f"{'Macro avg':<14} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    m = results["_micro"]
    print(f"{'Micro avg':<14} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"\nExact match accuracy: {results['_exact_match']:.3f} ({int(results['_exact_match'] * results['_n_evaluated'])}/{results['_n_evaluated']})")


def predict_full_corpus(corpus_path: Path, output_path: Path) -> None:
    """Apply outcome extraction to full corpus and save predictions."""
    predictions = []
    with open(corpus_path) as f:
        for line in f:
            record = json.loads(line)
            outcomes = extract_outcomes_improved(record["full_text"])
            predictions.append({
                "id": record["id"],
                "predicted_outcomes": outcomes,
            })

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Summary stats
    outcome_counts = defaultdict(int)
    for pred in predictions:
        for o in pred["predicted_outcomes"]:
            outcome_counts[o] += 1
    no_outcome = sum(1 for p in predictions if not p["predicted_outcomes"])

    print(f"\nFull corpus outcome extraction ({len(predictions)} decisions):")
    for code, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {code}: {count} ({count/len(predictions)*100:.1f}%)")
    print(f"  No outcome detected: {no_outcome} ({no_outcome/len(predictions)*100:.1f}%)")


def show_errors(gold: list[dict], corpus: dict[str, dict]) -> None:
    """Show false positives and false negatives for debugging."""
    eval_codes = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
                  "CAUTION", "FINE", "UNDERTAKING", "NO_ADVERSE"]

    fp_examples = defaultdict(list)
    fn_examples = defaultdict(list)

    for entry in gold:
        doc_id = entry["id"]
        if doc_id not in corpus:
            continue

        text = corpus[doc_id]["full_text"]
        predicted = set(extract_outcomes_improved(text))
        raw_outcomes = entry.get("outcomes") or []
        gold_outcomes = set(raw_outcomes) & set(eval_codes)
        pred_eval = predicted & set(eval_codes)

        for code in pred_eval - gold_outcomes:
            fp_examples[code].append(doc_id)
        for code in gold_outcomes - pred_eval:
            fn_examples[code].append(doc_id)

    print("\n=== Error Analysis ===")
    for code in eval_codes:
        if fp_examples[code] or fn_examples[code]:
            print(f"\n{code}:")
            if fp_examples[code]:
                print(f"  FP ({len(fp_examples[code])}): {fp_examples[code][:5]}")
            if fn_examples[code]:
                print(f"  FN ({len(fn_examples[code])}): {fn_examples[code][:5]}")


def main():
    print("=== Outcome Extraction Validation ===\n")

    # Load data
    with open(GOLD_STANDARD_PATH) as f:
        gold = json.load(f)
    print(f"Gold standard: {len(gold)} decisions")

    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            record = json.loads(line)
            corpus[record["id"]] = record
    print(f"Corpus: {len(corpus)} decisions")

    # Evaluate
    results = evaluate_extraction(gold, corpus)
    print_evaluation(results)

    # Show errors
    show_errors(gold, corpus)

    # Apply to full corpus
    print("\n\nApplying to full corpus...")
    predict_full_corpus(CORPUS_PATH, OUTPUT_PATH)
    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
