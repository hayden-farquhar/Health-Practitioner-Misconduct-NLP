"""Inter-rater reliability (IRR) analysis for annotation quality.

Imports two coders' annotations from Label Studio JSON exports and computes:
- Cohen's kappa per misconduct category (binary per label)
- Krippendorff's alpha for multi-label coding
- Percent agreement per category and overall

Usage:
    python -m src.analysis.irr_analysis --coder1 export1.json --coder2 export2.json
    python -m src.analysis.irr_analysis --test  # run with synthetic data
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Level-1 misconduct categories
L1_CATEGORIES = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF", "N/A"]

# Finding categories
FINDING_CATEGORIES = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]

# Outcome categories
OUTCOME_CATEGORIES = [
    "CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND",
    "FINE", "CAUTION", "UNDERTAKING", "NO_ADVERSE", "OTHER", "N/A",
]


def _parse_label_studio_export(filepath: Path) -> dict[str, dict]:
    """Parse a Label Studio JSON export into a dict keyed by decision ID.

    Returns: {decision_id: {"misconduct": set, "finding": str, "outcome": set}}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    annotations = {}
    for task in tasks:
        doc_id = task.get("data", {}).get("id")
        if not doc_id:
            continue

        # Get the first annotation (or the most recent)
        annots = task.get("annotations", [])
        if not annots:
            continue

        annot = annots[-1]  # most recent
        results = annot.get("result", [])

        parsed = {"misconduct": set(), "finding": None, "outcome": set()}

        for result in results:
            from_name = result.get("from_name")
            value = result.get("value", {})

            if from_name == "misconduct":
                # Taxonomy selections — extract L1 codes
                taxonomy = value.get("taxonomy", [])
                for path in taxonomy:
                    # path is like ["CLIN — Clinical competence"] or
                    # ["CLIN — Clinical competence", "CLIN_DIAG — Diagnostic failure"]
                    if path:
                        code = path[0].split(" — ")[0].strip()
                        parsed["misconduct"].add(code)

            elif from_name == "finding":
                choices = value.get("choices", [])
                if choices:
                    parsed["finding"] = choices[0].split(" — ")[0].strip()

            elif from_name == "outcome":
                choices = value.get("choices", [])
                for c in choices:
                    code = c.split(" — ")[0].strip()
                    parsed["outcome"].add(code)

        annotations[doc_id] = parsed

    return annotations


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Calculate Cohen's kappa for two binary arrays."""
    n = len(y1)
    if n == 0:
        return float("nan")

    # Observed agreement
    po = np.sum(y1 == y2) / n

    # Expected agreement by chance
    p1_pos = np.sum(y1 == 1) / n
    p2_pos = np.sum(y2 == 1) / n
    p1_neg = 1 - p1_pos
    p2_neg = 1 - p2_pos
    pe = p1_pos * p2_pos + p1_neg * p2_neg

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def krippendorff_alpha_binary(data: np.ndarray) -> float:
    """Calculate Krippendorff's alpha for binary data (2 coders, multiple items).

    Args:
        data: numpy array of shape (2, n_items) with 0/1 values.
              NaN for missing values.
    """
    n_coders, n_items = data.shape

    # Observed disagreement
    observed_disagreement = 0
    n_pairs = 0
    for j in range(n_items):
        vals = data[:, j]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            continue
        # All pairs within this item
        for a in range(len(valid)):
            for b in range(a + 1, len(valid)):
                observed_disagreement += (valid[a] - valid[b]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return float("nan")

    Do = observed_disagreement / n_pairs

    # Expected disagreement
    all_vals = data[~np.isnan(data)]
    if len(all_vals) < 2:
        return float("nan")

    n_total = len(all_vals)
    p1 = np.sum(all_vals == 1) / n_total
    p0 = np.sum(all_vals == 0) / n_total
    De = 2 * p1 * p0  # expected disagreement for binary nominal

    if De == 0:
        return 1.0 if Do == 0 else 0.0

    return 1 - Do / De


def compute_irr(
    coder1: dict[str, dict],
    coder2: dict[str, dict],
) -> dict:
    """Compute IRR statistics between two coders.

    Returns a dict with per-category kappa, overall alpha, and percent agreement.
    """
    # Find common decision IDs
    common_ids = sorted(set(coder1.keys()) & set(coder2.keys()))
    n = len(common_ids)

    if n == 0:
        logger.error("No overlapping decisions between the two coders.")
        return {"error": "No overlapping decisions", "n_common": 0}

    logger.info(f"Computing IRR on {n} common decisions")

    results = {
        "n_common": n,
        "misconduct_kappa": {},
        "misconduct_agreement": {},
        "finding_agreement": None,
        "finding_kappa": None,
        "outcome_kappa": {},
        "outcome_agreement": {},
        "overall_misconduct_alpha": None,
    }

    # Misconduct: Cohen's kappa per L1 category (binary)
    misconduct_matrix = np.zeros((2, n, len(L1_CATEGORIES)))
    for i, doc_id in enumerate(common_ids):
        for j, cat in enumerate(L1_CATEGORIES):
            misconduct_matrix[0, i, j] = 1 if cat in coder1[doc_id]["misconduct"] else 0
            misconduct_matrix[1, i, j] = 1 if cat in coder2[doc_id]["misconduct"] else 0

    for j, cat in enumerate(L1_CATEGORIES):
        y1 = misconduct_matrix[0, :, j].astype(int)
        y2 = misconduct_matrix[1, :, j].astype(int)

        kappa = cohens_kappa(y1, y2)
        agreement = np.mean(y1 == y2)

        results["misconduct_kappa"][cat] = round(kappa, 3)
        results["misconduct_agreement"][cat] = round(agreement, 3)

    # Krippendorff's alpha across all misconduct labels
    # Flatten: treat each (decision, category) as a separate item
    flat_data = np.zeros((2, n * len(L1_CATEGORIES)))
    for j, cat in enumerate(L1_CATEGORIES):
        flat_data[0, j * n : (j + 1) * n] = misconduct_matrix[0, :, j]
        flat_data[1, j * n : (j + 1) * n] = misconduct_matrix[1, :, j]

    results["overall_misconduct_alpha"] = round(
        krippendorff_alpha_binary(flat_data), 3
    )

    # Finding: Cohen's kappa (nominal, 4 categories)
    finding_y1 = []
    finding_y2 = []
    for doc_id in common_ids:
        f1 = coder1[doc_id].get("finding") or "NONE"
        f2 = coder2[doc_id].get("finding") or "NONE"
        finding_y1.append(f1)
        finding_y2.append(f2)

    finding_agreement = sum(a == b for a, b in zip(finding_y1, finding_y2)) / n
    results["finding_agreement"] = round(finding_agreement, 3)

    # Cohen's kappa for nominal finding (multi-class)
    all_finding_cats = sorted(set(finding_y1 + finding_y2))
    if len(all_finding_cats) > 1:
        # Calculate multi-class kappa
        n_cats = len(all_finding_cats)
        cat_idx = {c: i for i, c in enumerate(all_finding_cats)}
        confusion = np.zeros((n_cats, n_cats))
        for a, b in zip(finding_y1, finding_y2):
            confusion[cat_idx[a], cat_idx[b]] += 1
        po = np.trace(confusion) / n
        pe = sum(
            (confusion[i, :].sum() / n) * (confusion[:, i].sum() / n)
            for i in range(n_cats)
        )
        results["finding_kappa"] = round(
            (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0), 3
        )
    else:
        results["finding_kappa"] = 1.0

    # Outcome: Cohen's kappa per outcome category (binary)
    for cat in OUTCOME_CATEGORIES:
        y1 = np.array([1 if cat in coder1[doc_id]["outcome"] else 0 for doc_id in common_ids])
        y2 = np.array([1 if cat in coder2[doc_id]["outcome"] else 0 for doc_id in common_ids])
        kappa = cohens_kappa(y1, y2)
        agreement = np.mean(y1 == y2)
        results["outcome_kappa"][cat] = round(kappa, 3)
        results["outcome_agreement"][cat] = round(agreement, 3)

    return results


def format_report(results: dict) -> str:
    """Format IRR results as a readable report."""
    lines = []
    lines.append("=" * 65)
    lines.append("INTER-RATER RELIABILITY REPORT")
    lines.append("=" * 65)
    lines.append(f"Common decisions: {results['n_common']}")
    lines.append("")

    # Misconduct
    lines.append("MISCONDUCT TYPE (Level-1) — Cohen's Kappa")
    lines.append("-" * 50)
    lines.append(f"{'Category':<12} {'Kappa':>8} {'Agreement':>12} {'Status':>10}")
    lines.append("-" * 50)

    for cat in L1_CATEGORIES:
        kappa = results["misconduct_kappa"].get(cat, float("nan"))
        agreement = results["misconduct_agreement"].get(cat, float("nan"))
        status = "PASS" if kappa >= 0.7 else ("MARGINAL" if kappa >= 0.6 else "FAIL")
        lines.append(f"{cat:<12} {kappa:>8.3f} {agreement:>11.1%} {status:>10}")

    alpha = results.get("overall_misconduct_alpha", float("nan"))
    lines.append("-" * 50)
    lines.append(f"Krippendorff's alpha (all categories): {alpha:.3f}")
    lines.append("")

    # Finding
    lines.append("FINDING — Cohen's Kappa")
    lines.append("-" * 50)
    fk = results.get("finding_kappa", float("nan"))
    fa = results.get("finding_agreement", float("nan"))
    status = "PASS" if fk >= 0.7 else ("MARGINAL" if fk >= 0.6 else "FAIL")
    lines.append(f"Kappa: {fk:.3f}    Agreement: {fa:.1%}    {status}")
    lines.append("")

    # Outcome
    lines.append("OUTCOME — Cohen's Kappa")
    lines.append("-" * 50)
    lines.append(f"{'Category':<15} {'Kappa':>8} {'Agreement':>12} {'Status':>10}")
    lines.append("-" * 50)

    for cat in OUTCOME_CATEGORIES:
        kappa = results["outcome_kappa"].get(cat, float("nan"))
        agreement = results["outcome_agreement"].get(cat, float("nan"))
        status = "PASS" if kappa >= 0.7 else ("MARGINAL" if kappa >= 0.6 else "FAIL")
        lines.append(f"{cat:<15} {kappa:>8.3f} {agreement:>11.1%} {status:>10}")

    lines.append("=" * 65)

    # Overall summary
    misconduct_kappas = [
        v for v in results["misconduct_kappa"].values() if not np.isnan(v)
    ]
    if misconduct_kappas:
        mean_kappa = np.mean(misconduct_kappas)
        min_kappa = np.min(misconduct_kappas)
        lines.append(
            f"Mean misconduct kappa: {mean_kappa:.3f}  "
            f"(min: {min_kappa:.3f})"
        )
    target_met = all(k >= 0.7 for k in misconduct_kappas)
    lines.append(f"Target (kappa > 0.7 per category): {'MET' if target_met else 'NOT MET'}")
    lines.append("=" * 65)

    return "\n".join(lines)


def generate_synthetic_data(n: int = 50, seed: int = 42) -> tuple[dict, dict]:
    """Generate synthetic dual-coded data for testing.

    Creates two coders with ~80% agreement to test the IRR pipeline.
    """
    rng = np.random.RandomState(seed)

    coder1 = {}
    coder2 = {}

    for i in range(n):
        doc_id = f"test_{i:04d}"

        # Generate coder 1's annotations
        misconduct1 = set()
        for cat in L1_CATEGORIES:
            if rng.random() < 0.3:  # 30% prevalence per category
                misconduct1.add(cat)
        if not misconduct1:
            misconduct1.add(rng.choice(L1_CATEGORIES))

        finding1 = rng.choice(FINDING_CATEGORIES, p=[0.4, 0.25, 0.25, 0.1])

        outcome1 = set()
        for cat in OUTCOME_CATEGORIES[:5]:  # Focus on common outcomes
            if rng.random() < 0.25:
                outcome1.add(cat)
        if not outcome1:
            outcome1.add(rng.choice(OUTCOME_CATEGORIES[:5]))

        # Generate coder 2's annotations (~80% agreement)
        misconduct2 = set()
        for cat in L1_CATEGORIES:
            if cat in misconduct1:
                misconduct2.add(cat) if rng.random() < 0.85 else None
            else:
                misconduct2.add(cat) if rng.random() < 0.05 else None
        if not misconduct2:
            misconduct2.add(rng.choice(L1_CATEGORIES))

        finding2 = finding1 if rng.random() < 0.8 else rng.choice(FINDING_CATEGORIES)

        outcome2 = set()
        for cat in OUTCOME_CATEGORIES[:5]:
            if cat in outcome1:
                outcome2.add(cat) if rng.random() < 0.85 else None
            else:
                outcome2.add(cat) if rng.random() < 0.05 else None
        if not outcome2:
            outcome2.add(rng.choice(OUTCOME_CATEGORIES[:5]))

        coder1[doc_id] = {
            "misconduct": misconduct1,
            "finding": finding1,
            "outcome": outcome1,
        }
        coder2[doc_id] = {
            "misconduct": misconduct2,
            "finding": finding2,
            "outcome": outcome2,
        }

    return coder1, coder2


def main():
    parser = argparse.ArgumentParser(description="Inter-rater reliability analysis")
    parser.add_argument("--coder1", type=str, help="Label Studio export for coder 1")
    parser.add_argument("--coder2", type=str, help="Label Studio export for coder 2")
    parser.add_argument(
        "--test", action="store_true", help="Run with synthetic test data"
    )
    parser.add_argument(
        "--output", type=str, help="Output path for JSON results"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.test:
        logger.info("Generating synthetic test data...")
        coder1, coder2 = generate_synthetic_data()
    elif args.coder1 and args.coder2:
        logger.info(f"Loading coder 1: {args.coder1}")
        coder1 = _parse_label_studio_export(Path(args.coder1))
        logger.info(f"Loading coder 2: {args.coder2}")
        coder2 = _parse_label_studio_export(Path(args.coder2))
    else:
        parser.error("Provide --coder1 and --coder2, or use --test")

    results = compute_irr(coder1, coder2)
    report = format_report(results)
    print(report)

    if args.output:
        # Convert sets to lists for JSON serialization
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=list)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
