"""Compare LLM annotations against human annotations for IRR.

Reads human and LLM annotations and computes Cohen's kappa and
Krippendorff's alpha to measure agreement.

Supports two input formats:
- Label Studio JSON export (tasks with nested annotations/results)
- Flat JSON array (objects with id, misconduct_L1, finding, outcomes fields)

The format is auto-detected based on the structure of the first element.

Usage:
    python -m src.analysis.llm_irr_comparison --human human_annotations.json --llm llm_annotations.json
    python -m src.analysis.llm_irr_comparison --auto  # uses default paths
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.analysis.irr_analysis import (
    L1_CATEGORIES,
    FINDING_CATEGORIES,
    OUTCOME_CATEGORIES,
    cohens_kappa,
    krippendorff_alpha_binary,
    format_report,
    compute_irr,
)
from src.scraping.config import DATA_DIR

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = DATA_DIR / "annotations"
HUMAN_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "human_annotations.json"
LLM_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "llm_annotations.json"
IRR_RESULTS_PATH = ANNOTATIONS_DIR / "llm_irr_results.json"
IRR_REPORT_PATH = ANNOTATIONS_DIR / "llm_irr_report.txt"

# Map Label Studio choice values back to codes
FINDING_CHOICE_MAP = {
    "PM — Professional misconduct": "PM",
    "UPC_SAT — Unsatisfactory professional conduct": "UPC_SAT",
    "UPC_GEN — Unprofessional conduct": "UPC_GEN",
    "NO_FINDING — No finding": "NO_FINDING",
}

OUTCOME_CHOICE_MAP = {
    "CANCEL — Cancellation": "CANCEL",
    "SUSPEND — Suspension": "SUSPEND",
    "CONDITIONS — Conditions": "CONDITIONS",
    "REPRIMAND — Reprimand": "REPRIMAND",
    "FINE — Fine": "FINE",
    "CAUTION — Caution": "CAUTION",
    "UNDERTAKING — Undertaking": "UNDERTAKING",
    "NO_ADVERSE — No adverse finding": "NO_ADVERSE",
    "OTHER — Other": "OTHER",
    "N/A — Not a disciplinary decision": "N/A",
}


def _parse_label_studio_format(data: list[dict]) -> dict[str, dict]:
    """Parse annotations from Label Studio JSON export format.

    Returns: {decision_id: {"misconduct": set, "finding": str, "outcome": set}}
    """
    annotations = {}
    for task in data:
        doc_id = task.get("data", {}).get("id")
        if not doc_id:
            continue

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
                taxonomy = value.get("taxonomy", [])
                for path in taxonomy:
                    if path:
                        code = path[0].split(" — ")[0].strip()
                        parsed["misconduct"].add(code)

            elif from_name == "finding":
                choices = value.get("choices", [])
                if choices:
                    choice = choices[0]
                    parsed["finding"] = FINDING_CHOICE_MAP.get(choice, choice.split(" — ")[0].strip())

            elif from_name == "outcome":
                choices = value.get("choices", [])
                for c in choices:
                    code = OUTCOME_CHOICE_MAP.get(c, c.split(" — ")[0].strip())
                    parsed["outcome"].add(code)

        annotations[doc_id] = parsed

    return annotations


def _parse_flat_format(data: list[dict]) -> dict[str, dict]:
    """Parse annotations from flat JSON array format (agent/LLM style).

    Each entry has: id, is_disciplinary, misconduct_L1, finding, outcomes.

    Returns: {decision_id: {"misconduct": set, "finding": str, "outcome": set}}
    """
    annotations = {}
    for entry in data:
        doc_id = entry.get("id")
        if not doc_id or "error" in entry:
            continue

        # Non-disciplinary decisions map to N/A
        if not entry.get("is_disciplinary", True):
            annotations[doc_id] = {
                "misconduct": {"N/A"},
                "finding": "NO_FINDING",
                "outcome": {"N/A"},
            }
            continue

        misconduct = set(entry.get("misconduct_L1") or [])
        finding = entry.get("finding")
        outcomes = set(entry.get("outcomes") or [])

        annotations[doc_id] = {
            "misconduct": misconduct,
            "finding": finding,
            "outcome": outcomes,
        }

    return annotations


def _detect_and_parse(filepath: Path) -> dict[str, dict]:
    """Auto-detect format and parse annotations file.

    Distinguishes between:
    - Label Studio export: list of objects with "data" and "annotations" keys
    - Flat format: list of objects with "id" and "misconduct_L1" keys
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {}

    first = data[0]
    if "data" in first and "annotations" in first:
        logger.info(f"  Detected Label Studio format in {filepath.name}")
        return _parse_label_studio_format(data)
    elif "id" in first:
        logger.info(f"  Detected flat format in {filepath.name}")
        return _parse_flat_format(data)
    else:
        logger.error(f"Unrecognised annotation format in {filepath}")
        sys.exit(1)


def parse_annotations(filepath: Path) -> dict[str, dict]:
    """Parse annotations from any supported format.

    Returns: {decision_id: {"misconduct": set, "finding": str, "outcome": set}}
    """
    return _detect_and_parse(filepath)


def run_comparison(
    human_path: Path,
    llm_path: Path,
) -> dict:
    """Run full IRR comparison between human and LLM annotations."""
    logger.info(f"Loading human annotations from {human_path}")
    human = parse_annotations(human_path)
    logger.info(f"  Loaded {len(human)} human annotations")

    logger.info(f"Loading LLM annotations from {llm_path}")
    llm = parse_annotations(llm_path)
    logger.info(f"  Loaded {len(llm)} LLM annotations")

    # Run IRR using the existing compute_irr function
    results = compute_irr(human, llm)

    # Add summary metadata
    results["human_total"] = len(human)
    results["llm_total"] = len(llm)

    # Also compute per-decision agreement details
    common_ids = sorted(set(human.keys()) & set(llm.keys()))
    disagreements = []
    for doc_id in common_ids:
        h = human[doc_id]
        l = llm[doc_id]

        if h["misconduct"] != l["misconduct"] or h["finding"] != l["finding"] or h["outcome"] != l["outcome"]:
            disagreements.append({
                "id": doc_id,
                "human_misconduct": sorted(h["misconduct"]),
                "llm_misconduct": sorted(l["misconduct"]),
                "human_finding": h["finding"],
                "llm_finding": l["finding"],
                "human_outcome": sorted(h["outcome"]),
                "llm_outcome": sorted(l["outcome"]),
            })

    results["n_disagreements"] = len(disagreements)
    results["agreement_rate"] = round(1 - len(disagreements) / len(common_ids), 3) if common_ids else 0
    results["disagreements"] = disagreements[:50]  # Cap at 50 for readability

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM annotations against human annotations"
    )
    parser.add_argument("--human", type=str, help="Path to Label Studio JSON export")
    parser.add_argument("--llm", type=str, help="Path to LLM annotations JSON")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use default paths (data/annotations/)",
    )
    parser.add_argument("--output", type=str, help="Output path for results JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.auto:
        # Default: human_annotations.json and llm_annotations.json
        human_path = HUMAN_ANNOTATIONS_PATH
        if not human_path.exists():
            # Fallback: look for Label Studio export
            human_path = None
            for p in sorted(ANNOTATIONS_DIR.glob("*.json")):
                if p.name.startswith("project-") or p.name == "label_studio_export.json":
                    human_path = p
                    break
            if not human_path:
                logger.error(
                    "No human annotations found in data/annotations/. "
                    "Expected human_annotations.json or a Label Studio export."
                )
                sys.exit(1)

        llm_path = LLM_ANNOTATIONS_PATH
    elif args.human and args.llm:
        human_path = Path(args.human)
        llm_path = Path(args.llm)
    else:
        parser.error("Provide --human and --llm paths, or use --auto")

    if not human_path.exists():
        logger.error(f"Human annotations not found: {human_path}")
        sys.exit(1)
    if not llm_path.exists():
        logger.error(f"LLM annotations not found: {llm_path}")
        sys.exit(1)

    # Run comparison
    results = run_comparison(human_path, llm_path)

    # Print report
    report = format_report(results)
    print(report)

    # Additional summary
    print(f"\nOverall exact-match agreement: {results['agreement_rate']:.1%}")
    print(f"Decisions with any disagreement: {results['n_disagreements']}/{results['n_common']}")

    if results.get("disagreements"):
        print(f"\nSample disagreements (first 10):")
        for d in results["disagreements"][:10]:
            print(f"  {d['id']}:")
            if d["human_misconduct"] != d["llm_misconduct"]:
                print(f"    misconduct: human={d['human_misconduct']} vs llm={d['llm_misconduct']}")
            if d["human_finding"] != d["llm_finding"]:
                print(f"    finding: human={d['human_finding']} vs llm={d['llm_finding']}")
            if d["human_outcome"] != d["llm_outcome"]:
                print(f"    outcome: human={d['human_outcome']} vs llm={d['llm_outcome']}")

    # Save results
    output_path = Path(args.output) if args.output else IRR_RESULTS_PATH
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=list)
    logger.info(f"Results saved to {output_path}")

    # Save report text
    with open(IRR_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\n\nOverall exact-match agreement: {results['agreement_rate']:.1%}\n")
        f.write(f"Decisions with any disagreement: {results['n_disagreements']}/{results['n_common']}\n")
    logger.info(f"Report saved to {IRR_REPORT_PATH}")


if __name__ == "__main__":
    main()
