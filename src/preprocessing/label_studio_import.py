"""Convert annotation sample to Label Studio import format.

Reads the annotation sample JSONL and the full corpus to produce a JSON file
ready for import into Label Studio, with pre-populated predictions for known
metadata fields.

Usage:
    python -m src.preprocessing.label_studio_import [--max-chars 50000]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.scraping.config import ALL_DECISIONS_PATH, DATA_DIR

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = DATA_DIR / "annotations"
SAMPLE_PATH = ANNOTATIONS_DIR / "annotation_sample.jsonl"
LS_TASKS_PATH = ANNOTATIONS_DIR / "label_studio_tasks.json"

# Map internal outcome codes to Label Studio choice values
OUTCOME_MAP = {
    "cancellation": "CANCEL — Cancellation",
    "suspension": "SUSPEND — Suspension",
    "conditions": "CONDITIONS — Conditions",
    "reprimand": "REPRIMAND — Reprimand",
    "fine": "FINE — Fine",
    "caution": "CAUTION — Caution",
    "undertaking": "UNDERTAKING — Undertaking",
    "no_adverse_finding": "NO_ADVERSE — No adverse finding",
}


def _load_full_texts() -> dict[str, str]:
    """Load full texts from corpus, indexed by decision ID."""
    texts = {}
    with open(ALL_DECISIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts[record["id"]] = record.get("full_text", "")
    return texts


def _load_sample() -> list[dict]:
    """Load the annotation sample metadata."""
    records = []
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_predictions(record: dict) -> list[dict]:
    """Build Label Studio prediction results from known metadata."""
    results = []

    # Pre-populate profession if known
    profession = record.get("profession")
    if profession:
        results.append({
            "from_name": "profession",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [profession]},
        })

    # Pre-populate outcomes if known
    outcomes = record.get("outcome_raw")
    if outcomes:
        mapped = [OUTCOME_MAP[o] for o in outcomes if o in OUTCOME_MAP]
        if mapped:
            results.append({
                "from_name": "outcome",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": mapped},
            })

    return results


def convert_to_label_studio(max_chars: int = 50000) -> list[dict]:
    """Convert annotation sample to Label Studio task format."""
    sample = _load_sample()
    logger.info(f"Loaded {len(sample)} sample records")

    full_texts = _load_full_texts()
    logger.info(f"Loaded {len(full_texts)} full texts from corpus")

    tasks = []
    missing_text = 0

    for record in sample:
        doc_id = record["id"]
        text = full_texts.get(doc_id, "")

        if not text:
            missing_text += 1
            logger.warning(f"No full text found for {doc_id}")
            continue

        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_chars} characters ...]"

        task = {
            "data": {
                "text": text,
                "id": doc_id,
                "tribunal": record.get("tribunal"),
                "jurisdiction": record.get("jurisdiction"),
                "date": record.get("date"),
                "citation": record.get("citation"),
                "source": record.get("source"),
            },
        }

        # Add predictions for known metadata
        predictions = _build_predictions(record)
        if predictions:
            task["predictions"] = [{
                "model_version": "metadata_enrichment_v1",
                "result": predictions,
            }]

        tasks.append(task)

    if missing_text:
        logger.warning(f"{missing_text} decisions had no full text")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotation sample to Label Studio format"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50000,
        help="Maximum characters per decision text (default: 50000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not SAMPLE_PATH.exists():
        logger.error(
            f"Annotation sample not found: {SAMPLE_PATH}\n"
            "Run sample_selection.py first."
        )
        sys.exit(1)

    tasks = convert_to_label_studio(max_chars=args.max_chars)

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LS_TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    logger.info(f"Written {len(tasks)} tasks to {LS_TASKS_PATH}")

    # Summary
    with_predictions = sum(1 for t in tasks if "predictions" in t)
    print(f"\nLabel Studio import file created: {LS_TASKS_PATH}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  With pre-populated predictions: {with_predictions}")
    print(f"\nTo import into Label Studio:")
    print(f"  1. Open Label Studio (label-studio start)")
    print(f"  2. Create/open project")
    print(f"  3. Settings > Labeling Interface > paste config/label_studio_config.xml")
    print(f"  4. Import > Upload file > select {LS_TASKS_PATH}")


if __name__ == "__main__":
    main()
