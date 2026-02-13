"""Combine all predictions into a unified enriched corpus.

Merges:
- Original corpus fields (from all_decisions.jsonl)
- Misconduct predictions (from sklearn or BERT)
- Outcome extraction (regex-based)
- NER entities (medications, schedules, conditions, boards)
- Topic assignments (from BERTopic)

Produces: data/extracted/enriched_corpus.jsonl

Usage:
    python -m src.classification.predict_full_corpus
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"

CORPUS_PATH = EXTRACTED_DIR / "all_decisions.jsonl"
MISCONDUCT_PATH = EXTRACTED_DIR / "misconduct_predictions_sklearn.jsonl"
OUTCOME_PATH = EXTRACTED_DIR / "outcome_predictions.jsonl"
ENTITIES_PATH = EXTRACTED_DIR / "entities.jsonl"
TOPIC_PATH = EXTRACTED_DIR / "topic_assignments.jsonl"
OUTPUT_PATH = EXTRACTED_DIR / "enriched_corpus.jsonl"

# Optional BERT predictions (if available from Colab)
BERT_PATH = EXTRACTED_DIR / "misconduct_predictions_bert.jsonl"


def load_jsonl_dict(path: Path, id_field: str = "id") -> dict:
    """Load a JSONL file into a dict keyed by id_field."""
    data = {}
    if not path.exists():
        print(f"  WARNING: {path.name} not found — skipping")
        return data
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            data[record[id_field]] = record
    return data


def main():
    print("=== Full Corpus Enrichment ===\n")

    # Load all data sources
    print("Loading data sources...")

    print(f"  Corpus: {CORPUS_PATH.name}")
    corpus = load_jsonl_dict(CORPUS_PATH)
    print(f"    {len(corpus)} records")

    print(f"  Misconduct predictions: {MISCONDUCT_PATH.name}")
    misconduct = load_jsonl_dict(MISCONDUCT_PATH)
    print(f"    {len(misconduct)} records")

    print(f"  Outcome predictions: {OUTCOME_PATH.name}")
    outcomes = load_jsonl_dict(OUTCOME_PATH)
    print(f"    {len(outcomes)} records")

    print(f"  Entity extraction: {ENTITIES_PATH.name}")
    entities = load_jsonl_dict(ENTITIES_PATH)
    print(f"    {len(entities)} records")

    print(f"  Topic assignments: {TOPIC_PATH.name}")
    topics = load_jsonl_dict(TOPIC_PATH)
    print(f"    {len(topics)} records")

    # Optional BERT predictions
    bert = {}
    if BERT_PATH.exists():
        print(f"  BERT predictions: {BERT_PATH.name}")
        bert = load_jsonl_dict(BERT_PATH)
        print(f"    {len(bert)} records")

    # Merge into enriched corpus
    print("\nMerging...")
    enriched = []
    missing_counts = Counter()

    for doc_id, record in corpus.items():
        enriched_record = {
            # Original fields
            "id": doc_id,
            "source": record.get("source"),
            "tribunal": record.get("tribunal"),
            "jurisdiction": record.get("jurisdiction"),
            "date": record.get("date"),
            "citation": record.get("citation"),
            "url": record.get("url"),
            "practitioner_name": record.get("practitioner_name"),
            "profession": record.get("profession"),
            "board": record.get("board"),
            "catchwords": record.get("catchwords"),
            "full_text": record.get("full_text"),
        }

        # Misconduct predictions
        if doc_id in misconduct:
            pred = misconduct[doc_id]
            enriched_record["predicted_misconduct_L1"] = pred.get("predicted_misconduct_L1", [])
            enriched_record["predicted_finding"] = pred.get("predicted_finding")
            enriched_record["predicted_is_disciplinary"] = pred.get("predicted_is_disciplinary")
        else:
            missing_counts["misconduct"] += 1
            enriched_record["predicted_misconduct_L1"] = []
            enriched_record["predicted_finding"] = None
            enriched_record["predicted_is_disciplinary"] = None

        # BERT predictions (if available)
        if doc_id in bert:
            enriched_record["predicted_misconduct_L1_bert"] = bert[doc_id].get("predicted_misconduct_L1_bert", [])
            enriched_record["misconduct_probs_bert"] = bert[doc_id].get("misconduct_probs", {})

        # Outcome extraction
        if doc_id in outcomes:
            enriched_record["predicted_outcomes"] = outcomes[doc_id].get("predicted_outcomes", [])
        else:
            missing_counts["outcomes"] += 1
            enriched_record["predicted_outcomes"] = []

        # Entities (stored as {"id": ..., "entities": {type: [...]}})
        if doc_id in entities:
            ent_data = entities[doc_id].get("entities", {})
            # Flatten to just entity text lists for compactness
            enriched_record["medications"] = [e["text"] for e in ent_data.get("MEDICATION", [])]
            enriched_record["schedules"] = [e.get("normalised", e["text"]) for e in ent_data.get("SCHEDULE", [])]
            enriched_record["conditions"] = [e["text"] for e in ent_data.get("CONDITION", [])]
            enriched_record["boards"] = [e["text"] for e in ent_data.get("BOARD", [])]
        else:
            missing_counts["entities"] += 1
            enriched_record["medications"] = []
            enriched_record["schedules"] = []
            enriched_record["conditions"] = []
            enriched_record["boards"] = []

        # Topic assignment
        if doc_id in topics:
            enriched_record["topic"] = topics[doc_id].get("topic")
        else:
            missing_counts["topics"] += 1
            enriched_record["topic"] = None

        enriched.append(enriched_record)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        for record in enriched:
            f.write(json.dumps(record, default=str) + "\n")

    print(f"\nEnriched corpus saved to {OUTPUT_PATH}")
    print(f"  {len(enriched)} decisions")

    if missing_counts:
        print("\n  Missing data:")
        for source, count in missing_counts.items():
            print(f"    {source}: {count} decisions without predictions")

    # Verification
    print("\n=== Verification ===")
    has_misconduct = sum(1 for r in enriched if r["predicted_misconduct_L1"])
    has_outcomes = sum(1 for r in enriched if r["predicted_outcomes"])
    has_meds = sum(1 for r in enriched if r["medications"])
    has_topic = sum(1 for r in enriched if r["topic"] is not None)
    has_finding = sum(1 for r in enriched if r["predicted_finding"])
    has_disc = sum(1 for r in enriched if r["predicted_is_disciplinary"] is not None)

    print(f"  Has misconduct predictions: {has_misconduct}/{len(enriched)}")
    print(f"  Has outcome predictions:    {has_outcomes}/{len(enriched)}")
    print(f"  Has medications:            {has_meds}/{len(enriched)}")
    print(f"  Has topic assignment:       {has_topic}/{len(enriched)}")
    print(f"  Has finding prediction:     {has_finding}/{len(enriched)}")
    print(f"  Has is_disciplinary:        {has_disc}/{len(enriched)}")

    # Null check
    null_fields = Counter()
    for r in enriched:
        for field in ["predicted_misconduct_L1", "predicted_outcomes", "predicted_finding",
                       "predicted_is_disciplinary", "topic"]:
            if r.get(field) is None or r.get(field) == []:
                null_fields[field] += 1

    print(f"\n  Fields with null/empty values:")
    for field, count in null_fields.most_common():
        print(f"    {field}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
