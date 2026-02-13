"""Merge NSW and AustLII decision files into a unified corpus.

Deduplicates by citation, generates corpus statistics, and writes the
final all_decisions.jsonl file.

Usage:
    python -m src.scraping.merge_corpus
"""

import json
import logging
from collections import Counter
from pathlib import Path

from .config import (
    ALL_DECISIONS_PATH,
    AUSTLII_DECISIONS_PATH,
    CORPUS_STATS_PATH,
    EXTRACTED_DIR,
    METADATA_DIR,
    NONNSW_HF_DECISIONS_PATH,
    NSW_DECISIONS_PATH,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {path}:{line_num}: {e}")

    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def merge_and_deduplicate():
    """Merge NSW and AustLII JSONL files, deduplicate, generate stats."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sources
    nsw_records = _load_jsonl(NSW_DECISIONS_PATH)
    nonnsw_hf_records = _load_jsonl(NONNSW_HF_DECISIONS_PATH)
    austlii_records = _load_jsonl(AUSTLII_DECISIONS_PATH)

    # Deduplicate by citation. Priority order:
    # 1. AustLII-extracted (richest metadata from HTML parsing)
    # 2. Non-NSW HuggingFace (same schema as NSW)
    # 3. NSW HuggingFace
    seen_citations = {}
    merged = []
    duplicates = 0

    for source_records in [austlii_records, nonnsw_hf_records, nsw_records]:
        for record in source_records:
            citation = record.get("citation")
            if citation and citation in seen_citations:
                duplicates += 1
                continue
            if citation:
                seen_citations[citation] = True
            merged.append(record)

    logger.info(
        f"Merged: {len(nsw_records)} NSW + {len(nonnsw_hf_records)} non-NSW HF + "
        f"{len(austlii_records)} AustLII = "
        f"{len(merged)} unique ({duplicates} duplicates removed)"
    )

    # Write merged corpus
    with open(ALL_DECISIONS_PATH, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Written: {ALL_DECISIONS_PATH}")

    # Generate corpus statistics
    stats = _generate_statistics(merged)
    with open(CORPUS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistics written: {CORPUS_STATS_PATH}")

    return {
        "nsw_count": len(nsw_records),
        "nonnsw_hf_count": len(nonnsw_hf_records),
        "austlii_count": len(austlii_records),
        "merged_count": len(merged),
        "duplicates": duplicates,
    }


def _generate_statistics(records: list[dict]) -> dict:
    """Generate corpus statistics from merged records."""
    by_tribunal = Counter()
    by_jurisdiction = Counter()
    by_year = Counter()
    by_profession = Counter()
    by_source = Counter()
    by_jurisdiction_year = Counter()

    for record in records:
        tribunal = record.get("tribunal", "UNKNOWN")
        jurisdiction = record.get("jurisdiction", "UNKNOWN")
        source = record.get("source", "UNKNOWN")
        profession = record.get("profession")
        date = record.get("date")

        by_tribunal[tribunal] += 1
        by_jurisdiction[jurisdiction] += 1
        by_source[source] += 1

        if profession:
            by_profession[profession] += 1

        if date:
            year = date[:4]
            by_year[year] += 1
            by_jurisdiction_year[f"{jurisdiction}_{year}"] += 1

    return {
        "total_decisions": len(records),
        "by_tribunal": dict(sorted(by_tribunal.items())),
        "by_jurisdiction": dict(sorted(by_jurisdiction.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_profession": dict(sorted(by_profession.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_jurisdiction_year": dict(sorted(by_jurisdiction_year.items())),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stats = merge_and_deduplicate()
    print(
        f"\nMerge complete: {stats['merged_count']} decisions "
        f"({stats['duplicates']} duplicates removed)"
    )


if __name__ == "__main__":
    main()
