"""Enrich metadata for all decisions in the corpus.

Re-runs extraction logic (profession, outcomes, sections, catchwords) on
the full text of every decision, filling in null fields without overwriting
existing non-null values.

Usage:
    python -m src.preprocessing.metadata_enrichment [--dry-run]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Reuse extraction functions from the scraping pipeline
from src.scraping.austlii_extractor import (
    OUTCOME_PATTERNS,
    PROFESSION_PATTERNS,
    BOARD_PATTERN,
    extract_sections,
    extract_outcomes,
    extract_profession,
    extract_board,
)
from src.scraping.config import ALL_DECISIONS_PATH, CORPUS_STATS_PATH, METADATA_DIR

logger = logging.getLogger(__name__)

# Additional profession patterns to improve coverage beyond the base set.
# These catch common phrasings in NSW decisions that the original patterns miss.
EXTRA_PROFESSION_PATTERNS = {
    "medical_practitioner": re.compile(
        r"\b(?:Dr\s+[A-Z]|the\s+(?:respondent|practitioner)\s+is\s+a\s+(?:medical|registered\s+medical))"
        r"|\bregistered\s+as\s+a\s+medical\s+practitioner\b"
        r"|\bmedical\s+practice\b"
        r"|\bmedical\s+registration\b",
        re.IGNORECASE,
    ),
    "nurse": re.compile(
        r"\bnurse\b"
        r"|\bnursing\s+(?:practice|registration|and\s+midwifery)\b",
        re.IGNORECASE,
    ),
    "dentist": re.compile(
        r"\bdental\s+(?:practice|registration|hygienist|therapist|prosthetist)\b",
        re.IGNORECASE,
    ),
    "pharmacist": re.compile(
        r"\bpharmacy\s+(?:practice|registration)\b",
        re.IGNORECASE,
    ),
    "psychologist": re.compile(
        r"\bpsychology\s+(?:practice|registration)\b"
        r"|\bpsychological\s+(?:practice|services)\b",
        re.IGNORECASE,
    ),
}

# Additional outcome patterns for better coverage
EXTRA_OUTCOME_PATTERNS = {
    "cancellation": re.compile(
        r"\bregistration\s+(?:of\s+the\s+respondent\s+)?(?:is\s+|be\s+)?cancelled\b"
        r"|\border(?:s|ed)?\s+(?:that\s+)?(?:the\s+)?registration.*?cancelled\b",
        re.IGNORECASE,
    ),
    "suspension": re.compile(
        r"\bregistration.*?suspended\b"
        r"|\bsuspension\s+(?:of\s+)?(?:the\s+)?(?:respondent'?s?\s+)?registration\b"
        r"|\bsuspended\s+(?:for|from)\b",
        re.IGNORECASE,
    ),
    "conditions": re.compile(
        r"\bthe\s+following\s+conditions\b"
        r"|\bconditions?\s+(?:are|be)\s+(?:imposed|placed|attached)\b"
        r"|\bpractise\s+subject\s+to\b",
        re.IGNORECASE,
    ),
    "reprimand": re.compile(
        r"\breprimand(?:ed)?\b",
        re.IGNORECASE,
    ),
    "no_adverse_finding": re.compile(
        r"\bcomplaint\s+(?:is\s+)?dismissed\b"
        r"|\bapplication\s+(?:is\s+)?dismissed\b"
        r"|\bno\s+(?:further\s+)?(?:disciplinary\s+)?action\b"
        r"|\bcharges?\s+(?:not\s+)?(?:established|proven|made\s+out|dismissed)\b",
        re.IGNORECASE,
    ),
}

# Catchword extraction from decision text (when HTML metadata not available)
CATCHWORD_PATTERNS = [
    # "CATCHWORDS: ..." or "Catchwords - ..."
    re.compile(
        r"(?:CATCHWORDS?|Catchwords?)\s*[:–\-]\s*(.+?)(?:\n\s*\n|\n[A-Z]{2,})",
        re.DOTALL,
    ),
    # "Key Words: ..."
    re.compile(
        r"(?:KEY\s*WORDS?|Key\s*[Ww]ords?)\s*[:–\-]\s*(.+?)(?:\n\s*\n|\n[A-Z]{2,})",
        re.DOTALL,
    ),
]

# Practitioner name extraction patterns (expanded)
NAME_PATTERNS = [
    # "Board v Name" or "HCCC v Name"
    re.compile(
        r"(?:Board\s+of\s+Australia|HCCC|Health\s+Care\s+Complaints\s+Commission|"
        r"Health\s+Ombudsman|Council\s+of\s+New\s+South\s+Wales|"
        r"Nursing\s+and\s+Midwifery\s+Council|Medical\s+Council|"
        r"Dental\s+Council|Pharmacy\s+Council|Psychology\s+Council)"
        r"\s+v\s+(.+?)(?:\s*\[|\s*\(|\s*$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "In the matter of [Name]"
    re.compile(
        r"[Ii]n\s+the\s+matter\s+of\s+(?:Dr\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ),
]


def enrich_profession(record: dict) -> str | None:
    """Extract profession from full text using base + extra patterns."""
    text = record.get("full_text", "")
    if not text:
        return None

    # Try base patterns first (searches first 5000 chars)
    result = extract_profession(text)
    if result:
        return result

    # Try extra patterns on a larger window
    search_text = text[:10000]
    for profession, pattern in EXTRA_PROFESSION_PATTERNS.items():
        if pattern.search(search_text):
            return profession

    return None


def enrich_outcomes(record: dict) -> list[str] | None:
    """Extract outcomes from full text using base + extra patterns."""
    text = record.get("full_text", "")
    if not text:
        return None

    # Try base extraction first
    outcomes = extract_outcomes(text)

    # Supplement with extra patterns on the orders section / last portion
    search_text = text[-8000:]
    for outcome_type, pattern in EXTRA_OUTCOME_PATTERNS.items():
        if outcome_type not in outcomes and pattern.search(search_text):
            outcomes.append(outcome_type)

    return outcomes if outcomes else None


def enrich_sections(record: dict) -> dict[str, str] | None:
    """Extract sections from full text."""
    text = record.get("full_text", "")
    if not text:
        return None

    sections = extract_sections(text)
    return sections if sections else None


def enrich_catchwords(record: dict) -> str | None:
    """Extract catchwords from decision text."""
    text = record.get("full_text", "")
    if not text:
        return None

    # Search in the first portion where catchwords typically appear
    search_text = text[:8000]
    for pattern in CATCHWORD_PATTERNS:
        match = pattern.search(search_text)
        if match:
            catchwords = match.group(1).strip()
            # Clean up: remove excessive whitespace
            catchwords = re.sub(r"\s+", " ", catchwords)
            if len(catchwords) > 10:  # Minimum meaningful length
                return catchwords

    return None


def enrich_practitioner_name(record: dict) -> str | None:
    """Extract practitioner name from decision text."""
    text = record.get("full_text", "")
    if not text:
        return None

    search_text = text[:5000]
    for pattern in NAME_PATTERNS:
        match = pattern.search(search_text)
        if match:
            name = match.group(1).strip()
            # Basic cleanup
            name = re.sub(r"\s+", " ", name)
            if 2 < len(name) < 100:
                return name

    # Try extracting from citation (e.g. "HCCC v Farrell [2017]")
    citation = record.get("citation", "")
    if citation:
        cit_match = re.search(r"v\s+(.+?)(?:\s*\[)", citation)
        if cit_match:
            return cit_match.group(1).strip()

    return None


def enrich_corpus(dry_run: bool = False) -> dict:
    """Enrich all decisions in the corpus.

    Reads all_decisions.jsonl, applies extraction to fill null fields,
    and writes back. Only fills gaps — never overwrites existing values.

    Returns a dict of before/after statistics.
    """
    if not ALL_DECISIONS_PATH.exists():
        logger.error(f"Corpus file not found: {ALL_DECISIONS_PATH}")
        sys.exit(1)

    # Load all records
    records = []
    with open(ALL_DECISIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    logger.info(f"Loaded {total} decisions from {ALL_DECISIONS_PATH}")

    # Before stats
    before = {
        "profession": sum(1 for r in records if r.get("profession")),
        "outcome_raw": sum(1 for r in records if r.get("outcome_raw")),
        "sections": sum(1 for r in records if r.get("sections")),
        "catchwords": sum(1 for r in records if r.get("catchwords")),
        "practitioner_name": sum(1 for r in records if r.get("practitioner_name")),
        "board": sum(1 for r in records if r.get("board")),
    }

    # Enrich each record
    enriched_counts = {
        "profession": 0,
        "outcome_raw": 0,
        "sections": 0,
        "catchwords": 0,
        "practitioner_name": 0,
        "board": 0,
    }

    for i, record in enumerate(records):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processing {i + 1}/{total}...")

        # Profession
        if not record.get("profession"):
            result = enrich_profession(record)
            if result:
                record["profession"] = result
                enriched_counts["profession"] += 1

        # Outcomes
        if not record.get("outcome_raw"):
            result = enrich_outcomes(record)
            if result:
                record["outcome_raw"] = result
                enriched_counts["outcome_raw"] += 1

        # Sections
        if not record.get("sections"):
            result = enrich_sections(record)
            if result:
                record["sections"] = result
                enriched_counts["sections"] += 1

        # Catchwords
        if not record.get("catchwords"):
            result = enrich_catchwords(record)
            if result:
                record["catchwords"] = result
                enriched_counts["catchwords"] += 1

        # Practitioner name
        if not record.get("practitioner_name"):
            result = enrich_practitioner_name(record)
            if result:
                record["practitioner_name"] = result
                enriched_counts["practitioner_name"] += 1

        # Board
        if not record.get("board"):
            text = record.get("full_text", "")
            if text:
                result = extract_board(text)
                if result:
                    record["board"] = result
                    enriched_counts["board"] += 1

    # After stats
    after = {
        "profession": sum(1 for r in records if r.get("profession")),
        "outcome_raw": sum(1 for r in records if r.get("outcome_raw")),
        "sections": sum(1 for r in records if r.get("sections")),
        "catchwords": sum(1 for r in records if r.get("catchwords")),
        "practitioner_name": sum(1 for r in records if r.get("practitioner_name")),
        "board": sum(1 for r in records if r.get("board")),
    }

    # Report
    logger.info("")
    logger.info("=" * 60)
    logger.info("METADATA ENRICHMENT RESULTS")
    logger.info("=" * 60)
    for field in before:
        b = before[field]
        a = after[field]
        e = enriched_counts[field]
        logger.info(
            f"  {field:25s}: {b:4d} -> {a:4d}  "
            f"({b/total*100:5.1f}% -> {a/total*100:5.1f}%)  "
            f"[+{e} enriched]"
        )
    logger.info("=" * 60)

    # Write back (unless dry run)
    if not dry_run:
        # Write to a temp file first, then rename (atomic-ish)
        tmp_path = ALL_DECISIONS_PATH.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.rename(ALL_DECISIONS_PATH)
        logger.info(f"Written enriched corpus to {ALL_DECISIONS_PATH}")

        # Update corpus statistics
        _update_statistics(records)
    else:
        logger.info("[DRY RUN] No files written.")

    return {
        "total": total,
        "before": before,
        "after": after,
        "enriched": enriched_counts,
    }


def _update_statistics(records: list[dict]) -> None:
    """Regenerate corpus_statistics.json after enrichment."""
    from collections import Counter

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    by_tribunal = Counter()
    by_jurisdiction = Counter()
    by_year = Counter()
    by_profession = Counter()
    by_source = Counter()
    by_outcome = Counter()

    for record in records:
        by_tribunal[record.get("tribunal", "UNKNOWN")] += 1
        by_jurisdiction[record.get("jurisdiction", "UNKNOWN")] += 1
        by_source[record.get("source", "UNKNOWN")] += 1

        profession = record.get("profession")
        if profession:
            by_profession[profession] += 1

        date = record.get("date")
        if date:
            by_year[date[:4]] += 1

        outcomes = record.get("outcome_raw")
        if outcomes:
            for o in outcomes:
                by_outcome[o] += 1

    stats = {
        "total_decisions": len(records),
        "by_tribunal": dict(sorted(by_tribunal.items())),
        "by_jurisdiction": dict(sorted(by_jurisdiction.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_profession": dict(sorted(by_profession.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "field_coverage": {
            "profession": sum(1 for r in records if r.get("profession")),
            "outcome_raw": sum(1 for r in records if r.get("outcome_raw")),
            "sections": sum(1 for r in records if r.get("sections")),
            "catchwords": sum(1 for r in records if r.get("catchwords")),
            "practitioner_name": sum(1 for r in records if r.get("practitioner_name")),
            "board": sum(1 for r in records if r.get("board")),
        },
    }

    with open(CORPUS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Updated statistics: {CORPUS_STATS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Enrich corpus metadata")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without writing changes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = enrich_corpus(dry_run=args.dry_run)
    total = stats["total"]
    print(f"\nEnrichment complete ({total} decisions processed).")
    for field in stats["before"]:
        b = stats["before"][field]
        a = stats["after"][field]
        print(f"  {field}: {b/total*100:.1f}% -> {a/total*100:.1f}%")


if __name__ == "__main__":
    main()
