"""Download and filter the Open Australian Legal Corpus from HuggingFace.

Streams the dataset to avoid loading 9.4 GB into memory. Filters to
NSW health practitioner tribunal decisions and outputs to the project's
standard JSONL schema.

Usage:
    python -m src.scraping.corpus_download
"""

import json
import logging
import re
from datetime import datetime, timezone

from .config import (
    EXTRACTED_DIR,
    HUGGINGFACE_DATASET_ID,
    NSW_DECISIONS_PATH,
    load_tribunals,
)
from .filters import is_health_decision, match_citation

logger = logging.getLogger(__name__)

# NSW tribunal citation patterns to look for in the HuggingFace corpus
NSW_HEALTH_CITATION_PREFIXES = [
    "NSWCATOD",
    "NSWMT",
    "NSWNMT",
    "NSWCATAP",
    "NSWChirT",
    "NSWDenT",
    "NSWOptT",
    "NSWOstT",
]

_NSW_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s*("
    + "|".join(NSW_HEALTH_CITATION_PREFIXES)
    + r")\s+(\d+)",
    re.IGNORECASE,
)


def _make_document_id(citation: str) -> str:
    """Generate a document ID from a citation.

    E.g. "[2023] NSWCATOD 456" -> "nswcatod_2023_456"
    """
    m = _NSW_CITATION_RE.search(citation)
    if m:
        year, tribunal, number = m.groups()
        return f"{tribunal.lower()}_{year}_{number}"
    # Fallback: slugify the citation
    slug = re.sub(r"[^\w]+", "_", citation.lower()).strip("_")
    return slug


def _parse_date(date_str: str | None) -> str | None:
    """Parse date from HuggingFace format to ISO date string."""
    if not date_str:
        return None
    try:
        # HuggingFace format: "YYYY-MM-DD HH:MM:SS"
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _to_standard_schema(record: dict) -> dict:
    """Convert a HuggingFace corpus record to the project's standard schema."""
    citation = record.get("citation", "")
    tribunal_id = match_citation(citation)

    config = load_tribunals()
    tribunal_info = config["tribunals"].get(tribunal_id, {}) if tribunal_id else {}

    return {
        "id": _make_document_id(citation),
        "source": "open_australian_legal_corpus",
        "tribunal": tribunal_id or "UNKNOWN",
        "jurisdiction": tribunal_info.get("jurisdiction", "NSW"),
        "date": _parse_date(record.get("date")),
        "citation": citation,
        "url": record.get("url", ""),
        "practitioner_name": None,
        "profession": None,
        "board": None,
        "catchwords": None,
        "full_text": record.get("text", ""),
        "sections": {},
        "outcome_raw": None,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def download_and_filter():
    """Download the HuggingFace corpus, filter to NSW health decisions, write JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
        raise

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming Open Australian Legal Corpus from HuggingFace...")
    dataset = load_dataset(
        HUGGINGFACE_DATASET_ID,
        split="corpus",
        streaming=True,
    )

    total_scanned = 0
    citation_matched = 0
    health_filtered = 0
    written = 0
    seen_ids = set()

    with open(NSW_DECISIONS_PATH, "w", encoding="utf-8") as f:
        for record in dataset:
            total_scanned += 1

            if total_scanned % 10000 == 0:
                logger.info(
                    f"  Scanned {total_scanned:,} records, "
                    f"matched {citation_matched:,} citations, "
                    f"kept {written:,} health decisions"
                )

            # Step 1: Filter by source — only NSW Caselaw
            source = record.get("source", "")
            if source != "nsw_caselaw":
                continue

            # Step 2: Filter by type — decisions only
            doc_type = record.get("type", "")
            if doc_type != "decision":
                continue

            # Step 3: Filter by citation pattern — NSW health tribunals
            citation = record.get("citation", "")
            if not _NSW_CITATION_RE.search(citation):
                continue

            citation_matched += 1
            tribunal_id = match_citation(citation)

            # Step 4: For non-health-specific tribunals (NSWCATOD, NSWCATAP),
            # apply keyword filtering
            text = record.get("text", "")
            if not is_health_decision(citation, text, tribunal_id):
                continue

            health_filtered += 1

            # Convert to standard schema
            doc = _to_standard_schema(record)

            # Deduplicate by ID
            if doc["id"] in seen_ids:
                continue
            seen_ids.add(doc["id"])

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        f"Done. Scanned {total_scanned:,} records total. "
        f"Citation-matched: {citation_matched:,}. "
        f"Health-filtered: {health_filtered:,}. "
        f"Written: {written:,} decisions to {NSW_DECISIONS_PATH}"
    )

    return {
        "total_scanned": total_scanned,
        "citation_matched": citation_matched,
        "health_filtered": health_filtered,
        "written": written,
        "output_path": str(NSW_DECISIONS_PATH),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy HTTP request logs from datasets/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    stats = download_and_filter()
    print(f"\nCorpus download complete: {stats['written']} NSW decisions saved.")


if __name__ == "__main__":
    main()
