"""Download and filter the HuggingFace corpus for non-NSW jurisdictions.

Before scraping AustLII directly (120s crawl delay per request), this script
checks whether the Open Australian Legal Corpus already contains decisions
from non-NSW health practitioner tribunals (VIC, QLD, SA, WA, ACT, NT, TAS).

Run --explore first to survey coverage, then --download if hits are found.

Usage:
    python -m src.scraping.corpus_download_nonnsw --explore     # survey coverage
    python -m src.scraping.corpus_download_nonnsw --download    # full download
"""

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone

from .config import (
    EXTRACTED_DIR,
    HF_EXPLORE_PATH,
    HUGGINGFACE_DATASET_ID,
    METADATA_DIR,
    NONNSW_HF_DECISIONS_PATH,
    load_tribunals,
)
from .filters import is_health_decision, match_citation

logger = logging.getLogger(__name__)

# NSW sources to skip (already handled by Step 1)
NSW_SOURCES = {"nsw_caselaw", "nsw_legislation"}


def _get_nonnsw_tribunal_ids() -> set[str]:
    """Return tribunal IDs that source from AustLII (non-NSW)."""
    config = load_tribunals()
    return {
        tid
        for tid, info in config["tribunals"].items()
        if info.get("source") == "austlii"
    }


def _build_nonnsw_citation_regex() -> re.Pattern:
    """Build a regex matching any non-NSW tribunal citation.

    Extracts the tribunal abbreviation from each citation_pattern in
    tribunals.json and combines them into a single alternation pattern.
    """
    config = load_tribunals()
    abbrevs = []
    for tid, info in config["tribunals"].items():
        if info.get("source") != "austlii":
            continue
        # Extract abbreviation: e.g. "\\[\\d{4}\\]\\s*VCAT\\s+\\d+" -> "VCAT"
        m = re.search(r"\\s\*(\w+)\\s\+", info["citation_pattern"])
        if m:
            abbrevs.append(m.group(1))

    return re.compile(
        r"\[(\d{4})\]\s*(" + "|".join(abbrevs) + r")\s+(\d+)",
        re.IGNORECASE,
    )


def _make_document_id(citation: str, tribunal_abbrev: str, year: str, number: str) -> str:
    """Generate a document ID from citation components."""
    return f"{tribunal_abbrev.lower()}_{year}_{number}"


def _parse_date(date_str: str | None) -> str | None:
    """Parse date from HuggingFace format to ISO date string."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _to_standard_schema(record: dict, tribunal_id: str, citation_match: re.Match) -> dict:
    """Convert a HuggingFace record to the project's standard JSONL schema."""
    config = load_tribunals()
    tribunal_info = config["tribunals"].get(tribunal_id, {})
    citation = record.get("citation", "")

    year, abbrev, number = citation_match.groups()
    doc_id = _make_document_id(citation, abbrev, year, number)

    return {
        "id": doc_id,
        "source": "open_australian_legal_corpus",
        "tribunal": tribunal_id,
        "jurisdiction": tribunal_info.get("jurisdiction", "UNKNOWN"),
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


def explore(max_records: int = 200_000):
    """Survey the HuggingFace corpus for non-NSW tribunal coverage.

    Scans the first max_records and reports:
    - All unique source values
    - All unique jurisdiction values
    - Citation matches for our non-NSW tribunals

    This helps decide whether a full download is worthwhile before
    committing to the ~30-minute full corpus scan.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install with: pip install datasets")
        raise

    citation_re = _build_nonnsw_citation_regex()
    nonnsw_tids = _get_nonnsw_tribunal_ids()

    logger.info(f"Surveying first {max_records:,} records for non-NSW tribunal content...")
    logger.info(f"Looking for tribunals: {sorted(nonnsw_tids)}")

    dataset = load_dataset(HUGGINGFACE_DATASET_ID, split="corpus", streaming=True)

    sources = Counter()
    jurisdictions = Counter()
    doc_types = Counter()
    tribunal_hits = Counter()
    sample_citations: dict[str, list[str]] = {}  # tribunal_id -> first 3 citations

    for i, record in enumerate(dataset):
        if i >= max_records:
            break

        if i % 25_000 == 0 and i > 0:
            logger.info(
                f"  Scanned {i:,} records | "
                f"tribunal hits so far: {dict(tribunal_hits)}"
            )

        source = record.get("source", "unknown")
        sources[source] += 1
        jurisdictions[record.get("jurisdiction", "unknown")] += 1
        doc_types[record.get("type", "unknown")] += 1

        # Skip NSW sources (already handled)
        if source in NSW_SOURCES:
            continue

        # Check for non-NSW tribunal citations
        citation = record.get("citation", "")
        cm = citation_re.search(citation)
        if cm:
            tid = match_citation(citation)
            if tid and tid in nonnsw_tids:
                tribunal_hits[tid] += 1
                if tid not in sample_citations:
                    sample_citations[tid] = []
                if len(sample_citations[tid]) < 3:
                    sample_citations[tid].append(citation)

    results = {
        "records_scanned": min(i + 1, max_records),
        "sources": dict(sources.most_common()),
        "jurisdictions": dict(jurisdictions.most_common()),
        "doc_types": dict(doc_types.most_common()),
        "tribunal_hits": dict(tribunal_hits.most_common()),
        "sample_citations": sample_citations,
        "surveyed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save results
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HF_EXPLORE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"SURVEY RESULTS ({results['records_scanned']:,} records)")
    logger.info(f"{'='*60}")
    logger.info(f"\nSources ({len(sources)}): {dict(sources.most_common(15))}")
    logger.info(f"\nJurisdictions: {dict(jurisdictions.most_common())}")
    logger.info(f"\nDocument types: {dict(doc_types.most_common())}")
    logger.info(f"\nNon-NSW tribunal hits: {dict(tribunal_hits.most_common())}")

    if sample_citations:
        logger.info("\nSample citations found:")
        for tid, cites in sample_citations.items():
            for c in cites:
                logger.info(f"  {tid}: {c}")

    total_hits = sum(tribunal_hits.values())
    if total_hits > 0:
        logger.info(
            f"\nFound {total_hits} non-NSW tribunal decisions in the first "
            f"{results['records_scanned']:,} records. "
            f"A full download is recommended: --download"
        )
    else:
        logger.info(
            "\nNo non-NSW tribunal decisions found in the sample. "
            "The HuggingFace corpus may not cover these tribunals. "
            "Proceed with AustLII scraping (Step 3) instead."
        )

    logger.info(f"\nFull results saved to: {HF_EXPLORE_PATH}")
    return results


def download_and_filter():
    """Stream the full HuggingFace corpus, filter for non-NSW health decisions.

    Writes matching decisions to NONNSW_HF_DECISIONS_PATH in the standard
    JSONL schema. Applies the same health filtering logic as Step 1 (NSW).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install with: pip install datasets")
        raise

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    citation_re = _build_nonnsw_citation_regex()
    nonnsw_tids = _get_nonnsw_tribunal_ids()
    config = load_tribunals()

    logger.info("Streaming full HuggingFace corpus for non-NSW health decisions...")
    logger.info(f"Target tribunals: {sorted(nonnsw_tids)}")

    dataset = load_dataset(HUGGINGFACE_DATASET_ID, split="corpus", streaming=True)

    total_scanned = 0
    citation_matched = 0
    health_filtered = 0
    written = 0
    seen_ids = set()
    by_tribunal = Counter()

    with open(NONNSW_HF_DECISIONS_PATH, "w", encoding="utf-8") as f:
        for record in dataset:
            total_scanned += 1

            if total_scanned % 25_000 == 0:
                logger.info(
                    f"  Scanned {total_scanned:,} | "
                    f"citation-matched {citation_matched:,} | "
                    f"kept {written:,}"
                )

            # Skip NSW sources
            source = record.get("source", "")
            if source in NSW_SOURCES:
                continue

            # Only decisions
            if record.get("type", "") != "decision":
                continue

            # Check citation against non-NSW tribunal patterns
            citation = record.get("citation", "")
            cm = citation_re.search(citation)
            if not cm:
                continue

            tribunal_id = match_citation(citation)
            if not tribunal_id or tribunal_id not in nonnsw_tids:
                continue

            citation_matched += 1

            # For health-specific tribunals, include directly.
            # For general tribunals, apply health keyword filtering.
            tribunal_info = config["tribunals"].get(tribunal_id, {})
            text = record.get("text", "")

            if not tribunal_info.get("health_specific", False):
                if not is_health_decision(citation, text, tribunal_id):
                    continue

            health_filtered += 1

            # Convert to standard schema
            doc = _to_standard_schema(record, tribunal_id, cm)

            # Deduplicate
            if doc["id"] in seen_ids:
                continue
            seen_ids.add(doc["id"])

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
            by_tribunal[tribunal_id] += 1

    logger.info(f"\n{'='*60}")
    logger.info("NON-NSW HUGGINGFACE DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total scanned:     {total_scanned:,}")
    logger.info(f"Citation-matched:  {citation_matched:,}")
    logger.info(f"Health-filtered:   {health_filtered:,}")
    logger.info(f"Written:           {written:,}")
    logger.info(f"By tribunal:       {dict(by_tribunal.most_common())}")
    logger.info(f"Output:            {NONNSW_HF_DECISIONS_PATH}")

    # Report which tribunals had zero coverage (need AustLII scraping)
    covered = set(by_tribunal.keys())
    uncovered = nonnsw_tids - covered
    if uncovered:
        logger.info(
            f"\nTribunals NOT found in HuggingFace (need AustLII scraping): "
            f"{sorted(uncovered)}"
        )
    else:
        logger.info("\nAll non-NSW tribunals found in HuggingFace corpus.")

    return {
        "total_scanned": total_scanned,
        "citation_matched": citation_matched,
        "health_filtered": health_filtered,
        "written": written,
        "by_tribunal": dict(by_tribunal),
        "uncovered_tribunals": sorted(uncovered),
        "output_path": str(NONNSW_HF_DECISIONS_PATH),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download non-NSW health practitioner decisions from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run --explore first to check if the HuggingFace corpus covers\n"
            "non-NSW tribunals. If it does, run --download to get them.\n"
            "If not, proceed with AustLII scraping (Step 3)."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--explore",
        action="store_true",
        help="Survey corpus for non-NSW tribunal coverage (fast, ~10 min)",
    )
    group.add_argument(
        "--download",
        action="store_true",
        help="Full download and filter for non-NSW health decisions (~30 min)",
    )
    parser.add_argument(
        "--max-explore",
        type=int,
        default=200_000,
        help="Max records to scan in explore mode (default: 200000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "httpcore", "huggingface_hub",
                   "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.explore:
        results = explore(max_records=args.max_explore)
        total = sum(results["tribunal_hits"].values())
        print(f"\nSurvey complete: {total} non-NSW tribunal hits found.")
        print(f"Results saved to: {HF_EXPLORE_PATH}")
    else:
        stats = download_and_filter()
        print(f"\nDownload complete: {stats['written']} non-NSW decisions saved.")
        if stats["uncovered_tribunals"]:
            print(f"Need AustLII scraping: {stats['uncovered_tribunals']}")


if __name__ == "__main__":
    main()
