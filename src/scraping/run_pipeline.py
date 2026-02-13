"""CLI orchestrator for the full data acquisition pipeline.

Runs steps in sequence with clear logging. Supports running individual
steps via the --step flag so long-running operations (like the AustLII
scrape) can be resumed independently.

Usage:
    python -m src.scraping.run_pipeline              # run all steps
    python -m src.scraping.run_pipeline --step 1     # NSW corpus download only
    python -m src.scraping.run_pipeline --step 2     # Non-NSW HuggingFace explore
    python -m src.scraping.run_pipeline --step 2d    # Non-NSW HuggingFace download
    python -m src.scraping.run_pipeline --step 3     # AustLII discovery (gap-fill)
    python -m src.scraping.run_pipeline --step 4     # AustLII download (resumable)
    python -m src.scraping.run_pipeline --step 4w    # Wayback Machine download (410-Gone tribunals)
    python -m src.scraping.run_pipeline --step 5     # Extract from raw HTML/RTF/PDF
    python -m src.scraping.run_pipeline --step 6     # Merge into unified corpus
"""

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)

STEPS = {
    "1": "Download & filter HuggingFace corpus (NSW)",
    "2": "Explore HuggingFace corpus for non-NSW coverage",
    "2d": "Download & filter HuggingFace corpus (non-NSW)",
    "3": "Discover AustLII decision links (gap-fill)",
    "4": "Download AustLII decision HTML (resumable)",
    "4w": "Download from Wayback Machine (410-Gone tribunals)",
    "5": "Extract structured data from raw HTML/RTF/PDF",
    "6": "Merge into unified corpus",
}

# Steps to run in full sequential pipeline (excludes explore-only step 2)
ALL_STEPS = ["1", "2d", "3", "4", "4w", "5", "6"]


def step_1_download_nsw():
    """Step 1: Download and filter the HuggingFace corpus for NSW decisions."""
    from .corpus_download import download_and_filter

    logger.info("=" * 60)
    logger.info("STEP 1: Download & filter HuggingFace corpus (NSW)")
    logger.info("=" * 60)

    stats = download_and_filter()
    logger.info(f"Step 1 complete: {stats['written']} NSW decisions saved")
    return stats


def step_2_explore_nonnsw():
    """Step 2: Explore HuggingFace corpus for non-NSW tribunal coverage."""
    from .corpus_download_nonnsw import explore

    logger.info("=" * 60)
    logger.info("STEP 2: Explore HuggingFace corpus for non-NSW coverage")
    logger.info("=" * 60)

    stats = explore()
    total = sum(stats["tribunal_hits"].values())
    logger.info(f"Step 2 complete: {total} non-NSW tribunal hits found")
    return stats


def step_2d_download_nonnsw():
    """Step 2d: Download non-NSW health decisions from HuggingFace."""
    from .corpus_download_nonnsw import download_and_filter

    logger.info("=" * 60)
    logger.info("STEP 2d: Download & filter HuggingFace corpus (non-NSW)")
    logger.info("=" * 60)

    stats = download_and_filter()
    logger.info(f"Step 2d complete: {stats['written']} non-NSW decisions saved")
    if stats["uncovered_tribunals"]:
        logger.info(
            f"  Tribunals needing AustLII scraping: {stats['uncovered_tribunals']}"
        )
    return stats


def step_3_discover_austlii():
    """Step 3: Discover AustLII decision links (index pages only)."""
    from .austlii_scraper import AustLIIScraper

    logger.info("=" * 60)
    logger.info("STEP 3: Discover AustLII decision links (gap-fill)")
    logger.info("=" * 60)

    scraper = AustLIIScraper(dry_run=True)
    stats = scraper.run()
    logger.info(f"Step 3 complete: {stats['total_discovered']} decisions discovered")
    return stats


def step_4_download_austlii():
    """Step 4: Download AustLII decision HTML (long-running, resumable)."""
    from .austlii_scraper import AustLIIScraper

    logger.info("=" * 60)
    logger.info("STEP 4: Download AustLII decision HTML")
    logger.info("This step is resumable — safe to interrupt and restart.")
    logger.info("=" * 60)

    scraper = AustLIIScraper(dry_run=False)
    stats = scraper.run()
    logger.info(f"Step 4 complete: {stats['total_downloaded']} decisions downloaded")
    return stats


def step_4w_wayback():
    """Step 4w: Download decisions from the Wayback Machine for 410-Gone tribunals."""
    from .wayback_scraper import run as wayback_run

    logger.info("=" * 60)
    logger.info("STEP 4w: Download from Wayback Machine (NTHPRT, TasHPT, TasCAT)")
    logger.info("=" * 60)

    stats = wayback_run()
    logger.info(f"Step 4w complete: {stats['total_downloaded']} decisions downloaded")
    return stats


def step_5_extract():
    """Step 5: Extract structured data from raw HTML files."""
    from .austlii_extractor import extract_all

    logger.info("=" * 60)
    logger.info("STEP 5: Extract structured data from raw HTML")
    logger.info("=" * 60)

    stats = extract_all()
    logger.info(f"Step 5 complete: {stats['total_extracted']} decisions extracted")
    return stats


def step_6_merge():
    """Step 6: Merge all sources into unified corpus."""
    from .merge_corpus import merge_and_deduplicate

    logger.info("=" * 60)
    logger.info("STEP 6: Merge into unified corpus")
    logger.info("=" * 60)

    stats = merge_and_deduplicate()
    logger.info(f"Step 6 complete: {stats['merged_count']} total decisions")
    return stats


STEP_FUNCTIONS = {
    "1": step_1_download_nsw,
    "2": step_2_explore_nonnsw,
    "2d": step_2d_download_nonnsw,
    "3": step_3_discover_austlii,
    "4": step_4_download_austlii,
    "4w": step_4w_wayback,
    "5": step_5_extract,
    "6": step_6_merge,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the health practitioner decision data acquisition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  Step {k}: {v}" for k, v in STEPS.items()),
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=list(STEPS.keys()),
        default=None,
        help="Run a specific step only (default: run all steps in sequence)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy HTTP request logs from datasets/httpx
    for noisy_logger in ("httpx", "urllib3", "httpcore", "huggingface_hub",
                         "datasets", "fsspec"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    start_time = time.time()

    if args.step:
        steps_to_run = [args.step]
    else:
        steps_to_run = ALL_STEPS

    logger.info(f"Running steps: {steps_to_run}")

    for step_key in steps_to_run:
        step_fn = STEP_FUNCTIONS[step_key]
        try:
            step_fn()
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during step {step_key}. Progress saved.")
            sys.exit(1)
        except Exception:
            logger.exception(f"Step {step_key} failed")
            sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
