"""Select a stratified sample of decisions for manual annotation.

Selects 300 decisions stratified by jurisdiction, profession, and year band,
with minimum representation guarantees per stratum.

Usage:
    python -m src.preprocessing.sample_selection [--n 300] [--seed 42]
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.scraping.config import ALL_DECISIONS_PATH, DATA_DIR

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = DATA_DIR / "annotations"
SAMPLE_PATH = ANNOTATIONS_DIR / "annotation_sample.jsonl"
SAMPLE_STATS_PATH = ANNOTATIONS_DIR / "sample_statistics.json"

# Year bands for stratification
YEAR_BANDS = {
    "pre-2015": (0, 2014),
    "2015-2019": (2015, 2019),
    "2020+": (2020, 9999),
}

# Minimum decisions per stratum
MIN_PER_JURISDICTION = 5
MIN_PER_PROFESSION = 10  # only for professions with 10+ total decisions


def _get_year_band(date_str: str | None) -> str:
    """Map a date string to a year band."""
    if not date_str:
        return "unknown"
    try:
        year = int(date_str[:4])
    except (ValueError, TypeError):
        return "unknown"
    for band_name, (low, high) in YEAR_BANDS.items():
        if low <= year <= high:
            return band_name
    return "unknown"


def load_corpus() -> list[dict]:
    """Load all decisions (metadata only, no full_text for memory efficiency)."""
    records = []
    with open(ALL_DECISIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Keep only metadata fields for sampling
            records.append({
                "id": record.get("id"),
                "tribunal": record.get("tribunal"),
                "jurisdiction": record.get("jurisdiction"),
                "date": record.get("date"),
                "profession": record.get("profession"),
                "citation": record.get("citation"),
                "source": record.get("source"),
                "catchwords": record.get("catchwords"),
                "outcome_raw": record.get("outcome_raw"),
                "practitioner_name": record.get("practitioner_name"),
            })
    return records


def stratified_sample(
    records: list[dict],
    n: int = 300,
    seed: int = 42,
) -> list[dict]:
    """Select a stratified sample ensuring minimum representation.

    Strategy:
    1. First, ensure minimum per-jurisdiction (5 each).
    2. Then ensure minimum per-profession (10 each for professions with 10+ total).
    3. Fill remaining slots proportionally by jurisdiction x year_band.
    """
    import random

    rng = random.Random(seed)

    # Add year_band to each record
    for r in records:
        r["year_band"] = _get_year_band(r.get("date"))

    # Index records
    by_jurisdiction = defaultdict(list)
    by_profession = defaultdict(list)
    by_stratum = defaultdict(list)  # (jurisdiction, year_band)

    for r in records:
        j = r.get("jurisdiction", "UNKNOWN")
        p = r.get("profession") or "unknown"
        yb = r["year_band"]
        by_jurisdiction[j].append(r)
        by_profession[p].append(r)
        by_stratum[(j, yb)].append(r)

    selected_ids = set()
    selected = []

    def _add(record):
        if record["id"] not in selected_ids:
            selected_ids.add(record["id"])
            selected.append(record)

    # Phase 1: Ensure minimum per jurisdiction
    for jurisdiction, recs in sorted(by_jurisdiction.items()):
        current = sum(1 for s in selected if s.get("jurisdiction") == jurisdiction)
        needed = max(0, MIN_PER_JURISDICTION - current)
        if needed > 0:
            available = [r for r in recs if r["id"] not in selected_ids]
            rng.shuffle(available)
            for r in available[:needed]:
                _add(r)

    # Phase 2: Ensure minimum per profession (for professions with 10+ total)
    for profession, recs in sorted(by_profession.items()):
        if profession == "unknown":
            continue
        if len(recs) < MIN_PER_PROFESSION:
            continue
        current = sum(1 for s in selected if s.get("profession") == profession)
        needed = max(0, MIN_PER_PROFESSION - current)
        if needed > 0:
            available = [r for r in recs if r["id"] not in selected_ids]
            rng.shuffle(available)
            for r in available[:needed]:
                _add(r)

    # Phase 3: Fill remaining slots proportionally by (jurisdiction, year_band)
    remaining = n - len(selected)
    if remaining > 0:
        # Calculate proportional allocation
        total_available = sum(
            len([r for r in recs if r["id"] not in selected_ids])
            for recs in by_stratum.values()
        )

        if total_available > 0:
            for stratum_key, recs in sorted(by_stratum.items()):
                available = [r for r in recs if r["id"] not in selected_ids]
                if not available:
                    continue
                # Proportional allocation
                alloc = max(1, round(remaining * len(available) / total_available))
                rng.shuffle(available)
                for r in available[:alloc]:
                    if len(selected) >= n:
                        break
                    _add(r)

    # If we still need more, draw randomly from remaining
    if len(selected) < n:
        remaining_pool = [r for r in records if r["id"] not in selected_ids]
        rng.shuffle(remaining_pool)
        for r in remaining_pool:
            if len(selected) >= n:
                break
            _add(r)

    # If we have too many (due to rounding), trim
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    return selected


def compute_statistics(
    sample: list[dict], corpus: list[dict]
) -> dict:
    """Compute sample statistics and compare to corpus distribution."""
    stats = {
        "sample_size": len(sample),
        "corpus_size": len(corpus),
    }

    for label, records_list, key in [
        ("sample", sample, "sample"),
        ("corpus", corpus, "corpus"),
    ]:
        by_jurisdiction = Counter(r.get("jurisdiction", "UNKNOWN") for r in records_list)
        by_profession = Counter(r.get("profession") or "unknown" for r in records_list)
        by_year_band = Counter(_get_year_band(r.get("date")) for r in records_list)
        by_tribunal = Counter(r.get("tribunal", "UNKNOWN") for r in records_list)

        stats[f"{key}_by_jurisdiction"] = dict(sorted(by_jurisdiction.items()))
        stats[f"{key}_by_profession"] = dict(sorted(by_profession.items()))
        stats[f"{key}_by_year_band"] = dict(sorted(by_year_band.items()))
        stats[f"{key}_by_tribunal"] = dict(sorted(by_tribunal.items()))

    # Proportional comparison
    proportions = {}
    for jurisdiction in stats.get("corpus_by_jurisdiction", {}):
        corp_pct = stats["corpus_by_jurisdiction"][jurisdiction] / len(corpus) * 100
        samp_count = stats["sample_by_jurisdiction"].get(jurisdiction, 0)
        samp_pct = samp_count / len(sample) * 100 if sample else 0
        proportions[jurisdiction] = {
            "corpus_pct": round(corp_pct, 1),
            "sample_pct": round(samp_pct, 1),
            "sample_n": samp_count,
        }
    stats["jurisdiction_proportions"] = proportions

    return stats


def main():
    parser = argparse.ArgumentParser(description="Select stratified annotation sample")
    parser.add_argument("--n", type=int, default=300, help="Sample size (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not ALL_DECISIONS_PATH.exists():
        logger.error(f"Corpus not found: {ALL_DECISIONS_PATH}")
        sys.exit(1)

    # Load corpus
    corpus = load_corpus()
    logger.info(f"Loaded {len(corpus)} decisions from corpus")

    # Select sample
    sample = stratified_sample(corpus, n=args.n, seed=args.seed)
    logger.info(f"Selected {len(sample)} decisions for annotation")

    # Write sample
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for record in sample:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Written sample to {SAMPLE_PATH}")

    # Compute and write statistics
    stats = compute_statistics(sample, corpus)
    with open(SAMPLE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Written statistics to {SAMPLE_STATS_PATH}")

    # Print summary
    print(f"\nSample selected: {len(sample)} decisions (seed={args.seed})")
    print(f"\nBy jurisdiction:")
    for j, count in sorted(stats["sample_by_jurisdiction"].items()):
        corp_pct = stats["jurisdiction_proportions"].get(j, {}).get("corpus_pct", 0)
        samp_pct = stats["jurisdiction_proportions"].get(j, {}).get("sample_pct", 0)
        print(f"  {j:10s}: {count:4d}  (sample {samp_pct:.1f}% vs corpus {corp_pct:.1f}%)")
    print(f"\nBy profession:")
    for p, count in sorted(stats["sample_by_profession"].items(), key=lambda x: -x[1]):
        print(f"  {p:30s}: {count:4d}")
    print(f"\nBy year band:")
    for yb, count in sorted(stats["sample_by_year_band"].items()):
        print(f"  {yb:15s}: {count:4d}")

    print(f"\nOutput files:")
    print(f"  {SAMPLE_PATH}")
    print(f"  {SAMPLE_STATS_PATH}")


if __name__ == "__main__":
    main()
