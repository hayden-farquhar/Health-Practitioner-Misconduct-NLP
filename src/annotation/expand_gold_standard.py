"""Sample and prepare 300 new decisions for LLM annotation.

Produces data/annotations/new_sample_300.jsonl with truncated text for each decision.
"""

import json
import random
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent.parent
CORPUS = BASE / "data" / "extracted" / "all_decisions.jsonl"
GOLD = BASE / "data" / "annotations" / "gold_standard.json"
OUTPUT = BASE / "data" / "annotations" / "new_sample_300.jsonl"


def load_corpus():
    decisions = []
    with open(CORPUS) as f:
        for line in f:
            decisions.append(json.loads(line))
    return decisions


def load_existing_ids():
    with open(GOLD) as f:
        gs = json.load(f)
    return {d["id"] for d in gs}


def extract_key_text(decision, max_chars=8000):
    """Extract the most annotation-relevant text from a decision."""
    full = decision.get("full_text", "")
    sections = decision.get("sections", {})

    parts = []

    # Start of document (background, allegations)
    parts.append(full[:3000])

    # Sections if available
    for key in ["findings", "orders", "decision", "allegations", "evidence"]:
        if key in sections:
            sec_text = sections[key]
            if isinstance(sec_text, str) and len(sec_text) > 50:
                parts.append(f"\n--- {key.upper()} ---\n" + sec_text[:2500])

    # If no sections, take end of document (usually orders)
    if len(sections) == 0 and len(full) > 3000:
        parts.append("\n--- END OF DECISION ---\n" + full[-3000:])

    combined = "\n".join(parts)
    return combined[:max_chars]


def stratified_sample(candidates, n=300):
    """Stratified sample by jurisdiction and disciplinary status proxy."""
    random.seed(42)

    # Group by jurisdiction
    by_juris = {}
    for d in candidates:
        j = d.get("jurisdiction", "unknown")
        by_juris.setdefault(j, []).append(d)

    # Sample proportionally by jurisdiction
    total = len(candidates)
    sampled = []
    for juris, decs in sorted(by_juris.items()):
        n_juris = max(3, round(n * len(decs) / total))
        if len(decs) <= n_juris:
            sampled.extend(decs)
        else:
            sampled.extend(random.sample(decs, n_juris))

    # Trim or pad to exactly n
    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [d for d in candidates if d not in sampled]
        sampled.extend(random.sample(remaining, n - len(sampled)))

    return sampled


def main():
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  Total decisions: {len(corpus)}")

    existing_ids = load_existing_ids()
    print(f"  Already annotated: {len(existing_ids)}")

    candidates = [d for d in corpus if d["id"] not in existing_ids]
    print(f"  Available for sampling: {len(candidates)}")

    # Filter out very short decisions (< 500 chars)
    candidates = [d for d in candidates if len(d.get("full_text", "")) >= 500]
    print(f"  After filtering short texts: {len(candidates)}")

    print("\nSampling 300 decisions...")
    sample = stratified_sample(candidates, 300)

    # Print distribution
    juris_dist = Counter(d.get("jurisdiction", "unknown") for d in sample)
    print(f"\n  Jurisdiction distribution:")
    for j, c in sorted(juris_dist.items(), key=lambda x: -x[1]):
        print(f"    {j}: {c}")

    # Prepare output
    print(f"\nPreparing annotation file...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        for d in sample:
            entry = {
                "id": d["id"],
                "citation": d.get("citation", ""),
                "jurisdiction": d.get("jurisdiction", "unknown"),
                "profession": d.get("profession", "unknown"),
                "date": d.get("date", ""),
                "text_excerpt": extract_key_text(d),
                "text_length": len(d.get("full_text", "")),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Saved to {OUTPUT}")
    print(f"  Total: {len(sample)} decisions")

    # Also save full IDs for later merging
    ids_file = BASE / "data" / "annotations" / "new_sample_300_ids.json"
    with open(ids_file, "w") as f:
        json.dump([d["id"] for d in sample], f, indent=2)
    print(f"  IDs saved to {ids_file}")


if __name__ == "__main__":
    main()
