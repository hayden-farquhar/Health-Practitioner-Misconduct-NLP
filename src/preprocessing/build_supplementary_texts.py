#!/usr/bin/env python3
"""
Write the 300 supplementary training decisions (new_sample_300.jsonl) as text files for the
logged model pass (Nov 2026 retrain protocol, config/retrain_protocol_2026-11.md).

Same layout as the re-code packet (build_recode_packet.py): a neutral header, then the corpus
full_text. No earlier labels are included.

Usage (from project root):
    python3 src/preprocessing/build_supplementary_texts.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT_ROOT / "data" / "annotations" / "new_sample_300.jsonl"
GOLD = PROJECT_ROOT / "data" / "annotations" / "gold_standard_2026-11.json"
CORPUS = PROJECT_ROOT / "data" / "extracted" / "all_decisions.jsonl"
OUT = PROJECT_ROOT / "data" / "annotations" / "supplementary_2026-11"
TEXTS = OUT / "texts"


def main():
    ids = sorted(json.loads(line)["id"] for line in open(SAMPLE))
    overlap = set(ids) & {g["id"] for g in json.load(open(GOLD))}
    if overlap:
        raise SystemExit(f"{len(overlap)} supplementary ids are in the gold standard")
    want = set(ids)
    docs = {}
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                docs[r["id"]] = r
    if want - docs.keys():
        raise SystemExit(f"ids missing from corpus: {sorted(want - docs.keys())[:5]}")

    TEXTS.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, doc_id in enumerate(ids, 1):
        r = docs[doc_id]
        item = f"SUP-{i:03d}"
        header = (
            f"ITEM: {item}\nCITATION: {r.get('citation')}\nTRIBUNAL: {r.get('tribunal')} "
            f"({r.get('jurisdiction')})\nDATE: {r.get('date')}\nURL: {r.get('url')}\n"
            + "=" * 78 + "\n\n"
        )
        (TEXTS / f"{item}.txt").write_text(header + (r.get("full_text") or ""), encoding="utf-8")
        mapping[item] = doc_id
    (OUT / "item_ids.json").write_text(json.dumps(mapping, indent=1))
    print(f"Wrote {len(mapping)} texts to {TEXTS}")


if __name__ == "__main__":
    main()
