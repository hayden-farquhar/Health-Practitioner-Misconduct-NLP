#!/usr/bin/env python3
"""
Sample the second test set and build its blinded coding packet (Nov 2026).

Implements the sampling section of config/external_test_protocol_2026-11.md: 40 decisions per
period stratum (1999-2009, 2010-2014, 2015-2019, 2020-2026), drawn at random from decisions that
no one has ever labelled. The packet has the same layout and fields as the October re-code, and
contains only decision text and neutral metadata (no labels, predictions or enriched fields).

Usage (from project root):
    python3 src/preprocessing/build_test2_packet.py
Refuses to overwrite an existing results workbook.
"""

import json
import random
from collections import Counter
from pathlib import Path

from build_recode_packet import CORPUS, TAXONOMY, write_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANN = PROJECT_ROOT / "data" / "annotations"
OUT = ANN / "test2_2026-11"
TEXTS = OUT / "texts"
RESULTS = OUT / "P13_test2_results.xlsx"
SAMPLE = OUT / "sample_manifest.json"
SEED = 20261106
PER_STRATUM = 40
STRATA = [("1999-2009", 1999, 2009), ("2010-2014", 2010, 2014),
          ("2015-2019", 2015, 2019), ("2020-2026", 2020, 2026)]


def labelled_ids():
    ids = {g["id"] for g in json.load(open(ANN / "gold_standard_2026-11.json"))}
    ids |= set(json.load(open(ANN / "supplementary_2026-11" / "item_ids.json")).values())
    ids |= set(json.load(open(ANN / "new_sample_300_ids.json")))
    ids |= {json.loads(line)["id"] for line in open(ANN / "splits" / "train_expanded.jsonl")}
    return ids


def stratum(date):
    year = int(str(date)[:4])
    return next(name for name, lo, hi in STRATA if lo <= year <= hi)


def main():
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} already exists; refusing to overwrite coded work.")
    excluded = labelled_ids()
    pool = {}
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in excluded or len(r.get("full_text") or "") < 500 or not r.get("date"):
                continue
            pool.setdefault(stratum(r["date"]), []).append(r)

    rng = random.Random(SEED)
    sample = []
    for name, _, _ in STRATA:
        members = sorted(pool[name], key=lambda r: r["id"])
        sample.extend(rng.sample(members, PER_STRATUM))
    rng.shuffle(sample)

    TEXTS.mkdir(parents=True, exist_ok=True)
    rows, items = [], []
    for i, r in enumerate(sample, 1):
        item = f"T2-{i:03d}"
        header = (
            f"ITEM: {item}\nCITATION: {r.get('citation')}\nTRIBUNAL: {r.get('tribunal')} "
            f"({r.get('jurisdiction')})\nDATE: {r.get('date')}\nURL: {r.get('url')}\n"
            + "=" * 78 + "\n\n"
        )
        (TEXTS / f"{item}.txt").write_text(header + r["full_text"], encoding="utf-8")
        rows.append((i, item, r["id"], r.get("citation"), r.get("jurisdiction"),
                     f"texts/{item}.txt", r.get("url")))
        items.append({"item": item, "decision_id": r["id"], "stratum": stratum(r["date"])})

    SAMPLE.write_text(json.dumps({
        "protocol": "config/external_test_protocol_2026-11.md",
        "seed": SEED,
        "n_excluded_labelled": len(excluded),
        "pool_size_by_stratum": {name: len(pool[name]) for name, _, _ in STRATA},
        "sample_size_by_stratum": dict(Counter(x["stratum"] for x in items)),
        "items": items,
    }, indent=2))

    professions = json.load(open(TAXONOMY))["professions"] + ["other", "unknown"]
    write_workbook(rows, professions, results=RESULTS, seed=SEED,
                   title="Second test set (Nov 2026): coder declaration")
    print(f"Wrote {len(rows)} texts to {TEXTS}, {RESULTS} and {SAMPLE} (seed {SEED}).")


if __name__ == "__main__":
    main()
