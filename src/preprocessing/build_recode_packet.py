#!/usr/bin/env python3
"""
Build the blinded re-coding packet for the 300-decision gold standard (Oct 2026).

Why: the original "human" annotations were partly produced by AI coding agents,
so the gold standard is being re-coded in full by the
author, blind to every earlier label and machine prediction.

The packet contains only the decision text and neutral metadata (citation,
tribunal, jurisdiction, date, URL). It never includes the existing human/LLM/gold
labels, the regex outcome extraction, or the enriched profession field.

Usage (from project root):
    python3 src/preprocessing/build_recode_packet.py
Refuses to overwrite an existing results workbook.
"""

import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD = PROJECT_ROOT / "data" / "annotations" / "gold_standard.json"
CORPUS = PROJECT_ROOT / "data" / "extracted" / "all_decisions.jsonl"
TAXONOMY = PROJECT_ROOT / "config" / "taxonomy.json"
OUT = PROJECT_ROOT / "data" / "annotations" / "recode_2026-10"
TEXTS = OUT / "texts"
RESULTS = OUT / "P13_recode_results.xlsx"
SEED = 20261017

MISCONDUCT = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF"]
BOUND_SUB = ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]
FINDINGS = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING", "NA"]


def main():
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} already exists; refusing to overwrite coded work.")
    gold_ids = [g["id"] for g in json.load(open(GOLD))]
    want = set(gold_ids)
    docs = {}
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                docs[r["id"]] = r
    missing = want - docs.keys()
    if missing:
        raise SystemExit(f"{len(missing)} gold ids missing from corpus: {sorted(missing)[:5]}")

    order = sorted(gold_ids)
    random.Random(SEED).shuffle(order)

    TEXTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, doc_id in enumerate(order, 1):
        r = docs[doc_id]
        item = f"HPM-{i:03d}"
        header = (
            f"ITEM: {item}\nCITATION: {r.get('citation')}\nTRIBUNAL: {r.get('tribunal')} "
            f"({r.get('jurisdiction')})\nDATE: {r.get('date')}\nURL: {r.get('url')}\n"
            + "=" * 78 + "\n\n"
        )
        (TEXTS / f"{item}.txt").write_text(header + (r.get("full_text") or ""), encoding="utf-8")
        rows.append((i, item, doc_id, r.get("citation"), r.get("jurisdiction"),
                     f"texts/{item}.txt", r.get("url")))

    professions = json.load(open(TAXONOMY))["professions"] + ["other", "unknown"]
    write_workbook(rows, professions)
    print(f"Wrote {len(rows)} texts to {TEXTS} and {RESULTS} (seed {SEED}).")


def write_workbook(rows, professions, results=RESULTS, seed=SEED,
                   title="Gold-standard re-code (Oct 2026): coder declaration",
                   extra_coded=(), extra_declarations=()):
    wb = Workbook()
    ws = wb.active
    ws.title = "Coding"
    fixed = ["order", "item", "decision_id", "citation", "jurisdiction", "text_file", "url"]
    coded = (list(extra_coded) + ["is_disciplinary", "finding"] + MISCONDUCT + BOUND_SUB + OUTCOMES
             + ["profession", "confidence", "notes", "date_coded"])
    ws.append(fixed + coded)
    for r in rows:
        ws.append(list(r))

    grey = PatternFill("solid", fgColor="EEEEEE")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for col in range(1, len(fixed) + 1):
        for cell in ws.iter_cols(min_col=col, max_col=col, min_row=2):
            for c in cell:
                c.fill = grey
    ws.freeze_panes = "E2"
    n = len(rows) + 1

    lists = wb.create_sheet("Lists")
    lists.append(["profession"])
    for p in professions:
        lists.append([p])

    def col(name):
        return ws.cell(row=1, column=(fixed + coded).index(name) + 1).column_letter

    def add(formula, names):
        dv = DataValidation(type="list", formula1=formula, allow_blank=True)
        ws.add_data_validation(dv)
        for name in names:
            dv.add(f"{col(name)}2:{col(name)}{n}")

    add('"Y,N"', ["is_disciplinary"])
    add('"' + ",".join(FINDINGS) + '"', ["finding"])
    add('"Y"', MISCONDUCT + BOUND_SUB + OUTCOMES)
    add(f"Lists!$A$2:$A${len(professions) + 1}", ["profession"])
    add('"High,Medium,Low"', ["confidence"])

    proto = wb.create_sheet("Protocol", 0)
    for line in [
        [title],
        [],
        ["Coder (full name)", ""],
        ["Date started", ""],
        ["Date finished", ""],
        ["I coded every row myself, without AI or script assistance (Y/N)", ""],
        ["I did not open earlier labels or predictions while coding (Y/N)", ""],
        *[[d, ""] for d in extra_declarations],
        ["If any answer is N, describe the exception", ""],
        [],
        ["Instructions: CODING_INSTRUCTIONS.md in this folder."],
        [f"Item order randomised with seed {seed}. Do not re-sort the Coding sheet."],
    ]:
        proto.append(line)
    proto["A1"].font = Font(bold=True, size=13)
    proto.column_dimensions["A"].width = 70
    proto.column_dimensions["B"].width = 40
    wb.save(results)


if __name__ == "__main__":
    main()
