#!/usr/bin/env python3
"""
Completeness and consistency check for the Oct 2026 gold-standard re-code.

Reports blank or internally inconsistent rows in
data/annotations/recode_2026-10/P13_recode_results.xlsx (or the workbook given). If the
workbook has a text_check column, it also checks that each entry appears in that row's text
file. It never judges the codes themselves and never writes to the workbook.

Usage (from project root):
    python3 src/preprocessing/check_recode.py [path/to/results.xlsx]
Exit 0 = complete and consistent; exit 1 = problems listed.
"""

import sys
from pathlib import Path

from openpyxl import load_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "data" / "annotations" / "recode_2026-10" / "P13_recode_results.xlsx"

MISCONDUCT = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF"]
BOUND_SUB = ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]


def blank(v):
    return v is None or str(v).strip() == ""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RESULTS
    wb = load_workbook(path, read_only=True)
    problems = []

    proto = {r[0]: (r[1] if len(r) > 1 else None)
             for r in wb["Protocol"].iter_rows(values_only=True) if r and r[0]}
    for key in ["Coder (full name)", "Date started", "Date finished"]:
        if blank(proto.get(key)):
            problems.append(f"Protocol: '{key}' is blank")
    for key, value in proto.items():
        if str(key).startswith("I ") and str(value).strip().upper() not in ("Y", "N"):
            problems.append(f"Protocol: declaration not answered: '{key}'")

    folder = Path(path).parent
    rows = list(wb["Coding"].iter_rows(values_only=True))
    head = list(rows[0])
    done = 0
    uncoded = []
    for r in rows[1:]:
        d = dict(zip(head, r))
        item = d["item"]
        ticked = lambda cols: [c for c in cols if str(d.get(c) or "").strip().upper() == "Y"]
        if blank(d.get("is_disciplinary")):
            uncoded.append(item)
            continue
        done += 1
        for f in ["finding", "profession", "confidence", "date_coded"]:
            if blank(d.get(f)):
                problems.append(f"{item}: '{f}' blank")
        if d["is_disciplinary"] == "N":
            if d.get("finding") != "NA":
                problems.append(f"{item}: is_disciplinary=N but finding is not NA")
            extra = ticked(MISCONDUCT + BOUND_SUB + OUTCOMES)
            if extra:
                problems.append(f"{item}: is_disciplinary=N but ticked {extra}")
        else:
            if d.get("finding") == "NA":
                problems.append(f"{item}: is_disciplinary=Y but finding is NA")
            if not ticked(OUTCOMES):
                problems.append(f"{item}: no outcome ticked")
            if not ticked(MISCONDUCT) and blank(d.get("notes")):
                problems.append(f"{item}: no misconduct type ticked; add a note if intended")
        if "text_check" in d:
            check = str(d.get("text_check") or "").strip()
            text = (folder / d["text_file"]).read_text(encoding="utf-8").lower()
            if not check:
                problems.append(f"{item}: 'text_check' blank")
            elif check.lower() != "unnamed" and check.lower() not in text:
                problems.append(f"{item}: text_check '{check}' does not appear in {d['text_file']}")
        if ticked(BOUND_SUB) and "BOUND" not in ticked(MISCONDUCT):
            problems.append(f"{item}: boundary subtype ticked without BOUND")

    print(f"Coded rows: {done}/{len(rows) - 1}")
    if uncoded:
        problems.append(f"{len(uncoded)} row(s) not coded yet, first: {uncoded[0]}")
    for p in problems:
        print(" -", p)
    print("OK" if not problems else f"{len(problems)} issue(s)")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
