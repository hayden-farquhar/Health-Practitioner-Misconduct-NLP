"""Completeness and consistency check for the Nov 2026 adjudication workbook.

Lists undecided rows, OTHER rows with no final value, missing rationale/adjudicator/date, and
decisions whose final codes (agreed fields + adjudicated fields) break the coding rules. It
never judges the decisions themselves and never writes to the workbook.

Usage (from project root):
    .venv-ml/bin/python -m src.annotation.check_adjudication
Exit 0 = complete and consistent; exit 1 = problems listed.
"""

import sys

from openpyxl import load_workbook

from src.analysis.irr_recode_2026 import BINARY, BOUND_SUB, MISCONDUCT, OUTCOMES, load_author
from src.annotation.build_adjudication_sheet import OUT as WORKBOOK
from src.annotation.model_second_coder import FINDINGS, PROFESSIONS


def blank(v):
    return v is None or str(v).strip() == ""


def main():
    rows = list(load_workbook(WORKBOOK, read_only=True)["Adjudication"].iter_rows(values_only=True))
    head = rows[0]
    final = {i: dict(r) for i, r in load_author().items()}  # start from author; agreed fields equal both
    problems, notes, decided = [], [], 0

    for r in rows[1:]:
        d = dict(zip(head, r))
        tag = f"row {d['row']} ({d['item']} {d['field']})"
        choice = str(d["decision"] or "").strip().upper()
        if choice not in ("AUTHOR", "MODEL", "OTHER"):
            problems.append(f"{tag}: no decision")
            continue
        decided += 1
        if choice == "AUTHOR":
            value = d["author_code"]
        elif choice == "MODEL":
            value = d["model_code"]
        else:
            value = d["final_if_other"]
            if blank(value):
                problems.append(f"{tag}: OTHER with no final value")
                continue
        value = str(value).strip()
        if d["field"] in BINARY:
            value = "Y" if value.upper() == "Y" else "N"
        final[d["item"]][d["field"]] = value
        if choice in ("MODEL", "OTHER") and blank(d["rationale"]):
            problems.append(f"{tag}: rationale required for {choice}")
        for col in ("adjudicator", "date"):
            if blank(d[col]):
                problems.append(f"{tag}: '{col}' blank")

    for item, c in sorted(final.items()):
        ticked = lambda codes: [k for k in codes if c[k] == "Y"]
        if c["finding"] not in FINDINGS:
            problems.append(f"{item}: finding '{c['finding']}' not a valid code")
        if c["profession"] not in PROFESSIONS:
            problems.append(f"{item}: profession '{c['profession']}' not a valid code")
        if c["is_disciplinary"] == "N":
            if c["finding"] != "NA":
                problems.append(f"{item}: non-disciplinary but finding is {c['finding']}")
            if ticked(BINARY):
                problems.append(f"{item}: non-disciplinary but codes ticked: {ticked(BINARY)}")
        else:
            if c["finding"] == "NA":
                problems.append(f"{item}: disciplinary but finding is NA")
            if not ticked(MISCONDUCT):
                if c["finding"] == "NO_FINDING":
                    notes.append(f"{item}: NO_FINDING with no misconduct code (confirm intended)")
                else:
                    problems.append(f"{item}: disciplinary with no misconduct code")
            if not ticked(OUTCOMES):
                problems.append(f"{item}: disciplinary with no outcome")
        if c["BOUND"] != "Y" and ticked(BOUND_SUB):
            problems.append(f"{item}: boundary subtype ticked without BOUND")

    print(f"Decided rows: {decided}/{len(rows) - 1}")
    for p in problems:
        print(" -", p)
    for n in notes:
        print(" - note:", n)
    print("OK" if not problems else f"{len(problems)} issue(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
