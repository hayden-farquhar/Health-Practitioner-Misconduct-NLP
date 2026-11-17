"""Build the adjudication workbook for the Nov 2026 gold-standard rebuild.

One row per disagreeing field between the author's blinded re-code and the logged model pass.
The author decides each row with both codes visible, so this step is recorded as UNBLINDED.
Fields on which the coders agree pass straight into the new gold standard and are not listed.

Usage (from project root):
    .venv-ml/bin/python -m src.annotation.build_adjudication_sheet
"""

import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src.analysis.irr_recode_2026 import AUTHOR, BINARY, MODEL, load_author, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data" / "annotations" / "adjudication_2026-11" / "P13_adjudication.xlsx"
FIELDS = ["is_disciplinary", "finding", "profession"] + BINARY


def main():
    author, model = load_author(), load_model()
    coding = list(load_workbook(AUTHOR, read_only=True)["Coding"].iter_rows(values_only=True))
    meta = {r[coding[0].index("item")]: dict(zip(coding[0], r)) for r in coding[1:]}
    model_notes = {d["item"]: d["notes"] for d in json.loads(MODEL.read_text())}

    wb = Workbook()
    info = wb.active
    info.title = "Read me"
    for line in [
        "Gold-standard adjudication (Nov 2026). This step is UNBLINDED: you see your own code and the model's.",
        "One row per field where your blinded re-code and the model pass disagree.",
        "Fields where the two agree go straight into the new gold standard and are not listed here.",
        "For each row, choose 'decision': AUTHOR (keep your code), MODEL (take the model's code), or",
        "OTHER (type the final value in 'final_if_other'). Add a short rationale where it helps.",
        "For Y/N codes, a blank means N. Open the decision in texts/ (recode_2026-10) if you need to re-read it.",
        "Fill in 'adjudicator' and 'date' on every row you decide. Do not edit grey columns.",
        "Your blinded codes are already recorded in the frozen re-code sheet and are not changed by this step.",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 110

    ws = wb.create_sheet("Adjudication")
    head = ["row", "item", "citation", "text_file", "field", "author_code", "model_code",
            "author_notes", "model_notes", "decision", "final_if_other", "rationale",
            "adjudicator", "date"]
    ws.append(head)
    n = 0
    for item in sorted(author):
        for f in FIELDS:
            if author[item][f] != model[item][f]:
                n += 1
                ws.append([n, item, meta[item]["citation"], meta[item]["text_file"], f,
                           author[item][f], model[item][f], meta[item].get("notes") or "",
                           model_notes[item], None, None, None, None, None])

    grey = PatternFill("solid", fgColor="E7E6E6")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2, max_col=9):
        for c in row:
            c.fill = grey
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for col, w in zip("ABCDEFGHIJKLMN", [5, 9, 18, 13, 16, 11, 11, 45, 45, 11, 13, 35, 16, 11]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "F2"
    dv = DataValidation(type="list", formula1='"AUTHOR,MODEL,OTHER"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"J2:J{n + 1}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {n} disagreement rows across "
          f"{len({r[1] for r in ws.iter_rows(min_row=2, values_only=True)})} items to {OUT}")


if __name__ == "__main__":
    main()
