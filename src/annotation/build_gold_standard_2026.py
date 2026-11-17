"""Freeze the Nov 2026 adjudication and build the new 300-decision gold standard.

Final value per field = the value both coders gave (author's blinded re-code and the logged
model pass agree), or, where they disagree, the author's adjudicated value (AUTHOR / MODEL /
OTHER) from the adjudication workbook. Each field records which of those it came from.

Output follows the schema of the Feb 2026 gold_standard.json (which it replaces), so
data_preparation.py can read it unchanged: non-disciplinary decisions carry ["N/A"] for
misconduct and outcomes, and finding NA is written as null. It adds boundary_subtypes and
per-field provenance.

Post-adjudication amendments (post_adjudication_amendments.json, beside the workbook) are
applied last. Each is a dated decision by the author on a field the sheet did not list, and the
field's provenance becomes "amended".

Usage (from project root, after check_adjudication exits 0):
    .venv-ml/bin/python -m src.annotation.build_gold_standard_2026
Freezes the workbook on the first run; later runs require the workbook to match the frozen copy.
"""

import hashlib
import json
import shutil
import stat
from pathlib import Path

from openpyxl import load_workbook

from src.analysis.irr_recode_2026 import (AUTHOR, BINARY, BOUND_SUB, MISCONDUCT, MODEL, OUTCOMES,
                                          load_author, load_model)
from src.annotation.build_adjudication_sheet import FIELDS, OUT as WORKBOOK

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN = WORKBOOK.with_name("P13_adjudication_FROZEN_2026-11-04.xlsx")
GOLD = PROJECT_ROOT / "data" / "annotations" / "gold_standard_2026-11.json"
MANIFEST = GOLD.with_name("gold_standard_2026-11_manifest.json")
AMENDMENTS = WORKBOOK.with_name("post_adjudication_amendments.json")


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main():
    if not FROZEN.exists():
        shutil.copy2(WORKBOOK, FROZEN)
        FROZEN.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    elif sha1(FROZEN) != sha1(WORKBOOK):
        raise SystemExit("working workbook differs from the frozen copy; refusing to rebuild")
    amendments = json.loads(AMENDMENTS.read_text()) if AMENDMENTS.exists() else []

    author, model = load_author(), load_model()
    coding = list(load_workbook(AUTHOR, read_only=True)["Coding"].iter_rows(values_only=True))
    ids = {r[coding[0].index("item")]: r[coding[0].index("decision_id")] for r in coding[1:]}

    rows = list(load_workbook(FROZEN, read_only=True)["Adjudication"].iter_rows(values_only=True))
    adjudicated = {}
    for r in rows[1:]:
        d = dict(zip(rows[0], r))
        # The grey columns must still match both coders' frozen codes.
        if (d["author_code"], d["model_code"]) != (author[d["item"]][d["field"]], model[d["item"]][d["field"]]):
            raise SystemExit(f"row {d['row']}: grey columns differ from the frozen codes")
        choice = d["decision"].strip().upper()
        value = {"AUTHOR": d["author_code"], "MODEL": d["model_code"]}.get(choice, d["final_if_other"])
        value = str(value).strip()
        if d["field"] in BINARY:
            value = "Y" if value.upper() == "Y" else "N"
        adjudicated[(d["item"], d["field"])] = (value, choice, d["rationale"], str(d["date"])[:10])

    disagreements = sum(author[i][f] != model[i][f] for i in author for f in FIELDS)
    if disagreements != len(adjudicated):
        raise SystemExit(f"{disagreements} disagreements but {len(adjudicated)} adjudicated rows")

    gold = []
    for item in sorted(author, key=lambda i: ids[i]):
        final, source = {}, {}
        for f in FIELDS:
            if (item, f) in adjudicated:
                final[f], choice, _, _ = adjudicated[(item, f)]
                source[f] = f"adjudicated_{choice.lower()}"
            else:
                final[f], source[f] = author[item][f], "agreed"
        for a in amendments:
            if a["item"] == item:
                if final[a["field"]] != a["from"]:
                    raise SystemExit(f"amendment {item} {a['field']}: expected {a['from']}, found {final[a['field']]}")
                final[a["field"]], source[a["field"]] = a["to"], "amended"
        disc = final["is_disciplinary"] == "Y"
        ticked = lambda codes: [k for k in codes if final[k] == "Y"]
        gold.append({
            "id": ids[item],
            "item": item,
            "is_disciplinary": disc,
            "misconduct_L1": ticked(MISCONDUCT) if disc else ["N/A"],
            "boundary_subtypes": ticked(BOUND_SUB),
            "finding": None if final["finding"] == "NA" else final["finding"],
            "outcomes": ticked(OUTCOMES) if disc else ["N/A"],
            "profession": final["profession"],
            "provenance": source,
            "adjudication_rationales": {f: adjudicated[(item, f)][2] for f in FIELDS if (item, f) in adjudicated},
        })

    GOLD.write_text(json.dumps(gold, indent=1))
    counts = {"agreed": 0, "adjudicated_author": 0, "adjudicated_model": 0, "adjudicated_other": 0,
              "amended": 0}
    for g in gold:
        for s in g["provenance"].values():
            counts[s] += 1
    manifest = {
        "built": "2026-11-04",  # date of the adjudication freeze and the HPM-257 amendment
        "coder_1": {"who": "Hayden Farquhar (author)", "blinded": True, "coded": "2026-11-01 to 2026-11-03",
                    "file": str(AUTHOR.relative_to(PROJECT_ROOT)), "sha1": sha1(AUTHOR)},
        "coder_2": {"who": "logged model pass, batch msgbatch_01P6SW9pMypxs8YJEFcPM1EP", "blinded": True,
                    "file": str(MODEL.relative_to(PROJECT_ROOT)), "sha1": sha1(MODEL)},
        "adjudication": {"who": "Hayden Farquhar (author)", "blinded": False,
                         "dates": sorted({v[3] for v in adjudicated.values()}),
                         "file": str(FROZEN.relative_to(PROJECT_ROOT)), "sha1": sha1(FROZEN)},
        "amendments": {"file": str(AMENDMENTS.relative_to(PROJECT_ROOT)), "n": len(amendments),
                       "sha1": sha1(AMENDMENTS) if amendments else None},
        "n_items": len(gold), "n_fields_per_item": len(FIELDS),
        "field_sources": counts,
        "gold_standard": {"file": str(GOLD.relative_to(PROJECT_ROOT)), "sha1": sha1(GOLD)},
        "replaces": "data/annotations/gold_standard.json (Feb 2026; 208/300 'human' codes were Claude agents)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
