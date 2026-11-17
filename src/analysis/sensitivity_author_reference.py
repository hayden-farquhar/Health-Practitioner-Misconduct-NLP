"""Post-hoc sensitivity analysis: score the chosen models against the author's blind re-code alone.

The gold standard combines the author's blind re-code with the logged model pass, and the same
model labelled the 300 supplementary training decisions. To check whether that shared source
flatters the test score, the models chosen under config/retrain_protocol_2026-11.md are scored
on the same 45 test decisions against the author's frozen blind codes, with no adjudication and
no model input. Requested by Hayden on 2026-11-05, after the main test scoring; it was not
pre-specified, and nothing is re-selected on it.

Uses validation_analysis.classification_validation unchanged, so metric definitions match the
main analysis. Its figures and tables go to outputs/*/sensitivity_author_reference/.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.sensitivity_author_reference
"""

import json

import pandas as pd
from openpyxl import load_workbook

from src.analysis import validation_analysis as va
from src.analysis.irr_recode_2026 import AUTHOR, MISCONDUCT, OUTCOMES, load_author

OUT_JSON = va.STATS_DIR / "sensitivity_author_reference.json"


def author_test_df(test: pd.DataFrame) -> pd.DataFrame:
    """The test split with labels replaced by the author's blind codes (gold-standard schema)."""
    author = load_author()
    coding = list(load_workbook(AUTHOR, read_only=True)["Coding"].iter_rows(values_only=True))
    item_of = {r[coding[0].index("decision_id")]: r[coding[0].index("item")] for r in coding[1:]}
    rows = []
    for _, t in test.iterrows():
        a = author[item_of[t["id"]]]
        disc = a["is_disciplinary"] == "Y"
        rows.append({**t.to_dict(),
                     "is_disciplinary": disc,
                     "misconduct_L1": [k for k in MISCONDUCT if a[k] == "Y"] if disc else ["N/A"],
                     "finding": None if a["finding"] == "NA" else a["finding"],
                     "outcomes": [k for k in OUTCOMES if a[k] == "Y"] if disc else ["N/A"]})
    return pd.DataFrame(rows)


def main():
    for attr in ("FIGURES_DIR", "TABLES_DIR"):
        d = getattr(va, attr) / "sensitivity_author_reference"
        d.mkdir(parents=True, exist_ok=True)
        setattr(va, attr, d)

    gold_test = va.load_split("test")
    test = author_test_df(gold_test)
    differ = {f: int(sum(str(g) != str(a) for g, a in zip(gold_test[f], test[f])))
              for f in ("is_disciplinary", "misconduct_L1", "finding", "outcomes")}
    print(f"Test decisions whose label differs from the gold standard: {differ}")

    results = va.classification_validation(test, *va.load_models()[:5],
                                           vectorizer_finding=va.load_models()[5])
    main_results = json.loads((va.STATS_DIR / "validation_results.json").read_text())["classification"]
    OUT_JSON.write_text(json.dumps({
        "reference": "author's blind re-code (frozen 2026-11-03), no adjudication",
        "pre_specified": False,
        "requested": "Hayden Farquhar, 2026-11-05, after the main test scoring",
        "n_test": len(test),
        "test_decisions_with_label_differing_from_gold": differ,
        "classification": results,
        "main_analysis_for_comparison": {
            "is_disciplinary_accuracy": main_results["is_disciplinary"]["accuracy"],
            "misconduct_macro_f1": main_results["misconduct"]["macro_f1"],
            "misconduct_micro_f1": main_results["misconduct"]["micro_f1"],
            "finding_accuracy": main_results["finding"]["accuracy"],
            "outcome_micro_f1": main_results["outcome_extraction"]["micro_f1"],
        },
    }, indent=2, default=str))
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
