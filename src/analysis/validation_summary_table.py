"""Manuscript Table 3: classifier and language-model coder performance on both test sets.

Reads only existing outputs: the first test set (table19, table29), the weighted second test set
(outputs/statistics/test2_results.json) and the logged coder on the second test set
(outputs/statistics/llm_test2_results.json). Writes outputs/tables/table37_validation_summary.csv.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.validation_summary_table
"""

import json
import os
from pathlib import Path

import pandas as pd

from src.analysis import validation_analysis as va

CATS = ["PROF", "FRAUD", "CLIN", "BOUND", "PRESC", "IMPR", "COMM"]
LABELS = {"PROF": "Professional conduct F1", "FRAUD": "Dishonesty/fraud F1",
          "CLIN": "Clinical competence F1", "BOUND": "Boundary violations F1",
          "PRESC": "Prescribing/dispensing F1", "IMPR": "Impairment F1", "COMM": "Communication F1"}


def fmt(v: dict) -> str:
    return f"{v['value']:.2f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"


def main():
    t19 = pd.read_csv(va.TABLES_DIR / "table19_classification_performance.csv")
    first = {(r.Task, r.Metric): r.Value for r in t19.itertuples()}
    f1ci = pd.read_csv(va.TABLES_DIR / "table29_f1_bootstrap_ci.csv").set_index("code")
    t2 = json.loads((va.STATS_DIR / "test2_results.json").read_text())["primary_weighted"]
    llm = json.loads((va.STATS_DIR / "llm_test2_results.json").read_text())["primary_weighted"]

    rows = [
        ("Is-disciplinary accuracy", f"{first[('Is-disciplinary (binary)', 'Accuracy')]:.2f}",
         "is_disciplinary_accuracy"),
        ("Misconduct macro-F1", f"{first[('Misconduct L1 (multi-label)', 'Macro F1')]:.2f}",
         "misconduct_macro_f1"),
        ("Misconduct micro-F1", f"{first[('Misconduct L1 (multi-label)', 'Micro F1')]:.2f}",
         "misconduct_micro_f1"),
        ("Finding accuracy", f"{first[('Finding (single-label)', 'Accuracy')]:.2f}", "finding_accuracy"),
        ("Outcome micro-F1", f"{first[('Outcome extraction (regex)', 'Micro F1')]:.2f}",
         "outcome_micro_f1"),
    ] + [(LABELS[c], "{:.2f} ({:.2f}-{:.2f})".format(*f1ci.loc[c, ["f1", "ci_lower", "ci_upper"]]),
          f"{c}_f1") for c in CATS]

    out = pd.DataFrame([{"Measure": name,
                         "Classifier, first test set (n = 45)": first_val,
                         "Classifier, second test set (n = 160, weighted)": fmt(t2[key]),
                         "Language-model coder, second test set (n = 160, weighted)": fmt(llm[key])}
                        for name, first_val, key in rows])
    out.to_csv(Path(os.path.relpath(va.TABLES_DIR)) / "table37_validation_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
