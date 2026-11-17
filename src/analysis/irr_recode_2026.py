"""Inter-rater agreement for the Nov 2026 gold-standard rebuild.

Coder 1: the author's blinded re-code (frozen copy P13_recode_results_SUBMITTED_2026-11-03.xlsx).
Coder 2: the logged model pass (data/annotations/model_pass_2026-11/model_codes.json).

Cohen's kappa with 95% percentile bootstrap CIs (2,000 resamples, seed 20261103), computed
for every field on all items both coders coded, and for the misconduct, boundary-subtype and
outcome codes again on the subset both coders called disciplinary. Items the model did not
code (refusals, errors) are excluded and counted.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.irr_recode_2026
"""

import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHOR = (PROJECT_ROOT / "data" / "annotations" / "recode_2026-10"
          / "P13_recode_results_SUBMITTED_2026-11-03.xlsx")
MODEL = PROJECT_ROOT / "data" / "annotations" / "model_pass_2026-11" / "model_codes.json"
OUT = PROJECT_ROOT / "outputs" / "statistics" / "irr_recode_2026.json"

MISCONDUCT = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF"]
BOUND_SUB = ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]
BINARY = MISCONDUCT + BOUND_SUB + OUTCOMES
N_BOOT, SEED = 2000, 20261103


def norm(v):
    return "" if v is None else str(v).strip()


def load_author():
    rows = list(load_workbook(AUTHOR, read_only=True)["Coding"].iter_rows(values_only=True))
    head = rows[0]
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        rec = {k: norm(d.get(k)) for k in ["is_disciplinary", "finding", "profession"]}
        rec.update({k: "Y" if norm(d.get(k)).upper() == "Y" else "N" for k in BINARY})
        out[d["item"]] = rec
    return out


def load_model():
    out = {}
    for d in json.loads(MODEL.read_text()):
        rec = {k: d[k] for k in ["is_disciplinary", "finding", "profession"]}
        rec.update({k: "Y" if d[k] == "Y" else "N" for k in BINARY})
        out[d["item"]] = rec
    return out


def kappa_ci(a, b, rng):
    a, b = np.asarray(a), np.asarray(b)
    agree = float(np.mean(a == b))
    if len(set(a) | set(b)) < 2:
        return {"n": len(a), "kappa": None, "ci95": None, "pct_agree": agree,
                "note": "only one category used by both coders"}
    k = cohen_kappa_score(a, b)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(a), len(a))
        if len(set(a[idx]) | set(b[idx])) > 1:
            boots.append(cohen_kappa_score(a[idx], b[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"n": len(a), "kappa": round(float(k), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "pct_agree": round(agree, 3),
            "positives_author": int(np.sum(a == "Y")) if set(a) <= {"Y", "N"} else None,
            "positives_model": int(np.sum(b == "Y")) if set(b) <= {"Y", "N"} else None}


def main():
    author, model = load_author(), load_model()
    items = sorted(set(author) & set(model))
    missing = sorted(set(author) - set(model))
    rng = np.random.default_rng(SEED)

    def field(f, subset):
        return kappa_ci([author[i][f] for i in subset], [model[i][f] for i in subset], rng)

    both_disc = [i for i in items
                 if author[i]["is_disciplinary"] == "Y" and model[i]["is_disciplinary"] == "Y"]
    results = {
        "coder_1": "author, blinded re-code 1-3 Nov 2026 (frozen sheet)",
        "coder_2": "model pass, see data/annotations/model_pass_2026-11/batch_manifest.json",
        "n_compared": len(items), "model_missing": missing,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "all_items": {f: field(f, items) for f in ["is_disciplinary", "finding", "profession"] + BINARY},
        "both_disciplinary": {"n": len(both_disc), **{f: field(f, both_disc) for f in ["finding"] + BINARY}},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print(f"Compared {len(items)} items; model missing {len(missing)}; both disciplinary {len(both_disc)}")
    for scope in ["all_items", "both_disciplinary"]:
        print(f"\n{scope}")
        for f, r in results[scope].items():
            if f == "n":
                continue
            ci = f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]" if r["ci95"] else ""
            k = f"{r['kappa']:.3f}" if r["kappa"] is not None else "  n/a"
            print(f"  {f:17} kappa {k} {ci:15} agree {r['pct_agree']:.1%}")


if __name__ == "__main__":
    main()
