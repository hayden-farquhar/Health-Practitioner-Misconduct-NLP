"""Agreement between the author and the independent second coder (config/second_coder_protocol_2026-11.md).

Coder 1: the author's frozen test2 re-code (P13_test2_results_SUBMITTED_2026-11-07.xlsx).
Coder 2: the Prolific second coder, two sessions of 30 (returns_frozen/, sha256 pinned below).
Unit: the 60 sampled decisions (private/mapping.json maps D01-D60 to test2 items).

Pre-specified analyses (protocol section "Analysis"):
  1. Cohen's kappa with 95% percentile bootstrap CI (2,000 resamples, seed 20261109) and raw
     agreement for disciplinary status, finding, the 7 misconduct types, the 3 boundary subtypes
     (among decisions both coders coded BOUND), the 9 outcomes and profession. A field ticked on
     a decision the coder called non-disciplinary counts as absent. Binary fields with fewer
     than 5 positives from either coder are reported descriptively, without kappa.
  2. Unweighted prevalence of each misconduct type among the decisions each coder called
     disciplinary, with bootstrap CIs, and the paired difference (second coder minus author).
  3. Disagreements listed by field (classification into definitional / reading / scope is a
     separate, descriptive step; nothing is adjudicated and no code is changed).
Also reported, labelled exploratory (not pre-specified): agreement by session, and pace.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.second_coder_agreement_2026
"""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SC = PROJECT_ROOT / "data" / "annotations" / "second_coder_2026-11"
AUTHOR = PROJECT_ROOT / "data" / "annotations" / "test2_2026-11" / "P13_test2_results_SUBMITTED_2026-11-07.xlsx"
CODER = {
    1: SC / "returns_frozen" / "69a5959624b49db31dba9e4b_session1_RECEIVED_2026-11-16.csv",
    2: SC / "returns_frozen" / "69a5959624b49db31dba9e4b_session2_RECEIVED_2026-11-17.csv",
}
MAPPING = SC / "private" / "mapping.json"
OUT = PROJECT_ROOT / "outputs" / "statistics" / "second_coder_agreement_2026.json"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables" / "second_coder"

PINNED = {  # sha1 for the author sheet (as in the protocol), sha256 for the coder returns
    AUTHOR: ("sha1", "b42fd27e8e3756ee1beae2a0f0033c8ab0210a6a"),
    CODER[1]: ("sha256", "2a4f903ef802d1cb1d6030db46bdb177e4e6e99bb18f35849e63c4a96ae4dbdd"),
    CODER[2]: ("sha256", "efcec77687c7223d142bff0b9e0e5dbed5613709ee4a0444724d612ab1177458"),
}

MISCONDUCT = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF"]
BOUND_SUB = ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]
BINARY = MISCONDUCT + BOUND_SUB + OUTCOMES
N_BOOT, SEED, MIN_POS = 2000, 20261109, 5


def norm(v):
    return "" if v is None else str(v).strip()


def check_pins():
    bad = {}
    for p, (algo, h) in PINNED.items():
        got = hashlib.new(algo, p.read_bytes()).hexdigest()
        if got != h:
            bad[p.name] = got
    if bad:
        raise SystemExit(f"Frozen inputs changed: {bad}")


def record(d):
    """One coder's codes for one decision, with ticks masked on non-disciplinary decisions."""
    disc = norm(d.get("is_disciplinary")).upper() == "Y"
    rec = {"is_disciplinary": "Y" if disc else "N",
           "finding": norm(d.get("finding")), "profession": norm(d.get("profession")),
           "notes": norm(d.get("notes"))}
    rec.update({k: "Y" if disc and norm(d.get(k)).upper() == "Y" else "N" for k in BINARY})
    return rec


def load():
    mapping = json.loads(MAPPING.read_text())
    rows = list(load_workbook(AUTHOR, read_only=True)["Coding"].iter_rows(values_only=True))
    author_all = {r[1]: dict(zip(rows[0], r)) for r in rows[1:]}
    coder_all, secs = {}, {}
    for s, path in CODER.items():
        for d in csv.DictReader(path.open(encoding="utf-8-sig")):
            coder_all[d["doc"]] = d
            secs[d["doc"]] = float(d["seconds_on_doc"])
    docs = []
    for m in mapping:
        if m["doc"] not in coder_all:
            raise SystemExit(f"{m['doc']} missing from the coder returns")
        docs.append({**m, "author": record(author_all[m["item"]]),
                     "coder": record(coder_all[m["doc"]]), "seconds": secs[m["doc"]],
                     "citation": norm(author_all[m["item"]].get("citation"))})
    if len(docs) != 60:
        raise SystemExit(f"expected 60 decisions, found {len(docs)}")
    return docs


def kappa_ci(a, b, rng, binary):
    a, b = np.asarray(a), np.asarray(b)
    out = {"n": len(a), "pct_agree": round(float(np.mean(a == b)), 3)}
    if binary:
        out.update(positives_author=int(np.sum(a == "Y")), positives_coder=int(np.sum(b == "Y")))
        if min(out["positives_author"], out["positives_coder"]) < MIN_POS:
            return {**out, "kappa": None, "ci95": None,
                    "note": f"fewer than {MIN_POS} positives from a coder: descriptive only"}
    if len(set(a) | set(b)) < 2:
        return {**out, "kappa": None, "ci95": None, "note": "only one category used by both coders"}
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(a), len(a))
        if len(set(a[idx]) | set(b[idx])) > 1:
            boots.append(cohen_kappa_score(a[idx], b[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {**out, "kappa": round(float(cohen_kappa_score(a, b)), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def agreement(docs, rng):
    def field(f, subset, binary):
        return kappa_ci([d["author"][f] for d in subset], [d["coder"][f] for d in subset], rng, binary)

    both_bound = [d for d in docs if d["author"]["BOUND"] == "Y" and d["coder"]["BOUND"] == "Y"]
    res = {f: field(f, docs, False) for f in ["is_disciplinary", "finding", "profession"]}
    res.update({f: field(f, docs, True) for f in MISCONDUCT + OUTCOMES})
    res["boundary_subtypes_among_both_BOUND"] = {
        "n": len(both_bound), **{f: field(f, both_bound, True) for f in BOUND_SUB}}
    return res


def prevalence(docs, rng):
    """Each coder's prevalence among the decisions that coder called disciplinary (unweighted)."""
    A = np.array([[d["author"][k] == "Y" for k in MISCONDUCT] for d in docs])
    C = np.array([[d["coder"][k] == "Y" for k in MISCONDUCT] for d in docs])
    da = np.array([d["author"]["is_disciplinary"] == "Y" for d in docs])
    dc = np.array([d["coder"]["is_disciplinary"] == "Y" for d in docs])

    def prev(X, disc):
        return X[disc].mean(axis=0) if disc.any() else np.full(len(MISCONDUCT), np.nan)

    pa, pc = prev(A, da), prev(C, dc)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(docs), len(docs))
        ba, bc = prev(A[idx], da[idx]), prev(C[idx], dc[idx])
        boots.append(np.stack([ba, bc, bc - ba]))
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)

    def ci(j, i):
        return [round(float(lo[j, i]), 3), round(float(hi[j, i]), 3)]

    return {"n_disciplinary_author": int(da.sum()), "n_disciplinary_coder": int(dc.sum()),
            **{k: {"author": round(float(pa[i]), 3), "author_ci95": ci(0, i),
                   "coder": round(float(pc[i]), 3), "coder_ci95": ci(1, i),
                   "difference_coder_minus_author": round(float(pc[i] - pa[i]), 3),
                   "difference_ci95": ci(2, i)}
               for i, k in enumerate(MISCONDUCT)}}


def disagreements(docs):
    out = []
    both_bound = {d["doc"] for d in docs if d["author"]["BOUND"] == "Y" and d["coder"]["BOUND"] == "Y"}
    for d in docs:
        fields = ["is_disciplinary", "finding", "profession"] + MISCONDUCT + OUTCOMES
        fields += BOUND_SUB if d["doc"] in both_bound else []
        for f in fields:
            if d["author"][f] != d["coder"][f]:
                out.append({"doc": d["doc"], "item": d["item"], "citation": d["citation"],
                            "session": d["session"], "stratum": d["stratum"], "field": f,
                            "author": d["author"][f], "coder": d["coder"][f],
                            "author_notes": d["author"]["notes"], "coder_notes": d["coder"]["notes"],
                            "classification": ""})
    return out


def pace(docs):
    out = {}
    for s in (1, 2):
        v = [d["seconds"] for d in docs if d["session"] == s]
        out[f"session{s}"] = {"n": len(v), "median_seconds": statistics.median(v),
                              "min_seconds": min(v), "max_seconds": max(v),
                              "total_hours": round(sum(v) / 3600, 2)}
    return out


def main():
    check_pins()
    docs = load()
    rng = np.random.default_rng(SEED)
    results = {
        "protocol": "config/second_coder_protocol_2026-11.md",
        "coder_1": "author, frozen test2 re-code (sha1 b42fd27e...)",
        "coder_2": "independent second coder recruited on Prolific (pseudonymous ID not published)",
        "n": len(docs), "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "min_positives_for_kappa": MIN_POS,
        "agreement": agreement(docs, rng),
        "prevalence": prevalence(docs, rng),
        "exploratory_not_prespecified": {
            "agreement_by_session": {f"session{s}": agreement([d for d in docs if d["session"] == s], rng)
                                     for s in (1, 2)},
            "pace": pace(docs),
        },
    }
    dis = disagreements(docs)
    results["n_disagreements"] = len(dis)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUT_TABLES / "disagreements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dis[0]) if dis else ["doc"])
        w.writeheader()
        w.writerows(dis)

    print(f"n={len(docs)}; disagreements (field-level) {len(dis)}")
    ag = results["agreement"]
    for f, r in {**{k: v for k, v in ag.items() if k != "boundary_subtypes_among_both_BOUND"},
                 **{f"{k} (both BOUND, n={ag['boundary_subtypes_among_both_BOUND']['n']})": v
                    for k, v in ag["boundary_subtypes_among_both_BOUND"].items() if k != "n"}}.items():
        ci = f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]" if r["ci95"] else ""
        k = f"{r['kappa']:.3f}" if r["kappa"] is not None else "  n/a"
        pos = f"pos A/C {r['positives_author']}/{r['positives_coder']}" if "positives_author" in r else ""
        print(f"  {f:34} kappa {k} {ci:15} agree {r['pct_agree']:.1%}  {pos}")
    p = results["prevalence"]
    print(f"\nprevalence among own-disciplinary (author n={p['n_disciplinary_author']}, "
          f"coder n={p['n_disciplinary_coder']})")
    for k in MISCONDUCT:
        r = p[k]
        print(f"  {k:6} author {r['author']:.1%}  coder {r['coder']:.1%}  "
              f"diff {r['difference_coder_minus_author']:+.1%} [{r['difference_ci95'][0]:+.2f}, {r['difference_ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
