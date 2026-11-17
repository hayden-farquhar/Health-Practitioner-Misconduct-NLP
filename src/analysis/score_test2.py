"""Score the frozen classifiers once on the second test set (config/external_test_protocol_2026-11.md).

Reference: the author's blind single-coder codes for 160 never-labelled decisions, 40 per period
stratum. Metric definitions follow validation_analysis.classification_validation (misconduct
macro-F1 over the 7 categories plus N/A; finding labels = union of true and predicted, with
non-disciplinary as NONE; outcome micro-F1 over the 8 scored codes, OTHER excluded). The script
first reproduces the main test-set metrics with unit weights to show the definitions match.

Stops if any pinned model file, or the boundary keyword function, differs from the protocol.
Outputs: outputs/statistics/test2_results.json and outputs/tables/test2/.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.score_test2
"""

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.analysis import validation_analysis as va
from src.analysis.sensitivity_author_reference import author_test_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
T2 = PROJECT_ROOT / "data" / "annotations" / "test2_2026-11"
FROZEN = T2 / "P13_test2_results_SUBMITTED_2026-11-07.xlsx"
MANIFEST = T2 / "sample_manifest.json"
RAW_CORPUS = PROJECT_ROOT / "data" / "extracted" / "all_decisions.jsonl"
OUT_JSON = va.STATS_DIR / "test2_results.json"
OUT_TABLES = va.TABLES_DIR / "test2"

SEED = 20261106
N_BOOT = 10_000
PINNED = {
    "models/sklearn_misconduct/misconduct_clf.pkl": "6fa13a4c3b31",
    "models/sklearn_misconduct/disciplinary_clf.pkl": "65ee6fce82df",
    "models/sklearn_misconduct/finding_clf.pkl": "451e923f131e",
    "models/sklearn_misconduct/finding_le.pkl": "99066fa715b9",
    "models/sklearn_misconduct/vectorizer.pkl": "74ff06d53607",
    "models/sklearn_misconduct/vectorizer_finding.pkl": "74ff06d53607",
    "src/classification/outcome_extraction.py": "0af0e9de8df6",
}
BOUNDARY_FN_SHA = "69c2609ae87b"
MC = va.MISCONDUCT_CODES            # 7 categories + "N/A"
CATS = [c for c in MC if c != "N/A"]
OC = va.OUTCOME_CODES               # 8 scored outcome codes (OTHER excluded)
SUBTYPES = {"sexual": "BOUND_SEXUAL", "emotional": "BOUND_EMOTIONAL", "financial": "BOUND_FINANCIAL"}


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def boundary_keywords() -> dict:
    """Keyword regexes from statistical_analysis.boundary_analysis, after checking its hash."""
    src = (PROJECT_ROOT / "src" / "analysis" / "statistical_analysis.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "boundary_analysis")
    seg = ast.get_source_segment(src, fn)
    if hashlib.sha1(seg.encode()).hexdigest()[:12] != BOUNDARY_FN_SHA:
        raise SystemExit("boundary_analysis differs from the protocol pin")
    kws = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.endswith("_kw"):
                kws[name[:-3]] = ast.literal_eval(node.value)
    return kws


def check_pins():
    bad = {p: sha(PROJECT_ROOT / p) for p, h in PINNED.items() if sha(PROJECT_ROOT / p) != h}
    if bad:
        raise SystemExit(f"Pinned files changed: {bad}")


def load_author_test2() -> pd.DataFrame:
    manifest = json.loads(MANIFEST.read_text())
    stratum = {x["item"]: x["stratum"] for x in manifest["items"]}
    weight = {s: manifest["pool_size_by_stratum"][s] / n
              for s, n in manifest["sample_size_by_stratum"].items()}
    rows = list(load_workbook(FROZEN, read_only=True)["Coding"].iter_rows(values_only=True))
    head = rows[0]
    coded = [dict(zip(head, r)) for r in rows[1:]]
    want = {c["decision_id"] for c in coded}
    texts = {}
    with open(RAW_CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                texts[r["id"]] = (r["full_text"], r["date"])
    y = lambda d, k: str(d.get(k) or "").strip().upper() == "Y"
    out = []
    for d in coded:
        disc = d["is_disciplinary"] == "Y"
        text, date = texts[d["decision_id"]]
        out.append({
            "id": d["decision_id"], "item": d["item"], "full_text": text,
            "year": int(date[:4]), "stratum": stratum[d["item"]],
            "weight": weight[stratum[d["item"]]],
            "is_disciplinary": disc,
            "misconduct_L1": [k for k in CATS if y(d, k)] if disc else ["N/A"],
            "finding": None if d["finding"] == "NA" else d["finding"],
            "outcomes": [k for k in OC if y(d, k)] if disc else ["N/A"],
            **{f"author_{s}": y(d, col) for s, col in SUBTYPES.items()},
        })
    return pd.DataFrame(out)


def predict(df: pd.DataFrame, models) -> dict:
    vec, mc_clf, find_clf, find_le, disc_clf, vec_find = models
    X, Xf = vec.transform(df["full_text"]), vec_find.transform(df["full_text"])
    enriched = {}
    with open(va.CORPUS_PATH) as f:
        for line in f:
            r = json.loads(line)
            enriched[r["id"]] = r.get("predicted_outcomes") or []
    return {
        "disc": disc_clf.predict(X),
        "mc": mc_clf.predict(X),
        "find": find_le.inverse_transform(find_clf.predict(Xf)),
        "oc": va.binarise(pd.Series([enriched[i] for i in df["id"]]), OC),
    }


def truth(df: pd.DataFrame) -> dict:
    return {
        "disc": df["is_disciplinary"].astype(int).values,
        "mc": va.binarise(df["misconduct_L1"], MC),
        "find": df["finding"].fillna("NONE").values,
        "oc": va.binarise(df["outcomes"], OC),
    }


def metrics(t: dict, p: dict, idx: np.ndarray, w: np.ndarray) -> dict:
    """Metrics on rows idx with sample weights w, using the main analysis definitions."""
    tm, pm = t["mc"][idx], p["mc"][idx]
    labels = sorted(set(t["find"][idx]) | set(p["find"][idx]))
    pr, rc, f1, _ = precision_recall_fscore_support(tm, pm, sample_weight=w, zero_division=0)
    return {
        "is_disciplinary_accuracy": accuracy_score(t["disc"][idx], p["disc"][idx], sample_weight=w),
        "misconduct_macro_f1": f1_score(tm, pm, average="macro", sample_weight=w, zero_division=0),
        "misconduct_micro_f1": f1_score(tm, pm, average="micro", sample_weight=w, zero_division=0),
        "finding_accuracy": accuracy_score(t["find"][idx], p["find"][idx], sample_weight=w),
        "outcome_micro_f1": f1_score(t["oc"][idx], p["oc"][idx], average="micro",
                                     sample_weight=w, zero_division=0),
        "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i],
                          "positives": int(tm[:, i].sum())} for i, c in enumerate(MC)},
    }


def flat(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k != "per_class"}
    for c, d in m["per_class"].items():
        for k in ("precision", "recall", "f1"):
            out[f"{c}_{k}"] = d[k]
    return out


def bootstrap(t, p, groups: list[np.ndarray], weights: np.ndarray, rng) -> dict:
    """Percentile 95% CIs, resampling within each group of row indices."""
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        draws.append(flat(metrics(t, p, idx, weights[idx])))
    d = pd.DataFrame(draws)
    return {k: [round(float(d[k].quantile(0.025)), 4), round(float(d[k].quantile(0.975)), 4)]
            for k in d.columns}


def with_ci(point: dict, ci: dict) -> dict:
    return {k: {"value": round(float(v), 4), "ci95": ci.get(k)} for k, v in flat(point).items()}


def reproduce_main(models):
    test = va.load_split("test")
    t, p = truth(test), predict(test, models)
    m = metrics(t, p, np.arange(len(test)), np.ones(len(test)))
    ref = json.loads((va.STATS_DIR / "validation_results.json").read_text())["classification"]
    pairs = [(m["is_disciplinary_accuracy"], ref["is_disciplinary"]["accuracy"]),
             (m["misconduct_macro_f1"], ref["misconduct"]["macro_f1"]),
             (m["misconduct_micro_f1"], ref["misconduct"]["micro_f1"]),
             (m["finding_accuracy"], ref["finding"]["accuracy"]),
             (m["outcome_micro_f1"], ref["outcome_extraction"]["micro_f1"])]
    if any(abs(round(a, 4) - b) > 1e-4 for a, b in pairs):
        raise SystemExit(f"Metric definitions do not reproduce the main test results: {pairs}")
    return test


def logit_or_per_decade(y: np.ndarray, year: np.ndarray) -> dict:
    n = len(y)
    if n < 10 or y.min() == y.max():
        return {"n": n, "events": int(y.sum()), "or_per_decade": None, "note": "descriptive only"}
    x = sm.add_constant((year - year.mean()) / 10)
    try:
        fit = sm.Logit(y, x).fit(disp=0)
    except Exception as e:  # perfect separation and similar
        return {"n": n, "events": int(y.sum()), "or_per_decade": None, "note": f"fit failed: {e}"}
    lo, hi = fit.conf_int()[1]
    return {"n": n, "events": int(y.sum()), "or_per_decade": round(float(np.exp(fit.params[1])), 3),
            "ci95": [round(float(np.exp(lo)), 3), round(float(np.exp(hi)), 3)],
            "p_value": float(f"{fit.pvalues[1]:.4g}")}


def main():
    check_pins()
    kws = boundary_keywords()
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    models = va.load_models()
    rng = np.random.default_rng(SEED)
    main_test = reproduce_main(models)

    df = load_author_test2()
    t, p = truth(df), predict(df, models)
    w = df["weight"].values
    strata = sorted(df["stratum"].unique())
    groups = [np.where(df["stratum"].values == s)[0] for s in strata]
    all_idx = np.arange(len(df))
    results = {"protocol": "config/external_test_protocol_2026-11.md",
               "frozen_workbook_sha1": hashlib.sha1(FROZEN.read_bytes()).hexdigest(),
               "n": len(df), "n_disciplinary_author": int(t["disc"].sum()),
               "main_test_metrics_reproduced": True}

    # Primary: weighted to the pool
    results["primary_weighted"] = with_ci(metrics(t, p, all_idx, w), bootstrap(t, p, groups, w, rng))

    # Secondary 1: by period (unweighted)
    results["by_period"] = {}
    for s, g in zip(strata, groups):
        ones = np.ones(len(df))
        results["by_period"][s] = with_ci(metrics(t, p, g, ones[g]), bootstrap(t, p, [g], ones, rng))

    # Secondary 2: combined author-reference set (first test set's 45 author codes + these 160)
    first = author_test_df(main_test)
    t1, p1 = truth(first), predict(first, models)
    tc = {k: np.concatenate([t1[k], t[k]]) for k in t}
    pc = {k: np.concatenate([p1[k], p[k]]) for k in p}
    n_c = len(first) + len(df)
    ones_c = np.ones(n_c)
    results["combined_author_reference"] = {
        "n": n_c, **with_ci(metrics(tc, pc, np.arange(n_c), ones_c),
                            bootstrap(tc, pc, [np.arange(n_c)], ones_c, rng))}

    # Secondary 3: recall and false-positive rate against decision year
    years = df["year"].values.astype(float)
    results["temporal_stability"] = {}
    for i, c in enumerate(CATS):
        j = MC.index(c)
        pos, neg = t["mc"][:, j] == 1, t["mc"][:, j] == 0
        results["temporal_stability"][c] = {
            "detection_among_positives": logit_or_per_decade(p["mc"][pos, j], years[pos]),
            "false_positive_among_negatives": logit_or_per_decade(p["mc"][neg, j], years[neg]),
        }

    # Secondary 4: predicted vs author-coded prevalence by period
    order = np.arange(len(strata), dtype=float)
    trend_rows, results["trend_check"] = [], {}
    for c in CATS:
        j = MC.index(c)
        pred_prev, coded_prev = [], []
        for g in groups:
            pd_mask, ad_mask = p["disc"][g] == 1, t["disc"][g] == 1
            pred_prev.append(float(p["mc"][g][pd_mask, j].mean()) if pd_mask.any() else np.nan)
            coded_prev.append(float(t["mc"][g][ad_mask, j].mean()) if ad_mask.any() else np.nan)
        slope = lambda v: float(np.polyfit(order, v, 1)[0])
        ts = results["temporal_stability"][c]
        stable = all(
            (x.get("ci95") is None) or (x["ci95"][0] <= 1 <= x["ci95"][1])
            for x in (ts["detection_among_positives"], ts["false_positive_among_negatives"]))
        same_dir = np.sign(slope(pred_prev)) == np.sign(slope(coded_prev))
        results["trend_check"][c] = {
            "predicted_prevalence_by_period": dict(zip(strata, [round(v, 3) for v in pred_prev])),
            "author_prevalence_by_period": dict(zip(strata, [round(v, 3) for v in coded_prev])),
            "slope_predicted": round(slope(pred_prev), 4), "slope_author": round(slope(coded_prev), 4),
            "same_direction": bool(same_dir), "no_clear_time_dependence": bool(stable),
            "robust": bool(same_dir and stable),
        }
        trend_rows.append({"Class": c, **{f"Pred {s}": v for s, v in zip(strata, pred_prev)},
                           **{f"Author {s}": v for s, v in zip(strata, coded_prev)},
                           "Same direction": bool(same_dir), "No time dependence": bool(stable)})
    pd.DataFrame(trend_rows).round(3).to_csv(OUT_TABLES / "trend_check.csv", index=False)

    # Secondary 5: boundary keyword subtypes against the author's subtype codes
    bound = df[t["mc"][:, MC.index("BOUND")] == 1]
    results["boundary_subtypes"] = {"n_author_bound": len(bound)}
    for s in SUBTYPES:
        flag = bound["full_text"].str.contains(kws[s], case=False, na=False).values
        gold = bound[f"author_{s}"].values
        tp, fp, fn = int((flag & gold).sum()), int((flag & ~gold).sum()), int((~flag & gold).sum())
        results["boundary_subtypes"][s] = {
            "author_positive": int(gold.sum()), "keyword_positive": int(flag.sum()),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}

    # Error-adjusted prevalence (Rogan-Gladen) with weighted sensitivity/specificity
    summary = json.loads((va.STATS_DIR / "summary_statistics.json").read_text())
    n_disc, observed = summary["disciplinary_decisions"], summary["misconduct_prevalence"]
    rg = []
    for c in CATS:
        j = MC.index(c)
        tt, pp = t["mc"][:, j] == 1, p["mc"][:, j] == 1
        sens = float(w[tt & pp].sum() / w[tt].sum()) if tt.any() else np.nan
        spec = float(w[~tt & ~pp].sum() / w[~tt].sum())
        apparent = observed.get(c, 0) / n_disc
        denom = sens + spec - 1
        adj = min(1, max(0, (apparent + spec - 1) / denom)) if abs(denom) > 0.01 else apparent
        rg.append({"code": c, "observed_pct": round(apparent * 100, 1), "sensitivity": round(sens, 3),
                   "specificity": round(spec, 3), "adjusted_pct": round(adj * 100, 1)})
    results["adjusted_prevalence_weighted"] = rg
    pd.DataFrame(rg).to_csv(OUT_TABLES / "adjusted_prevalence_test2.csv", index=False)

    summary_rows = []
    for name, block in [("Weighted (n=160)", results["primary_weighted"]),
                        *[(f"Period {s}", results["by_period"][s]) for s in strata],
                        ("Combined author reference (n=205)", results["combined_author_reference"])]:
        row = {"Set": name}
        for k in ("is_disciplinary_accuracy", "misconduct_macro_f1", "misconduct_micro_f1",
                  "finding_accuracy", "outcome_micro_f1", *[f"{c}_f1" for c in CATS]):
            v = block[k]
            row[k] = f"{v['value']:.3f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_TABLES / "test2_performance.csv", index=False)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(pd.DataFrame(summary_rows).T.to_string())
    print(json.dumps({"trend_check": {c: {k: v[k] for k in ("same_direction", "no_clear_time_dependence")}
                                      for c, v in results["trend_check"].items()},
                      "boundary_subtypes": results["boundary_subtypes"]}, indent=1))


if __name__ == "__main__":
    main()
