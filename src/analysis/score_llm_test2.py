"""Score the logged language-model coder once on the second test set (config/llm_coder_protocol_2026-11.md).

Uses the frozen author codes, weights, metric definitions and bootstrap of score_test2.py. A
refused, errored or unparseable item counts against the coder (scored as non-disciplinary with
no codes); a sensitivity analysis drops those items. Adds the paired bootstrap difference from
the TF-IDF classifier and applies the protocol's pass rule.

Outputs: outputs/statistics/llm_test2_results.json and outputs/tables/test2/llm_*.csv.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.score_llm_test2
"""

import hashlib
import json

import numpy as np
import pandas as pd

from src.analysis import score_test2 as s
from src.analysis import validation_analysis as va

CODES = s.T2 / "model_pass" / "model_codes.json"
SUMMARY = s.T2 / "model_pass" / "run_summary.json"
OUT_JSON = va.STATS_DIR / "llm_test2_results.json"
FROZEN_SHA1 = "b42fd27e8e3756ee1beae2a0f0033c8ab0210a6a"
PROMPT_SHA256 = "9f05c736ccd257afe8bcf4665f1416d357baa25b5370ac7741ced2a6ba322866"


def coder_predictions(df: pd.DataFrame) -> tuple[dict, list[str], list[dict]]:
    rows = {r["item"]: r for r in json.loads(CODES.read_text())}
    missing = [i for i in df["item"] if i not in rows]
    disc, mc, find, oc, subtypes = [], [], [], [], []
    for item in df["item"]:
        r = rows.get(item)
        d = r is not None and r["is_disciplinary"] == "Y"
        disc.append(int(d))
        mc.append([k for k in s.CATS if r[k] == "Y"] if d else ["N/A"])
        find.append(r["finding"] if d and r["finding"] not in ("NA", None) else "NONE")
        oc.append([k for k in s.OC if r[k] == "Y"] if d else [])
        subtypes.append({st: bool(d and r[col] == "Y") for st, col in s.SUBTYPES.items()})
    p = {"disc": np.array(disc), "mc": va.binarise(pd.Series(mc), s.MC),
         "find": np.array(find, dtype=object), "oc": va.binarise(pd.Series(oc), s.OC)}
    return p, missing, subtypes


def paired_difference(t, p_a, p_b, groups, w, rng) -> dict:
    """Bootstrap of macro-F1(a) - macro-F1(b) on identical within-stratum resamples."""
    diffs = []
    for _ in range(s.N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        a = s.metrics(t, p_a, idx, w[idx])["misconduct_macro_f1"]
        b = s.metrics(t, p_b, idx, w[idx])["misconduct_macro_f1"]
        diffs.append(a - b)
    full = np.arange(len(w))
    point = (s.metrics(t, p_a, full, w)["misconduct_macro_f1"]
             - s.metrics(t, p_b, full, w)["misconduct_macro_f1"])
    return {"difference": round(float(point), 4),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 4),
                     round(float(np.quantile(diffs, 0.975)), 4)]}


def main():
    if hashlib.sha1(s.FROZEN.read_bytes()).hexdigest() != FROZEN_SHA1:
        raise SystemExit("Frozen author workbook changed")
    manifest = json.loads((s.T2 / "model_pass" / "batch_manifest.json").read_text())
    if manifest["system_prompt_sha256"] != PROMPT_SHA256:
        raise SystemExit("Coder prompt differs from the protocol")

    rng = np.random.default_rng(s.SEED)
    df = s.load_author_test2()
    t = s.truth(df)
    p, missing, subtypes = coder_predictions(df)
    p_tfidf = s.predict(df, va.load_models())
    w = df["weight"].values
    strata = sorted(df["stratum"].unique())
    groups = [np.where(df["stratum"].values == st)[0] for st in strata]
    full = np.arange(len(df))

    results = {"protocol": "config/llm_coder_protocol_2026-11.md",
               "batch_id": manifest["batch_id"], "n": len(df),
               "failed_items_scored_against_coder": missing,
               "run_summary": json.loads(SUMMARY.read_text())}
    results["primary_weighted"] = s.with_ci(s.metrics(t, p, full, w),
                                            s.bootstrap(t, p, groups, w, rng))
    results["paired_difference_vs_tfidf_macro_f1"] = paired_difference(t, p, p_tfidf, groups, w, rng)

    if missing:
        keep = np.array([i not in missing for i in df["item"]])
        kept_groups = [g[keep[g]] for g in groups]
        results["sensitivity_failures_excluded"] = s.with_ci(
            s.metrics(t, p, np.where(keep)[0], w[keep]), s.bootstrap(t, p, kept_groups, w, rng))

    pw = results["primary_weighted"]
    diff = results["paired_difference_vs_tfidf_macro_f1"]
    results["pass_rule"] = {
        "macro_f1_ge_0.70": pw["misconduct_macro_f1"]["value"] >= 0.70,
        "is_disciplinary_accuracy_ge_0.90": pw["is_disciplinary_accuracy"]["value"] >= 0.90,
        "difference_ci_excludes_0": diff["ci95"][0] > 0 or diff["ci95"][1] < 0,
    }
    results["pass_rule"]["passes"] = all(results["pass_rule"].values())
    results["unreliable_categories_f1_lt_0.60"] = [
        c for c in s.CATS if pw[f"{c}_f1"]["value"] < 0.60]

    ones = np.ones(len(df))
    results["by_period"] = {st: s.with_ci(s.metrics(t, p, g, ones[g]), s.bootstrap(t, p, [g], ones, rng))
                            for st, g in zip(strata, groups)}

    years = df["year"].values.astype(float)
    order = np.arange(len(strata), dtype=float)
    slope = lambda v: float(np.polyfit(order, v, 1)[0])
    results["temporal_stability"], results["trend_check"] = {}, {}
    for c in s.CATS:
        j = s.MC.index(c)
        pos, neg = t["mc"][:, j] == 1, t["mc"][:, j] == 0
        ts = {"detection_among_positives": s.logit_or_per_decade(p["mc"][pos, j], years[pos]),
              "false_positive_among_negatives": s.logit_or_per_decade(p["mc"][neg, j], years[neg])}
        results["temporal_stability"][c] = ts
        pred_prev, coded_prev = [], []
        for g in groups:
            pd_mask, ad_mask = p["disc"][g] == 1, t["disc"][g] == 1
            pred_prev.append(float(p["mc"][g][pd_mask, j].mean()) if pd_mask.any() else np.nan)
            coded_prev.append(float(t["mc"][g][ad_mask, j].mean()) if ad_mask.any() else np.nan)
        stable = all(x.get("ci95") is None or x["ci95"][0] <= 1 <= x["ci95"][1] for x in ts.values())
        same_dir = bool(np.sign(slope(pred_prev)) == np.sign(slope(coded_prev)))
        results["trend_check"][c] = {
            "coder_prevalence_by_period": dict(zip(strata, [round(v, 3) for v in pred_prev])),
            "author_prevalence_by_period": dict(zip(strata, [round(v, 3) for v in coded_prev])),
            "same_direction": same_dir, "no_clear_time_dependence": bool(stable),
            "author_positives": int(pos.sum())}

    bound = np.where(t["mc"][:, s.MC.index("BOUND")] == 1)[0]
    results["boundary_subtypes"] = {"n_author_bound": int(len(bound))}
    for st in s.SUBTYPES:
        flag = np.array([subtypes[i][st] for i in bound])
        gold = df[f"author_{st}"].values[bound]
        tp, fp, fn = int((flag & gold).sum()), int((flag & ~gold).sum()), int((~flag & gold).sum())
        results["boundary_subtypes"][st] = {
            "author_positive": int(gold.sum()), "coder_positive": int(flag.sum()),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}

    rows = []
    for name, block in [("Coder, weighted (n=160)", pw),
                        *[(f"Coder, period {st}", results["by_period"][st]) for st in strata]]:
        row = {"Set": name}
        for k in ("is_disciplinary_accuracy", "misconduct_macro_f1", "misconduct_micro_f1",
                  "finding_accuracy", "outcome_micro_f1", *[f"{c}_f1" for c in s.CATS]):
            v = block[k]
            row[k] = f"{v['value']:.3f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"
        rows.append(row)
    pd.DataFrame(rows).to_csv(s.OUT_TABLES / "llm_test2_performance.csv", index=False)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(pd.DataFrame(rows).T.to_string())
    print(json.dumps({k: results[k] for k in ("failed_items_scored_against_coder",
                                              "paired_difference_vs_tfidf_macro_f1", "pass_rule",
                                              "unreliable_categories_f1_lt_0.60",
                                              "boundary_subtypes")}, indent=1))


if __name__ == "__main__":
    main()
