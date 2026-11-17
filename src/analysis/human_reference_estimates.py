"""Human-coded prevalence from the re-coded second test set (fallback, config/llm_coder_protocol_2026-11.md).

Implements the 2026-11-07 amendment: weighted ratio estimates (stratum weight = pool size / 40)
over the 160 author-coded test2 decisions, with 10,000 within-stratum bootstrap resamples
(seed 20261106); unweighted prevalence by period; and a logistic regression of each misconduct
type on decision year among author-coded disciplinary decisions. The estimates describe the
sampled pool (dated decisions of at least 500 characters never previously labelled). The script
also sets the classifier's corpus prevalence beside them for comparison.

Outputs: outputs/statistics/human_reference_estimates.json,
outputs/tables/table35_human_reference_prevalence.csv, table36_human_reference_by_period.csv,
table38_coded_sample_estimates.csv and
outputs/figures/fig20_human_vs_classifier_prevalence.png.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.human_reference_estimates
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' styles)
from openpyxl import load_workbook
from statsmodels.stats.multitest import multipletests

from src.analysis import score_test2 as s
from src.analysis import validation_analysis as va

try:
    import _integrity_guard  # noqa: F401  (optional author-side check on figure text; not needed to reproduce)
except ImportError:
    pass

OUT_JSON = va.STATS_DIR / "human_reference_estimates.json"
CLASSIFIER_SUMMARY = va.STATS_DIR / "summary_statistics.json"
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]
FINDINGS = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING"]
LABELS = {"PROF": "Professional conduct", "FRAUD": "Dishonesty/fraud",
          "CLIN": "Clinical competence", "BOUND": "Boundary violations",
          "PRESC": "Prescribing/dispensing", "IMPR": "Impairment", "COMM": "Communication"}
OKABE_ITO = {"blue": "#0072B2", "vermil": "#D55E00", "grey": "#999999"}


def load() -> pd.DataFrame:
    base = s.load_author_test2()
    rows = list(load_workbook(s.FROZEN, read_only=True)["Coding"].iter_rows(values_only=True))
    coded = {r[1]: dict(zip(rows[0], r)) for r in rows[1:]}
    y = lambda d, k: str(d.get(k) or "").strip().upper() == "Y"
    for k in OUTCOMES + ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]:
        base[k] = [y(coded[i], k) for i in base["item"]]
    for c in s.CATS:
        base[c] = [c in m for m in base["misconduct_L1"]]
    prof = {"medical_practitioner": "medical", "nurse": "nurse_midwife", "midwife": "nurse_midwife"}
    base["profession_group"] = [prof.get(coded[i]["profession"], "other") for i in base["item"]]
    base["finding_code"] = [coded[i]["finding"] for i in base["item"]]
    return base


def weighted(x, w, groups, rng) -> dict:
    est = float((x * w).sum() / w.sum())
    boot = []
    for _ in range(s.N_BOOT):
        i = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boot.append((x[i] * w[i]).sum() / w[i].sum())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"pct": round(100 * est, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
            "n_positive": int(x.sum()), "n": int(len(x))}


def plain(x, rng) -> dict:
    boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(s.N_BOOT)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"pct": round(100 * float(x.mean()), 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
            "n_positive": int(x.sum()), "n": int(len(x))}


def subset_estimates(d: pd.DataFrame, cols, rng) -> dict:
    w = d["weight"].values
    groups = [np.where(d["stratum"].values == st)[0] for st in sorted(d["stratum"].unique())]
    return {c: weighted(np.asarray(d[c], dtype=float), w, groups, rng) for c in cols}


FINDING_LABELS = {"finding_PM": "Professional misconduct", "finding_UPC_SAT": "Unsatisfactory professional conduct",
                  "finding_UPC_GEN": "Unprofessional conduct", "finding_NO_FINDING": "No finding"}
OUTCOME_LABELS = {"CANCEL": "Cancellation", "SUSPEND": "Suspension", "CONDITIONS": "Conditions",
                  "REPRIMAND": "Reprimand", "FINE": "Fine", "CAUTION": "Caution", "UNDERTAKING": "Undertaking",
                  "NO_ADVERSE": "No adverse finding", "OTHER": "Other orders (costs, disqualification, prohibition, other)"}
PROFESSION_LABELS = {"profession_medical": "Medical practitioner", "profession_nurse_midwife": "Nurse or midwife",
                     "profession_other": "Other profession"}


def ci_str(v: dict) -> str:
    return f"{v['pct']:.1f} ({v['ci95'][0]:.1f}-{v['ci95'][1]:.1f})"


def pool_exclusion_check() -> dict:
    """Classifier prevalence among predicted-disciplinary decisions: sampling pool vs the 600 excluded."""
    spec = importlib.util.spec_from_file_location(
        "build_test2_packet", s.PROJECT_ROOT / "src" / "preprocessing" / "build_test2_packet.py")
    sys.path.insert(0, str(s.PROJECT_ROOT / "src" / "preprocessing"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    excluded = mod.labelled_ids()
    groups = {"pool": [], "excluded_labelled": []}
    with open(va.CORPUS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("predicted_is_disciplinary"):
                continue
            if r["id"] in excluded:
                groups["excluded_labelled"].append(r)
            elif len(r.get("full_text") or "") >= 500 and r.get("date"):
                groups["pool"].append(r)
    return {g: {"n_predicted_disciplinary": len(rs),
                **{c: round(100 * sum(c in (r.get("predicted_misconduct_L1") or []) for r in rs) / len(rs), 1)
                   for c in s.CATS}} for g, rs in groups.items()}


def related_decisions(df: pd.DataFrame) -> np.ndarray:
    """Approximate number of corpus decisions about the same practitioner, per coded decision.

    The practitioner's surname is the coder's text_check entry. A corpus decision in the same
    jurisdiction counts as related if the surname appears as a word in its citation or, for
    neutral citations without party names, in its first 400 characters. Anonymised or unnamed
    practitioners count as 1. This is a rough matter proxy used only for a sensitivity analysis.
    """
    rows = list(load_workbook(s.FROZEN, read_only=True)["Coding"].iter_rows(values_only=True))
    coded = {r[1]: dict(zip(rows[0], r)) for r in rows[1:]}
    corpus = [json.loads(line) for line in open(s.RAW_CORPUS)]
    m = []
    for item in df["item"]:
        d = coded[item]
        name = str(d["text_check"]).strip()
        if name.lower() == "unnamed" or len(name) < 3:
            m.append(1)
            continue
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        n = 0
        for r in corpus:
            if r.get("jurisdiction") != d["jurisdiction"]:
                continue
            cit = r.get("citation") or ""
            has_parties = bool(re.search(r"[A-Za-z]{3,}", re.sub(r"\[\d{4}\]|[A-Z]{2,}\b", "", cit)))
            if pat.search(cit) or (not has_parties and pat.search((r.get("full_text") or "")[:400])):
                n += 1
        m.append(max(n, 1))
    return np.array(m)


def practitioner_level(df: pd.DataFrame, disc: pd.DataFrame, rng) -> dict:
    """Re-weight each coded decision by 1 / (related decisions) so each practitioner counts about once."""
    m_all = dict(zip(df["item"], related_decisions(df)))
    m = np.array([m_all[i] for i in disc["item"]], dtype=float)
    w = disc["weight"].values / m
    groups = [np.where(disc["stratum"].values == st)[0] for st in sorted(disc["stratum"].unique())]
    out = {"n_disciplinary_with_related_decisions": int((m > 1).sum()),
           "median_related_decisions": float(np.median(m)), "max_related_decisions": int(m.max())}
    for c in s.CATS:
        out[c] = weighted(np.asarray(disc[c], dtype=float), w, groups, rng)
    for k in ("CANCEL", "REPRIMAND"):
        out[k] = weighted(np.asarray(disc[k], dtype=float), w, groups, rng)
    return out


def main():
    rng = np.random.default_rng(s.SEED)
    df = load()
    disc = df[df["is_disciplinary"]].reset_index(drop=True)
    bound = disc[disc["BOUND"]].reset_index(drop=True)
    for f in FINDINGS:
        disc[f"finding_{f}"] = disc["finding_code"] == f
    for p in ["medical", "nurse_midwife", "other"]:
        disc[f"profession_{p}"] = disc["profession_group"] == p
    df["disciplinary"] = df["is_disciplinary"]

    results = {
        "protocol": "config/llm_coder_protocol_2026-11.md (2026-11-07 amendment)",
        "frozen_workbook_sha1": s.hashlib.sha1(s.FROZEN.read_bytes()).hexdigest(),
        "describes": "sampled pool: dated decisions >= 500 characters never previously labelled",
        "n_decisions": len(df), "n_disciplinary": len(disc), "n_bound": len(bound),
        "share_disciplinary": subset_estimates(df, ["disciplinary"], rng)["disciplinary"],
        "misconduct": subset_estimates(disc, s.CATS, rng),
        "boundary_subtypes_among_bound": subset_estimates(
            bound, ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"], rng),
        "outcomes": subset_estimates(disc, OUTCOMES, rng),
        "finding": subset_estimates(disc, [f"finding_{f}" for f in FINDINGS], rng),
        "profession": subset_estimates(
            disc, [f"profession_{p}" for p in ["medical", "nurse_midwife", "other"]], rng),
    }

    strata = sorted(disc["stratum"].unique())
    by_period, period_rows = {}, []
    years = disc["year"].values.astype(float)
    for c in s.CATS:
        x = np.asarray(disc[c], dtype=int)
        per = {st: plain(x[disc["stratum"].values == st].astype(float), rng) for st in strata}
        by_period[c] = {"by_period": per, "logit_year": s.logit_or_per_decade(x, years)}
        lr = by_period[c]["logit_year"]
        period_rows.append({"Misconduct type": LABELS[c],
                            **{st: f"{per[st]['pct']:.1f} ({per[st]['n_positive']}/{per[st]['n']})"
                               for st in strata},
                            "OR per decade (95% CI)": (
                                f"{lr['or_per_decade']:.2f} ({lr['ci95'][0]:.2f}-{lr['ci95'][1]:.2f})"
                                if lr.get("or_per_decade") is not None else "descriptive only")})
    results["misconduct_by_period"] = by_period
    tested = [c for c in s.CATS if by_period[c]["logit_year"].get("p_value") is not None]
    q = multipletests([by_period[c]["logit_year"]["p_value"] for c in tested], method="fdr_bh")[1]
    for c, qv in zip(tested, q):
        by_period[c]["logit_year"]["q_bh_across_types"] = round(float(qv), 4)
    for row, c in zip(period_rows, s.CATS):
        row["BH q"] = (f"{by_period[c]['logit_year']['q_bh_across_types']:.2f}"
                       if c in tested else "")

    clf = json.loads(CLASSIFIER_SUMMARY.read_text())
    clf_pct = {c: round(100 * clf["misconduct_prevalence"][c] / clf["disciplinary_decisions"], 1)
               for c in s.CATS}
    results["classifier_corpus_pct"] = clf_pct
    results["pool_exclusion_check"] = pool_exclusion_check()
    results["practitioner_level_sensitivity"] = practitioner_level(df, disc, rng)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))

    order = sorted(s.CATS, key=lambda c: -results["misconduct"][c]["pct"])
    table = [{"Misconduct type": LABELS[c],
              "Human-coded % (95% CI)": "{pct:.1f} ({ci95[0]:.1f}-{ci95[1]:.1f})".format(
                  **results["misconduct"][c]),
              "Author positives": results["misconduct"][c]["n_positive"],
              "Classifier corpus %": clf_pct.get(c)} for c in order]
    pd.DataFrame(table).to_csv(Path(os.path.relpath(va.TABLES_DIR)) / "table35_human_reference_prevalence.csv", index=False)
    pd.DataFrame(period_rows).to_csv(Path(os.path.relpath(va.TABLES_DIR)) / "table36_human_reference_by_period.csv",
                                     index=False)
    full = [{"Group": "All decisions", "Measure": "Disciplinary", "Weighted % (95% CI)": ci_str(results["share_disciplinary"]),
             "n": f"{results['share_disciplinary']['n_positive']}/{results['share_disciplinary']['n']}"}]
    for group, key, names in [("Misconduct type", "misconduct", LABELS), ("Finding", "finding", FINDING_LABELS),
                              ("Outcome", "outcomes", OUTCOME_LABELS), ("Profession", "profession", PROFESSION_LABELS)]:
        for k, v in results[key].items():
            full.append({"Group": group, "Measure": names[k], "Weighted % (95% CI)": ci_str(v),
                         "n": f"{v['n_positive']}/{v['n']}"})
    pd.DataFrame(full).to_csv(Path(os.path.relpath(va.TABLES_DIR)) / "table38_coded_sample_estimates.csv", index=False)

    with plt.style.context(["science", "nature", "no-latex"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ypos = np.arange(len(order))[::-1]
        est = [results["misconduct"][c] for c in order]
        ax.errorbar([e["pct"] for e in est], ypos,
                    xerr=[[e["pct"] - e["ci95"][0] for e in est], [e["ci95"][1] - e["pct"] for e in est]],
                    fmt="o", color=OKABE_ITO["blue"], ms=3, capsize=2,
                    label=f"Coded sample (weighted, 95% CI; n = {results['n_disciplinary']})")
        ax.plot([clf_pct[c] for c in order], ypos, "D", color=OKABE_ITO["vermil"], ms=3,
                label=f"Classifier, whole corpus (n = {clf['disciplinary_decisions']:,})")
        ax.set_yticks(ypos, [LABELS[c] for c in order])
        ax.set_xlabel("Disciplinary decisions with misconduct type (%)")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=6.5, frameon=False)
        out = Path(os.path.relpath(va.FIGURES_DIR)) / "fig20_human_vs_classifier_prevalence"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)

    print(pd.DataFrame(table).to_string(index=False))
    print(pd.DataFrame(period_rows).to_string(index=False))
    print(json.dumps({k: results[k] for k in ("share_disciplinary", "boundary_subtypes_among_bound",
                                              "outcomes", "finding", "profession")}, indent=1))


if __name__ == "__main__":
    main()
