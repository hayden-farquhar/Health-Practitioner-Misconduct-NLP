"""Disciplinary decisions per 10,000 registered practitioners, 2011-2025 (Part C item 4).

Turns annual counts of classified disciplinary decisions into rates, so that growth in
decisions can be compared with growth in the registered workforce.

Choices (fixed 2026-11-06, before any rate was computed):
- Numerator: decisions the classifier labels disciplinary (enriched corpus), by calendar year of
  the decision and by the profession field. The corpus "midwife" value is counted with "nurse",
  matching the Nursing and Midwifery Board register.
- Denominator: Ahpra registrants at 30 June of the same calendar year
  (data/reference/ahpra_registrants_2010_2025.csv; first count is 30 June 2011). Primary counts
  exclude the 2019/20-2021/22 COVID-19 pandemic sub-register (temporary registrants); the
  headline figures including it are a sensitivity analysis.
- Years: 2011-2025. 2010 has no 30 June count and 2026 is a partial year.
- Professions reported separately: medical, nurse/midwife, pharmacist, psychologist, dental,
  chiropractor. The all-professions rate uses the ten professions regulated throughout
  (registered since 1 July 2010), so professions joining in 2012 and 2018 do not shift it.
- Trend: quasi-Poisson GLM of annual counts on year with log(registrants) offset; rate ratio per
  year with 95% CI.
- Sensitivity: decisions with unknown profession allocated to professions in proportion to the
  known-profession decisions of the same year (main rates exclude them, so are lower bounds).

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.registrant_rates
"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' styles)
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "extracted" / "enriched_corpus.jsonl"
REGISTRANTS = PROJECT_ROOT / "data" / "reference" / "ahpra_registrants_2010_2025.csv"
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_JSON = PROJECT_ROOT / "outputs" / "statistics" / "registrant_rates.json"

YEARS = range(2011, 2026)
REPORTED = {
    "medical_practitioner": "Medical",
    "nurse": "Nursing and midwifery",
    "pharmacist": "Pharmacy",
    "psychologist": "Psychology",
    "dentist": "Dental",
    "chiropractor": "Chiropractic",
}
REGULATED_SINCE_2010 = ["medical_practitioner", "nurse", "pharmacist", "psychologist", "dentist",
                        "chiropractor", "physiotherapist", "osteopath", "optometrist", "podiatrist"]
PER = 10_000
COLOURS = {"medical_practitioner": "#0072B2", "nurse": "#E69F00", "pharmacist": "#009E73",
           "psychologist": "#CC79A7", "dentist": "#56B4E9", "chiropractor": "#D55E00",
           "ALL_REGULATED_SINCE_2010": "#000000"}  # Okabe-Ito


def load_registrants(include_subregister: bool) -> pd.DataFrame:
    reg = pd.read_csv(REGISTRANTS)
    reg = reg[reg["registrants"].notna() & reg["profession"].isin(REGULATED_SINCE_2010)].copy()
    reg["year"] = reg["as_at_date"].str[:4].astype(int)
    if not include_subregister:
        excl = reg["notes"].str.extract(r"excl_pandemic_subregister=(\d+)")[0]
        reg.loc[excl.notna(), "registrants"] = excl[excl.notna()].astype(float)
    reg["registrants"] = reg["registrants"].astype(float)
    return reg.pivot(index="year", columns="profession", values="registrants").loc[list(YEARS)]


def load_counts() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("predicted_is_disciplinary") or not r.get("date"):
                continue
            year = int(str(r["date"])[:4])
            if year in YEARS:
                prof = r.get("profession")
                rows.append({"year": year, "profession": "nurse" if prof == "midwife" else prof})
    df = pd.DataFrame(rows)
    unknown = df[df["profession"].isna()].groupby("year").size().reindex(list(YEARS), fill_value=0)
    counts = (df[df["profession"].isin(REGULATED_SINCE_2010)]
              .groupby(["year", "profession"]).size().unstack(fill_value=0)
              .reindex(index=list(YEARS), columns=REGULATED_SINCE_2010, fill_value=0))
    return counts, unknown


def allocate_unknown(counts: pd.DataFrame, unknown: pd.Series) -> pd.DataFrame:
    share = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts + share.mul(unknown, axis=0)


def trend(count: pd.Series, exposure: pd.Series) -> dict:
    years = np.array(list(YEARS), dtype=float)
    X = sm.add_constant(years - years.mean())
    fit = sm.GLM(count.values, X, family=sm.families.Poisson(),
                 offset=np.log(exposure.values)).fit(scale="X2")
    lo, hi = fit.conf_int()[1]
    return {"rate_ratio_per_year": round(float(np.exp(fit.params[1])), 4),
            "ci95": [round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)],
            "p_value": float(f"{fit.pvalues[1]:.4g}"),
            "dispersion": round(float(fit.scale), 2)}


def summarise(counts: pd.DataFrame, reg: pd.DataFrame) -> dict:
    groups = {k: [k] for k in REPORTED} | {"ALL_REGULATED_SINCE_2010": REGULATED_SINCE_2010}
    out = {}
    for key, profs in groups.items():
        c, e = counts[profs].sum(axis=1), reg[profs].sum(axis=1)
        rate = c / e * PER
        out[key] = {
            "decisions": int(round(c.sum())),
            "rate_per_10000_by_year": {int(y): round(float(v), 3) for y, v in rate.items()},
            "rate_2011_2013": round(float(c.loc[2011:2013].sum() / e.loc[2011:2013].sum() * PER), 3),
            "rate_2023_2025": round(float(c.loc[2023:2025].sum() / e.loc[2023:2025].sum() * PER), 3),
            "registrants_growth_2011_to_2025": round(float(e.loc[2025] / e.loc[2011]), 3),
            "decisions_growth_2011_13_to_2023_25": round(float(c.loc[2023:2025].sum() / c.loc[2011:2013].sum()), 3),
            "trend": trend(c, e),
        }
    return out


def main():
    counts, unknown = load_counts()
    reg = load_registrants(include_subregister=False)
    results = {
        "years": [YEARS.start, YEARS.stop - 1],
        "numerator": "classified disciplinary decisions by decision year and profession field",
        "denominator": "Ahpra registrants at 30 June of the decision year (incl. non-practising)",
        "unknown_profession_decisions_excluded": int(unknown.sum()),
        "primary": summarise(counts, reg),
        "sensitivity_including_pandemic_subregister": summarise(counts, load_registrants(True)),
        "sensitivity_unknown_profession_allocated": summarise(allocate_unknown(counts, unknown), reg),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    table = []
    for key, label in list(REPORTED.items()) + [("ALL_REGULATED_SINCE_2010", "Ten professions regulated since 2010")]:
        p = results["primary"][key]
        table.append({"Profession": label, "Decisions 2011-2025": p["decisions"],
                      "Rate 2011-13": p["rate_2011_2013"], "Rate 2023-25": p["rate_2023_2025"],
                      "Registrant growth 2011-25": p["registrants_growth_2011_to_2025"],
                      "Rate ratio per year": p["trend"]["rate_ratio_per_year"],
                      "95% CI": "{:.3f}-{:.3f}".format(*p["trend"]["ci95"]),
                      "p": p["trend"]["p_value"]})
    pd.DataFrame(table).to_csv(TABLES / "table34_registrant_rates.csv", index=False)

    with plt.style.context(["science", "nature", "no-latex"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        # Chiropractic (38 decisions) is in Table 34 but not plotted: its small counts swamp the axis.
        plotted = [(k, v) for k, v in REPORTED.items() if k != "chiropractor"]
        for key, label in plotted + [("ALL_REGULATED_SINCE_2010", "All (ten professions)")]:
            s = results["primary"][key]["rate_per_10000_by_year"]
            ax.plot(list(s), list(s.values()), marker="o", ms=3, label=label,
                    lw=2.5 if key.startswith("ALL") else 1.2, color=COLOURS[key])
        ax.set_xlabel("Decision year")
        ax.set_ylabel("Disciplinary decisions per 10,000 registrants")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        out = Path(os.path.relpath(FIGURES)) / "fig19_registrant_rates"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
    print(json.dumps({k: {x: v[x] for x in ("decisions", "rate_2011_2013", "rate_2023_2025", "trend")}
                      for k, v in results["primary"].items()}, indent=1))


if __name__ == "__main__":
    main()
