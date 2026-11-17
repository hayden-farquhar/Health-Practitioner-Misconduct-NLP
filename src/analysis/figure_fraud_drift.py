"""Figure 4: dishonesty/fraud in the coded sample by period, coded vs classifier, and classifier recall.

Reads only existing outputs: outputs/statistics/human_reference_estimates.json (coded prevalence
by period with bootstrap CIs) and outputs/statistics/test2_results.json (classifier prevalence
and recall by period on the same 160 decisions). Writes
outputs/figures/fig21_fraud_drift.png and .pdf.

Usage (from project root):
    .venv-ml/bin/python -m src.analysis.figure_fraud_drift
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the 'science' styles)

from src.analysis import validation_analysis as va

try:
    import _integrity_guard  # noqa: F401  (optional author-side check on figure text; not needed to reproduce)
except ImportError:
    pass

OKABE_ITO = {"blue": "#0072B2", "vermil": "#D55E00"}


def main():
    h = json.loads((va.STATS_DIR / "human_reference_estimates.json").read_text())
    t2 = json.loads((va.STATS_DIR / "test2_results.json").read_text())
    coded = h["misconduct_by_period"]["FRAUD"]["by_period"]
    periods = list(coded)
    pred = t2["trend_check"]["FRAUD"]["predicted_prevalence_by_period"]
    recall = {p: t2["by_period"][p]["FRAUD_recall"] for p in periods}
    x = np.arange(len(periods))

    with plt.style.context(["science", "nature", "no-latex"]):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
        a.errorbar(x - 0.06, [coded[p]["pct"] for p in periods],
                   yerr=[[coded[p]["pct"] - coded[p]["ci95"][0] for p in periods],
                         [coded[p]["ci95"][1] - coded[p]["pct"] for p in periods]],
                   fmt="o-", color=OKABE_ITO["blue"], ms=3, capsize=2, label="Coded (95% CI)")
        a.plot(x + 0.06, [100 * pred[p] for p in periods], "D--", color=OKABE_ITO["vermil"], ms=3,
               label="Classifier")
        a.set_xticks(x, periods)
        a.set_ylabel("Disciplinary decisions with\ndishonesty or fraud (%)")
        a.set_ylim(0, 80)
        a.set_title("(a) Prevalence", fontsize=7, loc="left")
        a.legend(fontsize=6, frameon=False, loc="upper left")

        b.errorbar(x, [100 * recall[p]["value"] for p in periods],
                   yerr=[[100 * (recall[p]["value"] - recall[p]["ci95"][0]) for p in periods],
                         [100 * (recall[p]["ci95"][1] - recall[p]["value"]) for p in periods]],
                   fmt="D-", color=OKABE_ITO["vermil"], ms=3, capsize=2)
        b.set_xticks(x, periods)
        b.set_ylabel("Coded fraud matters the\nclassifier detected (%)")
        b.set_ylim(0, 105)
        b.set_title("(b) Classifier recall", fontsize=7, loc="left")
        for ax in (a, b):
            ax.set_xlabel("Decision period")
        fig.tight_layout()
        out = Path(os.path.relpath(va.FIGURES_DIR)) / "fig21_fraud_drift"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
    print({p: (coded[p]["pct"], round(100 * pred[p], 1), round(100 * recall[p]["value"], 1)) for p in periods})


if __name__ == "__main__":
    main()
