"""Revision analyses: FDR correction, error-adjusted prevalence, data fixes.

Reads existing statistical results and produces revised outputs for
manuscript revision addressing reviewer feedback.

Usage:
    .venv-ml/bin/python -m src.analysis.revision_analyses
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# ---------------------------------------------------------------------------
# 1. FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def fdr_correction():
    """Apply Benjamini-Hochberg FDR correction to all p-values."""
    print("\n--- FDR Correction (Benjamini-Hochberg) ---")

    # Collect all p-values from chi-square results
    with open(STATS_DIR / "chi_square_results.json") as f:
        chi2_results = json.load(f)
    with open(STATS_DIR / "trend_tests.json") as f:
        trend_results = json.load(f)

    # Build a table of all tests and p-values
    all_tests = []
    for r in chi2_results:
        all_tests.append({
            "test": r["test"],
            "source": "chi_square",
            "p_value": r["p_value"],
        })
    for r in trend_results:
        all_tests.append({
            "test": f"trend_{r['misconduct_type']}",
            "source": "mann_kendall",
            "p_value": r["p_value"],
        })

    # Also add temporal split p-values from validation
    with open(STATS_DIR / "validation_results.json") as f:
        val_results = json.load(f)
    for comp in val_results["sensitivity"]["temporal_split"]["misconduct_comparison"]:
        all_tests.append({
            "test": f"temporal_split_{comp['Misconduct type']}",
            "source": "temporal_split",
            "p_value": comp["p-value"],
        })

    n_tests = len(all_tests)
    print(f"  Total tests: {n_tests}")

    # Sort by p-value for BH procedure
    all_tests.sort(key=lambda x: x["p_value"])

    # Benjamini-Hochberg
    for rank, t in enumerate(all_tests, 1):
        t["rank"] = rank
        t["bh_critical"] = (rank / n_tests) * 0.05
        t["p_adjusted"] = min(t["p_value"] * n_tests / rank, 1.0)

    # Step-up adjustment: ensure monotonicity (adjusted p-values must be non-decreasing from bottom)
    for i in range(len(all_tests) - 2, -1, -1):
        all_tests[i]["p_adjusted"] = min(all_tests[i]["p_adjusted"],
                                         all_tests[i + 1]["p_adjusted"])

    # Determine which are still significant
    for t in all_tests:
        t["significant_unadjusted"] = t["p_value"] < 0.05
        t["significant_fdr"] = t["p_adjusted"] < 0.05

    n_sig_unadjusted = sum(1 for t in all_tests if t["significant_unadjusted"])
    n_sig_fdr = sum(1 for t in all_tests if t["significant_fdr"])
    print(f"  Significant (unadjusted): {n_sig_unadjusted}/{n_tests}")
    print(f"  Significant (FDR q<0.05): {n_sig_fdr}/{n_tests}")

    # Report which tests lost significance
    lost = [t for t in all_tests if t["significant_unadjusted"] and not t["significant_fdr"]]
    if lost:
        print(f"  Tests losing significance after FDR:")
        for t in lost:
            print(f"    {t['test']}: p={t['p_value']:.4g} -> q={t['p_adjusted']:.4g}")
    else:
        print("  No tests lost significance after FDR correction.")

    # Save full results
    df = pd.DataFrame(all_tests)
    df.to_csv(TABLES_DIR / "table27_fdr_correction.csv", index=False)

    # Save JSON for manuscript reference
    fdr_summary = {
        "n_tests": n_tests,
        "n_significant_unadjusted": n_sig_unadjusted,
        "n_significant_fdr": n_sig_fdr,
        "tests_losing_significance": [
            {"test": t["test"], "p_unadjusted": t["p_value"], "p_adjusted": t["p_adjusted"]}
            for t in lost
        ],
        "all_tests": [
            {
                "test": t["test"],
                "source": t["source"],
                "p_unadjusted": t["p_value"],
                "p_adjusted": round(t["p_adjusted"], 6),
                "significant_fdr": t["significant_fdr"],
            }
            for t in all_tests
        ],
    }
    with open(STATS_DIR / "fdr_correction.json", "w") as f:
        json.dump(fdr_summary, f, indent=2)

    print(f"  Saved: table27_fdr_correction.csv, fdr_correction.json")
    return fdr_summary


# ---------------------------------------------------------------------------
# 2. Error-adjusted prevalence estimates
# ---------------------------------------------------------------------------

def error_adjusted_prevalence():
    """Compute prevalence estimates adjusted for known classifier error rates.

    Uses the Rogan-Gladen estimator:
        True prevalence = (Apparent prevalence + Specificity - 1) / (Sensitivity + Specificity - 1)

    where Sensitivity = recall, Specificity = 1 - FPR.
    """
    print("\n--- Error-Adjusted Prevalence Estimates ---")

    with open(STATS_DIR / "validation_results.json") as f:
        val = json.load(f)

    per_class = val["classification"]["misconduct"]["per_class"]
    n_disc = 2657  # disciplinary decisions

    # For each misconduct category, compute adjusted prevalence
    # The classifier is multi-label binary, so for each class:
    #   TP = recall * support
    #   FP = TP/precision - TP  (when precision > 0)
    #   Predicted positive rate in test set
    #   True positive rate in test set = support / n_test_disc
    # But we need rates on the FULL corpus, so we use test-set precision/recall
    # as estimates of the classifier's operating characteristics.

    n_test = 45
    # We need per-class: sensitivity (recall) and specificity
    # Specificity = TN / (TN + FP)
    # From precision and recall and support:
    #   TP = recall * support
    #   FP = TP * (1 - precision) / precision  (when precision > 0)
    #   FN = support - TP
    #   TN = n_test - support - FP

    observed_counts = {
        "CLIN": 680, "PRESC": 675, "BOUND": 800,
        "PROF": 868, "FRAUD": 334, "IMPR": 113, "COMM": 54,
    }

    results = []
    for code, metrics in per_class.items():
        if code == "N/A":
            continue

        precision = metrics["precision"]
        recall = metrics["recall"]
        support = metrics["support"]
        f1 = metrics["f1"]

        # Compute TP, FP, FN, TN from test set
        tp = round(recall * support)
        if precision > 0:
            predicted_pos = round(tp / precision)
            fp = predicted_pos - tp
        else:
            fp = 0
        fn = support - tp
        tn = n_test - support - fp

        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        # Observed (apparent) prevalence from classifier predictions
        observed_n = observed_counts.get(code, 0)
        apparent_prev = observed_n / n_disc

        # Rogan-Gladen adjusted prevalence
        denom = sensitivity + specificity - 1
        if abs(denom) > 0.01:
            adjusted_prev = (apparent_prev + specificity - 1) / denom
            adjusted_prev = max(0, min(1, adjusted_prev))
            adjusted_n = round(adjusted_prev * n_disc)
        else:
            # Denominator near zero — adjustment unreliable
            adjusted_prev = apparent_prev
            adjusted_n = observed_n

        # Direction of likely bias
        if recall < 1.0 and precision == 1.0:
            bias = "underestimate (low recall, no FP)"
        elif recall < 1.0 and precision < 1.0:
            bias = "uncertain (both FP and FN)"
        elif precision < 1.0:
            bias = "overestimate (FP present)"
        else:
            bias = "minimal"

        results.append({
            "code": code,
            "observed_n": observed_n,
            "observed_pct": round(apparent_prev * 100, 1),
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "adjusted_pct": round(adjusted_prev * 100, 1),
            "adjusted_n": adjusted_n,
            "direction_of_bias": bias,
            "test_support": support,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        })

        print(f"  {code}: observed={apparent_prev*100:.1f}% -> adjusted={adjusted_prev*100:.1f}% "
              f"(sens={sensitivity:.2f}, spec={specificity:.2f}, bias={bias})")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "table28_adjusted_prevalence.csv", index=False)

    with open(STATS_DIR / "adjusted_prevalence.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: table28_adjusted_prevalence.csv, adjusted_prevalence.json")
    return results


# ---------------------------------------------------------------------------
# 3. Bootstrap CIs for per-class F1
# ---------------------------------------------------------------------------

def bootstrap_f1_confidence_intervals(n_bootstrap=10000):
    """Bootstrap 95% CIs for per-class F1 scores on the test set."""
    print("\n--- Bootstrap CIs for Per-Class F1 ---")

    with open(STATS_DIR / "validation_results.json") as f:
        val = json.load(f)

    per_class = val["classification"]["misconduct"]["per_class"]
    n_test = 45
    rng = np.random.default_rng(42)

    results = []
    for code, metrics in per_class.items():
        if code == "N/A":
            continue

        support = metrics["support"]
        recall = metrics["recall"]
        precision = metrics["precision"]

        # Reconstruct the binary arrays for this class
        tp = round(recall * support)
        if precision > 0:
            predicted_pos = round(tp / precision)
            fp = predicted_pos - tp
        else:
            fp = 0
        fn = support - tp
        tn = n_test - support - fp

        # Create binary truth and pred arrays
        y_true = np.array([1]*support + [0]*(n_test - support))
        y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)

        # Bootstrap
        f1_samples = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_test, size=n_test, replace=True)
            yt = y_true[idx]
            yp = y_pred[idx]
            tp_b = np.sum((yt == 1) & (yp == 1))
            fp_b = np.sum((yt == 0) & (yp == 1))
            fn_b = np.sum((yt == 1) & (yp == 0))
            if tp_b + fp_b + fn_b == 0:
                f1_samples.append(0.0)
            else:
                prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
                rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
                f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0
                f1_samples.append(f1_b)

        f1_arr = np.array(f1_samples)
        ci_lower = np.percentile(f1_arr, 2.5)
        ci_upper = np.percentile(f1_arr, 97.5)

        results.append({
            "code": code,
            "f1": metrics["f1"],
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),
            "support": support,
        })
        print(f"  {code}: F1={metrics['f1']:.3f} (95% CI: {ci_lower:.3f}--{ci_upper:.3f}), n={support}")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "table29_f1_bootstrap_ci.csv", index=False)

    with open(STATS_DIR / "f1_bootstrap_ci.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: table29_f1_bootstrap_ci.csv, f1_bootstrap_ci.json")
    return results


# ---------------------------------------------------------------------------
# 4. Fix year range and COVID denominator discrepancy
# ---------------------------------------------------------------------------

def fix_data_discrepancies():
    """Document and fix known data discrepancies."""
    print("\n--- Data Discrepancy Checks ---")

    # Year range
    corpus_path = PROJECT_ROOT / "data" / "extracted" / "enriched_corpus.jsonl"
    years = []
    n_no_date = 0
    with open(corpus_path) as f:
        for line in f:
            rec = json.loads(line)
            date = rec.get("date", "")
            if date:
                try:
                    year = int(date[:4])
                    years.append(year)
                except (ValueError, IndexError):
                    n_no_date += 1
            else:
                n_no_date += 1

    min_year = min(years) if years else None
    max_year = max(years) if years else None
    print(f"  Year range: {min_year}--{max_year} (from {len(years)} dated decisions, {n_no_date} without date)")

    if max_year and max_year >= 2026:
        n_2026 = sum(1 for y in years if y >= 2026)
        print(f"  Decisions dated 2026+: {n_2026}")
        print(f"  These are likely early 2026 decisions. Title says 1999-2025.")
        print(f"  Recommendation: keep '1999-2025' in title (study period), note in methods")
    else:
        print(f"  Year range consistent with title (1999--2025)")

    # COVID denominators
    # Pre-COVID: <2020, COVID: 2020-2021, Post-COVID: 2022+
    # Need to check if decisions without dates are excluded
    with open(STATS_DIR / "summary_statistics.json") as f:
        summary = json.load(f)
    covid = summary.get("covid", {})
    total_covid = sum(v for v in covid.values() if isinstance(v, (int, float)))
    total_disc = summary["disciplinary_decisions"]
    gap = total_disc - total_covid
    print(f"  COVID periods sum: {total_covid}, disciplinary total: {total_disc}, gap: {gap}")
    if gap > 0:
        print(f"  Gap of {gap} decisions likely due to missing dates")

    # Update summary_statistics.json year range
    summary["year_range"] = f"{min_year}--{max_year}" if min_year and max_year else summary["year_range"]
    summary["year_range_note"] = (
        f"Latest decision dated {max_year}. Title uses 1999-2025 as the study period. "
        f"{n_no_date} decisions lack parseable dates."
    )
    summary["covid"]["note"] = f"{gap} disciplinary decisions excluded from COVID analysis due to missing dates"

    with open(STATS_DIR / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Fix table01 year range
    table01 = pd.read_csv(TABLES_DIR / "table01_corpus_overview.csv")
    for i, row in table01.iterrows():
        if "Year range" in str(row.get("Metric", "")):
            table01.at[i, "Value"] = f"{min_year}\u2013{max_year}"
    table01.to_csv(TABLES_DIR / "table01_corpus_overview.csv", index=False)

    print(f"  Updated summary_statistics.json and table01_corpus_overview.csv")

    return {
        "year_range": f"{min_year}--{max_year}",
        "n_2026_decisions": sum(1 for y in years if y >= 2026) if max_year and max_year >= 2026 else 0,
        "n_no_date": n_no_date,
        "covid_gap": gap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Revision Analyses")
    print("=" * 60)

    fdr = fdr_correction()
    adj = error_adjusted_prevalence()
    f1_ci = bootstrap_f1_confidence_intervals()
    fixes = fix_data_discrepancies()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  FDR: {fdr['n_significant_fdr']}/{fdr['n_tests']} tests remain significant")
    print(f"  {len(fdr['tests_losing_significance'])} tests lost significance after correction")
    print(f"  Year range: {fixes['year_range']}")
    print(f"  COVID gap: {fixes['covid_gap']} decisions without dates")
    print(f"\nNew outputs:")
    print(f"  table27_fdr_correction.csv")
    print(f"  table28_adjusted_prevalence.csv")
    print(f"  table29_f1_bootstrap_ci.csv")
    print(f"  fdr_correction.json")
    print(f"  adjusted_prevalence.json")
    print(f"  f1_bootstrap_ci.json")


if __name__ == "__main__":
    main()
