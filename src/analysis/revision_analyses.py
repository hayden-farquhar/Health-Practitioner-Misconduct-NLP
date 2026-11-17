"""Revision analyses: FDR correction, error-adjusted prevalence, data fixes,
misconduct co-occurrence profiles, and Mann-Kendall sensitivity.

Reads existing statistical results and produces revised outputs for
manuscript revision addressing reviewer feedback.

Usage:
    .venv-ml/bin/python -m src.analysis.revision_analyses
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CORPUS_PATH = PROJECT_ROOT / "data" / "extracted" / "enriched_corpus.jsonl"
LAST_COMPLETE_YEAR = 2025  # corpus assembled Feb 2026; count-based trends stop here

MISCONDUCT_CODES = ["CLIN", "PRESC", "BOUND", "PROF", "FRAUD", "IMPR", "COMM"]
MISCONDUCT_LABELS = {
    "CLIN": "Clinical competence",
    "PRESC": "Prescribing",
    "BOUND": "Boundary violations",
    "PROF": "Professional conduct",
    "FRAUD": "Dishonesty/fraud",
    "IMPR": "Impairment",
    "COMM": "Communication",
}


def _load_disciplinary_decisions() -> list[dict]:
    """Load enriched corpus and return disciplinary decisions only."""
    decisions = []
    with open(CORPUS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("predicted_is_disciplinary"):
                decisions.append(rec)
    return decisions


def _build_binary_matrix(decisions: list[dict]) -> np.ndarray:
    """Build an n x 7 binary matrix of misconduct type assignments."""
    matrix = np.zeros((len(decisions), len(MISCONDUCT_CODES)), dtype=int)
    for i, d in enumerate(decisions):
        predicted = d.get("predicted_misconduct_L1", [])
        for j, code in enumerate(MISCONDUCT_CODES):
            if code in predicted:
                matrix[i, j] = 1
    return matrix

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

    # Load observed counts from summary statistics (dynamically)
    with open(STATS_DIR / "summary_statistics.json") as f:
        summary = json.load(f)
    n_disc = summary["disciplinary_decisions"]

    n_test = 45

    observed_counts = summary["misconduct_prevalence"]

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
# 4. Misconduct co-occurrence profiles and clustering
# ---------------------------------------------------------------------------

def misconduct_profiles():
    """Compute misconduct co-occurrence profiles, network, and clusters."""
    print("\n--- Misconduct Co-occurrence Profiles ---")

    decisions = _load_disciplinary_decisions()
    n_disc = len(decisions)
    matrix = _build_binary_matrix(decisions)
    print(f"  Disciplinary decisions: {n_disc}")

    # --- Phi coefficient matrix (Pearson correlation on binary = phi) ---
    # Use manual computation to handle zero-variance columns
    n_codes = len(MISCONDUCT_CODES)
    phi_matrix = np.zeros((n_codes, n_codes))
    for i in range(n_codes):
        for j in range(n_codes):
            if i == j:
                phi_matrix[i, j] = 1.0
            else:
                a = matrix[:, i]
                b = matrix[:, j]
                n11 = np.sum((a == 1) & (b == 1))
                n10 = np.sum((a == 1) & (b == 0))
                n01 = np.sum((a == 0) & (b == 1))
                n00 = np.sum((a == 0) & (b == 0))
                denom = np.sqrt(
                    (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
                )
                if denom > 0:
                    phi_matrix[i, j] = (n11 * n00 - n10 * n01) / denom
                else:
                    phi_matrix[i, j] = 0.0

    # Save phi matrix as table
    phi_df = pd.DataFrame(
        np.round(phi_matrix, 3), index=MISCONDUCT_CODES, columns=MISCONDUCT_CODES
    )
    phi_df.to_csv(TABLES_DIR / "table30_phi_coefficients.csv")
    print("  Phi coefficient matrix computed")

    # --- Co-occurrence network graph (fig18) ---
    G = nx.Graph()
    prevalences = {}
    for i, code in enumerate(MISCONDUCT_CODES):
        prev = matrix[:, i].sum() / n_disc
        prevalences[code] = prev
        G.add_node(code, prevalence=prev)

    for i in range(n_codes):
        for j in range(i + 1, n_codes):
            phi = phi_matrix[i, j]
            if abs(phi) > 0.02:
                G.add_edge(
                    MISCONDUCT_CODES[i], MISCONDUCT_CODES[j],
                    weight=abs(phi), phi=round(phi, 3),
                )

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100)

    # Node sizes proportional to prevalence
    node_sizes = [prevalences[c] * 5000 + 300 for c in G.nodes()]
    node_colours = plt.cm.Set2(np.linspace(0, 1, len(G.nodes())))

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colours,
        edgecolors="black", linewidths=1.0, ax=ax,
    )

    # Node labels with prevalence
    labels = {c: f"{c}\n({prevalences[c]:.0%})" for c in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)

    # Edges: green for positive phi, red for negative
    edges = list(G.edges(data=True))
    if edges:
        e_widths = [e[2]["weight"] * 15 + 0.5 for e in edges]
        e_colours = ["#2ca02c" if e[2]["phi"] > 0 else "#d62728" for e in edges]
        e_styles = ["solid" if abs(e[2]["phi"]) > 0.05 else "dashed" for e in edges]
        nx.draw_networkx_edges(
            G, pos, width=e_widths, edge_color=e_colours,
            style=e_styles, alpha=0.7, ax=ax,
        )
        edge_labels = {(e[0], e[1]): f"φ={e[2]['phi']:.2f}" for e in edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=8, font_color="gray", ax=ax,
        )

    ax.set_title(
        "Misconduct Type Co-occurrence Network\n"
        "(node size ∝ prevalence; green = positive association, red = negative)",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig18_misconduct_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Network graph saved: fig18_misconduct_network.png")

    # --- Multi-label combinations (n >= 10) ---
    combo_counts: dict[tuple, int] = {}
    for row in matrix:
        combo = tuple(
            MISCONDUCT_CODES[j] for j in range(n_codes) if row[j]
        )
        if combo:
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

    significant_combos = sorted(
        [(c, n) for c, n in combo_counts.items() if n >= 10],
        key=lambda x: -x[1],
    )

    combo_rows = []
    for combo, count in significant_combos:
        combo_rows.append({
            "combination": " + ".join(combo),
            "n_types": len(combo),
            "n": count,
            "pct": round(count / n_disc * 100, 1),
        })
    combo_df = pd.DataFrame(combo_rows)
    combo_df.to_csv(TABLES_DIR / "table31_misconduct_combinations.csv", index=False)
    print(f"  Misconduct combinations (n>=10): {len(significant_combos)}")

    # --- Hierarchical clustering (Ward's method) ---
    # Only cluster decisions that have at least one misconduct label
    has_label = matrix.sum(axis=1) > 0
    matrix_labelled = matrix[has_label]
    decisions_labelled = [d for d, h in zip(decisions, has_label) if h]

    try:
        from sklearn.metrics import silhouette_score

        Z = linkage(matrix_labelled, method="ward", metric="euclidean")

        # Find optimal k by silhouette score
        best_k = 2
        best_score = -1
        scores = {}
        for k in range(2, 9):
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels)) > 1:
                score = silhouette_score(
                    matrix_labelled, labels, metric="euclidean",
                    sample_size=min(5000, len(matrix_labelled)),
                    random_state=42,
                )
                scores[k] = round(score, 4)
                if score > best_score:
                    best_score = score
                    best_k = k

        print(f"  Optimal clusters: k={best_k} (silhouette={best_score:.3f})")
        print(f"  Silhouette scores: {scores}")

        cluster_labels = fcluster(Z, t=best_k, criterion="maxclust")

        # Characterise each cluster
        cluster_profiles = []
        for k in range(1, best_k + 1):
            mask = cluster_labels == k
            cl_matrix = matrix_labelled[mask]
            cl_decisions = [
                d for d, m in zip(decisions_labelled, mask) if m
            ]

            # Misconduct prevalence within cluster
            cl_prevs = {
                MISCONDUCT_CODES[j]: round(cl_matrix[:, j].mean() * 100, 1)
                for j in range(n_codes)
            }
            dominant = sorted(cl_prevs.items(), key=lambda x: -x[1])

            # Top professions
            prof_counts: dict[str, int] = {}
            for d in cl_decisions:
                p = d.get("profession", "unknown") or "unknown"
                p = p.replace("_", " ").title()
                prof_counts[p] = prof_counts.get(p, 0) + 1
            top_profs = sorted(prof_counts.items(), key=lambda x: -x[1])[:3]

            # Cancellation rate
            cancel_n = sum(
                1 for d in cl_decisions
                if "CANCEL" in (d.get("predicted_outcomes") or [])
            )
            cancel_rate = round(cancel_n / len(cl_decisions) * 100, 1) if cl_decisions else 0

            cluster_profiles.append({
                "cluster": k,
                "n": int(mask.sum()),
                "pct_of_disciplinary": round(mask.sum() / n_disc * 100, 1),
                "dominant_misconduct": ", ".join(
                    f"{c} ({v}%)" for c, v in dominant if v > 5
                ),
                "top_professions": ", ".join(
                    f"{p} ({n})" for p, n in top_profs
                ),
                "cancellation_rate_pct": cancel_rate,
            })

        profile_df = pd.DataFrame(cluster_profiles)
        profile_df.to_csv(TABLES_DIR / "table32_misconduct_profiles.csv", index=False)
        print(f"  Cluster profiles saved: table32_misconduct_profiles.csv")

    except ImportError:
        print("  WARNING: sklearn not available; skipping clustering")
        cluster_profiles = []
        best_k = 0
        best_score = 0
        scores = {}

    # --- Summary statistics ---
    summary = {
        "n_disciplinary": n_disc,
        "phi_matrix": {
            MISCONDUCT_CODES[i]: {
                MISCONDUCT_CODES[j]: round(phi_matrix[i, j], 3)
                for j in range(n_codes)
            }
            for i in range(n_codes)
        },
        "n_combinations_gte10": len(significant_combos),
        "top_combinations": [
            {"combination": " + ".join(c), "n": n} for c, n in significant_combos[:10]
        ],
        "optimal_k": best_k,
        "silhouette_score": round(best_score, 3),
        "silhouette_scores": scores,
        "cluster_profiles": cluster_profiles,
    }

    with open(STATS_DIR / "misconduct_profiles.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  Saved: misconduct_profiles.json")
    return summary


# ---------------------------------------------------------------------------
# 5. Mann-Kendall sensitivity: excluding pre-2005 years
# ---------------------------------------------------------------------------

def mk_sensitivity_pre2005():
    """Re-run Mann-Kendall trend tests excluding years before 2005."""
    print("\n--- Mann-Kendall Sensitivity: Excluding Pre-2005 ---")

    import pymannkendall as mk

    decisions = _load_disciplinary_decisions()

    # Build year -> misconduct counts
    year_counts: dict[int, dict[str, int]] = {}
    year_totals: dict[int, int] = {}
    for d in decisions:
        date = d.get("date", "")
        if not date:
            continue
        try:
            year = int(date[:4])
        except (ValueError, IndexError):
            continue
        if year not in year_counts:
            year_counts[year] = {c: 0 for c in MISCONDUCT_CODES}
            year_totals[year] = 0
        year_totals[year] += 1
        for code in MISCONDUCT_CODES:
            if code in (d.get("predicted_misconduct_L1") or []):
                year_counts[year][code] += 1

    # Full range results
    all_years = sorted(year_counts.keys())
    # Filtered: 2005+
    filtered_years = [y for y in all_years if y >= 2005]

    results = []
    for code in MISCONDUCT_CODES + ["TOTAL"]:
        for label, years in [("full_range", all_years), ("2005_onwards", filtered_years)]:
            if code == "TOTAL":
                # Counts need complete calendar years: drop the partial final year and
                # count years with no decisions as zero (matches statistical_analysis).
                years = list(range(years[0], LAST_COMPLETE_YEAR + 1))
                series = [year_totals.get(y, 0) for y in years]
            else:
                series = [
                    year_counts[y][code] / year_totals[y] * 100
                    if year_totals.get(y, 0) > 0 else 0
                    for y in years
                ]

            if len(series) < 4:
                continue

            try:
                result = mk.original_test(series)
                results.append({
                    "misconduct_type": code,
                    "range": label,
                    "n_years": len(years),
                    "year_start": years[0],
                    "year_end": years[-1],
                    "tau": round(result.Tau, 3),
                    "p_value": round(result.p, 6),
                    "slope": round(result.slope, 4),
                    "trend": result.trend,
                    "significant": bool(result.p < 0.05),
                })
            except Exception as e:
                print(f"  WARNING: MK test failed for {code} ({label}): {e}")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "table33_mk_sensitivity_pre2005.csv", index=False)

    # Compare full vs filtered
    print(f"\n  {'Type':<8} {'Full tau':>10} {'Full p':>10} {'2005+ tau':>10} {'2005+ p':>10} {'Consistent':>12}")
    for code in MISCONDUCT_CODES + ["TOTAL"]:
        full = next((r for r in results if r["misconduct_type"] == code and r["range"] == "full_range"), None)
        filt = next((r for r in results if r["misconduct_type"] == code and r["range"] == "2005_onwards"), None)
        if full and filt:
            consistent = (full["significant"] == filt["significant"]) or (
                full["trend"] == filt["trend"]
            )
            print(f"  {code:<8} {full['tau']:>10.3f} {full['p_value']:>10.4f} "
                  f"{filt['tau']:>10.3f} {filt['p_value']:>10.4f} {'Yes' if consistent else 'CHANGED':>12}")

    with open(STATS_DIR / "mk_sensitivity_pre2005.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: table33_mk_sensitivity_pre2005.csv, mk_sensitivity_pre2005.json")
    return results


# ---------------------------------------------------------------------------
# 6. Fix year range and COVID denominator discrepancy
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
    profiles = misconduct_profiles()
    mk_sens = mk_sensitivity_pre2005()
    fixes = fix_data_discrepancies()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  FDR: {fdr['n_significant_fdr']}/{fdr['n_tests']} tests remain significant")
    print(f"  {len(fdr['tests_losing_significance'])} tests lost significance after correction")
    print(f"  Co-occurrence: {profiles['n_combinations_gte10']} combinations (n>=10)")
    print(f"  Clusters: k={profiles['optimal_k']} (silhouette={profiles['silhouette_score']:.3f})")
    print(f"  MK sensitivity: {len(mk_sens)} tests run")
    print(f"  Year range: {fixes['year_range']}")
    print(f"  COVID gap: {fixes['covid_gap']} decisions without dates")
    print(f"\nNew outputs:")
    print(f"  table27_fdr_correction.csv")
    print(f"  table28_adjusted_prevalence.csv")
    print(f"  table29_f1_bootstrap_ci.csv")
    print(f"  table30_phi_coefficients.csv")
    print(f"  table31_misconduct_combinations.csv")
    print(f"  table32_misconduct_profiles.csv")
    print(f"  table33_mk_sensitivity_pre2005.csv")
    print(f"  fig18_misconduct_network.png")
    print(f"  misconduct_profiles.json")
    print(f"  mk_sensitivity_pre2005.json")


if __name__ == "__main__":
    main()
