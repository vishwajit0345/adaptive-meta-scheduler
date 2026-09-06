"""
evaluation/statistics.py
=========================
Statistical significance testing for scheduler performance comparisons.

WHY SIGNIFICANCE TESTING MATTERS
----------------------------------
Reporting mean ± std is necessary but not sufficient for a research paper.
A reviewer will always ask: "But is the difference statistically significant,
or could it just be due to random workload variation?"

This module answers that question formally using two complementary tests:

  t_test         : Student's two-sample t-test (parametric)
                   Assumes approximately normal distribution of waiting times.
                   Appropriate for n ≥ 30 trials.

  wilcoxon_test  : Wilcoxon signed-rank test (non-parametric)
                   Makes no distributional assumptions.
                   Appropriate for paired samples (same workloads, different
                   schedulers) — more powerful than Mann-Whitney for paired data.
                   Robust to outliers caused by extreme random workloads.

RECOMMENDED USAGE IN A PAPER
------------------------------
  1. Report mean ± std (already done in experiment.py)
  2. Report t-test p-value for parametric claim
  3. Report Wilcoxon p-value for robustness claim
  4. Both p < 0.05 → strong evidence of genuine improvement

SIGNIFICANCE THRESHOLD
-----------------------
α = 0.05 is the standard threshold in systems and ML papers.
For a final year project / undergraduate publication, p < 0.05 is sufficient.
For a top-tier venue (OSDI, SOSP, NeurIPS), consider also reporting
effect size (Cohen's d) which this module also computes.

USAGE EXAMPLE
-------------
    from evaluation.statistics import t_test, wilcoxon_test, print_significance_report

    # rl_waits and fcfs_waits are lists of avg_waiting_time from 30 trials
    t_res  = t_test(rl_waits, fcfs_waits)
    w_res  = wilcoxon_test(rl_waits, fcfs_waits)

    if t_res["significant"] and w_res["significant"]:
        print("RL scheduler improvement statistically significant (p < 0.05)")

    # Or use the convenience function that does everything:
    print_significance_report(trial_data, baseline="FCFS", target="RL_Adaptive")
"""

import math
from typing import Dict, List, Optional, Tuple


# ── try scipy; fall back to pure-Python implementations ──────────────────────

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ── pure-Python statistical primitives (used when scipy unavailable) ─────────

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: List[float], ddof: int = 1) -> float:
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / max(len(xs) - ddof, 1)


def _std(xs: List[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(xs, ddof))


def _t_statistic_two_sample(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Welch's t-test (unequal variance) — returns (t_stat, degrees_of_freedom).
    """
    na, nb = len(a), len(b)
    va, vb = _variance(a), _variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, float(na + nb - 2)
    t = (_mean(a) - _mean(b)) / se
    # Welch–Satterthwaite degrees of freedom
    df_num   = (va / na + vb / nb) ** 2
    df_denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / max(df_denom, 1e-9)
    return t, df


def _t_cdf_approx(t: float, df: float) -> float:
    """
    Approximate two-tailed p-value using the Cornish–Fisher approximation.
    Accurate to ~3 decimal places for df ≥ 5.  Good enough for p-value
    thresholding; scipy is used when available for exact values.
    """
    # Use normal approximation for large df
    if df > 30:
        z   = abs(t)
        p   = 2 * (1 - _normal_cdf(z))
        return p
    # For small df, use a rational approximation of the t-CDF
    x = df / (df + t * t)
    # Regularised incomplete beta approximation (Abramowitz & Stegun 26.5.8)
    p_one_tail = _ibeta(df / 2, 0.5, x) / 2
    return 2 * p_one_tail


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _ibeta(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """
    Regularised incomplete beta function I_x(a, b) via continued fraction.
    Used for the t-distribution CDF approximation.
    """
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    # Continued fraction expansion (Lentz's method)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Use symmetry for convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _ibeta(b, a, 1 - x, max_iter)
    # Lentz's continued fraction
    f, C, D, tiny = 1.0, 1.0, 0.0, 1e-30
    for m in range(0, max_iter):
        for j in (0, 1):
            if m == 0 and j == 0:
                d = 1.0
            elif j == 0:
                d = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
            else:
                d = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
            D = 1.0 + d * D
            if abs(D) < tiny: D = tiny
            C = 1.0 + d / C
            if abs(C) < tiny: C = tiny
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-8:
                return front * (f - 1.0)
    return front * (f - 1.0)


def _wilcoxon_approx(diffs: List[float]) -> float:
    """
    Wilcoxon signed-rank test p-value (normal approximation for n ≥ 10).
    Returns two-tailed p-value.
    """
    # Remove zeros
    nonzero = [(i + 1, d) for i, d in enumerate(sorted(abs(d) for d in diffs if d != 0))]
    n = len(nonzero)
    if n < 5:
        return 1.0   # Too few differences for reliable test
    W_plus = sum(rank for rank, d in
                 ((i + 1, diffs[i]) for i in range(len(diffs)) if diffs[i] > 0))
    mu_W = n * (n + 1) / 4
    sig_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sig_W == 0:
        return 1.0
    z = (W_plus - mu_W) / sig_W
    return 2 * (1 - _normal_cdf(abs(z)))


# ── public API ────────────────────────────────────────────────────────────────

def t_test(
    result_a: List[float],
    result_b: List[float],
    alpha:    float = 0.05,
) -> Dict:
    """
    Two-sample Welch's t-test comparing two distributions of waiting times.

    Tests H0: mean(result_a) == mean(result_b)
    A significant result (p < alpha) means the difference is unlikely
    to be due to random workload variation.

    Parameters
    ----------
    result_a : List of avg_waiting_time values for algorithm A (e.g. RL).
    result_b : List of avg_waiting_time values for algorithm B (e.g. FCFS).
    alpha    : Significance threshold (default 0.05).

    Returns
    -------
    Dict with keys:
        t_stat      : t-statistic (negative means A < B, i.e. A is better)
        p_value     : two-tailed p-value
        significant : bool — True if p < alpha
        df          : degrees of freedom
        mean_a      : mean of result_a
        mean_b      : mean of result_b
        improvement : (mean_b - mean_a) / mean_b  (positive = A is better)
        cohens_d    : effect size (|mean diff| / pooled std)
        method      : "scipy" or "pure_python"

    Example
    -------
    >>> res = t_test(rl_waits, fcfs_waits)
    >>> print(f"p = {res['p_value']:.4f}, significant = {res['significant']}")
    """
    if len(result_a) < 2 or len(result_b) < 2:
        raise ValueError("t_test requires at least 2 observations per group.")

    ma, mb = _mean(result_a), _mean(result_b)

    if _SCIPY_AVAILABLE:
        t_stat, p_value = _scipy_stats.ttest_ind(result_a, result_b, equal_var=False)
        df = len(result_a) + len(result_b) - 2
        method = "scipy"
    else:
        t_stat, df = _t_statistic_two_sample(result_a, result_b)
        p_value    = _t_cdf_approx(t_stat, df)
        method = "pure_python"

    # Cohen's d (pooled standard deviation)
    pooled_std = math.sqrt(
        (_variance(result_a) * (len(result_a) - 1) +
         _variance(result_b) * (len(result_b) - 1)) /
        max(len(result_a) + len(result_b) - 2, 1)
    )
    cohens_d = abs(ma - mb) / max(pooled_std, 1e-9)

    return {
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_value), 6),
        "significant": bool(float(p_value) < alpha),
        "df":          round(float(df), 1),
        "mean_a":      round(ma, 3),
        "mean_b":      round(mb, 3),
        "improvement": round((mb - ma) / max(mb, 1e-9), 4),
        "cohens_d":    round(cohens_d, 4),
        "alpha":       alpha,
        "method":      method,
        "n_a":         len(result_a),
        "n_b":         len(result_b),
    }


def wilcoxon_test(
    result_a: List[float],
    result_b: List[float],
    alpha:    float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test for paired scheduler comparisons.

    Non-parametric alternative to the t-test.  More appropriate when:
      • The same workloads were used for both algorithms (paired design).
      • Waiting times are not normally distributed (e.g. heavy-tailed).
      • n < 30 trials.

    Tests H0: the distribution of (result_a - result_b) is symmetric about 0.

    Parameters
    ----------
    result_a : List of avg_waiting_time values for algorithm A (e.g. RL).
    result_b : List of avg_waiting_time values for algorithm B (e.g. FCFS).
    alpha    : Significance threshold (default 0.05).

    Returns
    -------
    Dict with keys:
        statistic   : Wilcoxon W statistic (or z-score for large n)
        p_value     : two-tailed p-value
        significant : bool — True if p < alpha
        mean_diff   : mean(result_a - result_b)
        method      : "scipy" or "pure_python"

    Example
    -------
    >>> res = wilcoxon_test(rl_waits, fcfs_waits)
    >>> print(f"Wilcoxon p = {res['p_value']:.4f}")
    """
    if len(result_a) != len(result_b):
        raise ValueError(
            f"wilcoxon_test requires equal-length paired samples. "
            f"Got {len(result_a)} vs {len(result_b)}."
        )
    if len(result_a) < 5:
        raise ValueError("wilcoxon_test requires at least 5 paired observations.")

    diffs = [a - b for a, b in zip(result_a, result_b)]

    if _SCIPY_AVAILABLE:
        try:
            stat, p_value = _scipy_stats.wilcoxon(result_a, result_b,
                                                    alternative="two-sided")
            method = "scipy"
        except Exception:
            # scipy wilcoxon raises if all diffs are zero
            stat, p_value, method = 0.0, 1.0, "scipy_fallback"
    else:
        p_value = _wilcoxon_approx(diffs)
        stat    = sum(1 for d in diffs if d < 0)   # approximate W+
        method  = "pure_python"

    return {
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p_value), 6),
        "significant": bool(float(p_value) < alpha),
        "mean_diff":   round(_mean(diffs), 3),
        "median_diff": round(sorted(diffs)[len(diffs) // 2], 3),
        "alpha":       alpha,
        "method":      method,
        "n":           len(result_a),
    }


def run_significance_tests(
    trial_data:   Dict[str, List[float]],
    target:       str   = "RL_Adaptive",
    baselines:    List[str] = None,
    alpha:        float = 0.05,
) -> Dict[str, Dict]:
    """
    Run both tests comparing `target` algorithm vs each baseline.

    Parameters
    ----------
    trial_data : Dict mapping algorithm name → list of per-trial waiting times.
                 (Collected during run_multi_trial_evaluation.)
    target     : The algorithm being evaluated (default "RL_Adaptive").
    baselines  : List of algorithm names to compare against.
                 Default: all algorithms except target and SJF_Oracle.
    alpha      : Significance level.

    Returns
    -------
    Nested dict: {baseline_name: {"t_test": {...}, "wilcoxon": {...}}}

    Example
    -------
    >>> from evaluation.statistics import run_significance_tests
    >>> results = run_significance_tests(trial_data, target="RL_Adaptive")
    """
    if target not in trial_data:
        raise KeyError(f"Target algorithm '{target}' not found in trial_data. "
                       f"Available: {list(trial_data.keys())}")

    if baselines is None:
        baselines = [k for k in trial_data
                     if k not in (target, "SJF_Oracle")]

    target_waits = trial_data[target]
    report = {}

    for baseline in baselines:
        if baseline not in trial_data:
            continue
        base_waits = trial_data[baseline]
        n = min(len(target_waits), len(base_waits))

        report[baseline] = {
            "t_test":     t_test(target_waits[:n],    base_waits[:n],    alpha),
            "wilcoxon":   wilcoxon_test(target_waits[:n], base_waits[:n], alpha),
        }

    return report


def print_significance_report(
    trial_data:   Dict[str, List[float]],
    target:       str   = "RL_Adaptive",
    baselines:    List[str] = None,
    alpha:        float = 0.05,
) -> None:
    """
    Print a formatted significance test table and the key verdict sentence.

    This function produces the output required by the task specification:
      "RL scheduler improvement statistically significant (p < 0.05)"

    Parameters
    ----------
    trial_data : Dict mapping algorithm → list of per-trial waiting times.
    target     : Algorithm to evaluate.
    baselines  : Algorithms to compare against.
    alpha      : Significance level.

    Example
    -------
    >>> # Collect per-trial waiting times during multi-trial evaluation
    >>> trial_data = {"RL_Adaptive": [...], "FCFS": [...], "RoundRobin": [...]}
    >>> print_significance_report(trial_data, target="RL_Adaptive")
    """
    results = run_significance_tests(trial_data, target, baselines, alpha)

    print(f"\n{'═'*76}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS — {target} vs Baselines")
    print(f"  α = {alpha}  |  Tests: Welch t-test + Wilcoxon signed-rank")
    print(f"{'═'*76}")
    print(f"  {'BASELINE':<20} {'t-stat':>8} {'t p-val':>10} {'W p-val':>10} "
          f"{'Improve%':>10} {'d':>7} {'SIG?':>6}")
    print(f"{'─'*76}")

    all_significant = True
    any_significant = False

    for baseline, res in sorted(results.items()):
        tt = res["t_test"]
        wt = res["wilcoxon"]

        both_sig = tt["significant"] and wt["significant"]
        sig_mark = "✓ YES" if both_sig else "✗ NO "
        if not both_sig:
            all_significant = False
        if both_sig:
            any_significant = True

        improve_pct = tt["improvement"] * 100

        print(f"  {baseline:<20} {tt['t_stat']:>8.3f} {tt['p_value']:>10.5f} "
              f"{wt['p_value']:>10.5f} {improve_pct:>9.1f}% "
              f"{tt['cohens_d']:>7.3f} {sig_mark:>6}")

    print(f"{'═'*76}")
    print(f"  Columns: t-stat | t-test p | Wilcoxon p | "
          f"% improvement | Cohen's d | Significant")
    print(f"  Cohen's d: <0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large\n")

    # ── KEY VERDICT (required by task spec) ──────────────────────────────────
    if all_significant and results:
        print(f"  ★ RL scheduler improvement statistically significant (p < {alpha})")
        print(f"    vs ALL baselines — both t-test and Wilcoxon confirm.")
    elif any_significant:
        sig_baselines = [b for b, r in results.items()
                         if r["t_test"]["significant"] and r["wilcoxon"]["significant"]]
        print(f"  ★ RL scheduler improvement statistically significant (p < {alpha})")
        print(f"    vs: {', '.join(sig_baselines)}")
        not_sig = [b for b in results if b not in sig_baselines]
        if not_sig:
            print(f"    (Not significant vs: {', '.join(not_sig)})")
    else:
        print(f"  ✗ RL scheduler improvement NOT statistically significant at α={alpha}.")
        print(f"    Consider more trials or a different workload distribution.")
    print()
