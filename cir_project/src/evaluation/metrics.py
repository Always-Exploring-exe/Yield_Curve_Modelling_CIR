"""
Evaluation Metrics
==================
Extended diagnostics beyond R² and RMSE.
Covers regime-level breakdown, Feller tracking, and model comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.prediction import PredictionResult, YIELD_COLS
from src.models.cir_math import CIRParams, check_feller
from src.data.preprocessing import CleanDataset


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(result: PredictionResult) -> pd.DataFrame:
    """
    Per-maturity metrics: RMSE, MAE, R², bias (mean error).

    Bias (mean signed error) shows systematic over/under-prediction.
    A positive bias means the model predicts too high on average.
    """
    yield_cols = list(result.rmse_by_maturity.keys())
    records = []
    for j, col in enumerate(yield_cols):
        residuals = result.actual[:, j] - result.predicted[:, j]
        records.append({
            "maturity": col,
            "bias_bps":  float(np.mean(residuals) * 10000),
            "rmse_bps":  float(np.sqrt(np.mean(residuals**2)) * 10000),
            "mae_bps":   float(np.mean(np.abs(residuals)) * 10000),
            "r2":        float(1 - np.sum(residuals**2) /
                               np.sum((result.actual[:, j] - result.actual[:, j].mean())**2)),
        })
    return pd.DataFrame(records).set_index("maturity").round(4)


def compare_methods(results: List[PredictionResult]) -> pd.DataFrame:
    """
    Side-by-side comparison of prediction methods.
    """
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "R²": round(r.r_squared, 4),
            "pass_0.85": "✓" if r.r_squared >= 0.85 else "✗",
            "mean_rmse_bps": round(
                np.mean(list(r.rmse_by_maturity.values())) * 10000, 2
            ),
            "max_rmse_bps": round(
                max(r.rmse_by_maturity.values()) * 10000, 2
            ),
            "worst_maturity": max(r.rmse_by_maturity, key=r.rmse_by_maturity.get),
        })
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------------------
# Regime-level breakdown
# ---------------------------------------------------------------------------

def regime_r2(
    result: PredictionResult,
    test_dataset: CleanDataset,
) -> pd.DataFrame:
    """
    Break R² down by rate regime (low / rising / high).

    This answers: "where does the model fail most?"

    Regimes defined by the 3M rate level on test days:
      low    : 3M < 2%        (near-zero rate environment)
      rising : 2% <= 3M < 4%  (tightening cycle)
      high   : 3M >= 4%       (restrictive policy)
    """
    r_t = result.r_t_used
    actual = result.actual
    predicted = result.predicted

    regimes = {
        "low (<2%)":      r_t < 0.02,
        "rising (2-4%)":  (r_t >= 0.02) & (r_t < 0.04),
        "high (>4%)":     r_t >= 0.04,
    }

    records = []
    for name, mask in regimes.items():
        if mask.sum() == 0:
            continue
        act_sub = actual[mask]
        pred_sub = predicted[mask]
        ss_res = np.sum((act_sub - pred_sub) ** 2)
        ss_tot = np.sum((act_sub - act_sub.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean((act_sub - pred_sub)**2)) * 10000)
        records.append({
            "regime": name, "n_days": int(mask.sum()),
            "R²": round(r2, 4), "mean_rmse_bps": round(rmse, 2),
        })
    return pd.DataFrame(records).set_index("regime")


def stress_vs_calm_r2(
    result: PredictionResult,
    test_dataset: CleanDataset,
) -> pd.DataFrame:
    """
    Compare R² on stress event days vs normal days.
    """
    stress = test_dataset.df["stress_event"].values
    records = []
    for label, mask in [("stress days", stress), ("calm days", ~stress)]:
        if mask.sum() == 0:
            continue
        act = result.actual[mask]
        pred = result.predicted[mask]
        ss_res = np.sum((act - pred)**2)
        ss_tot = np.sum((act - act.mean())**2)
        r2 = float(1 - ss_res / ss_tot)
        rmse = float(np.sqrt(np.mean((act - pred)**2)) * 10000)
        records.append({
            "subset": label, "n_days": int(mask.sum()),
            "R²": round(r2, 4), "mean_rmse_bps": round(rmse, 2),
        })
    return pd.DataFrame(records).set_index("subset")


# ---------------------------------------------------------------------------
# Feller condition over rolling windows
# ---------------------------------------------------------------------------

def rolling_feller(
    train_dataset: CleanDataset,
    params: CIRParams,
    window: int = 252,
) -> pd.DataFrame:
    """
    Track Feller condition value over rolling calibration windows.

    We don't re-calibrate (too expensive), but we can show how the
    Feller margin would fluctuate if parameters drifted with the data.
    We approximate by computing 2κθ - σ² with the fixed params but
    overlay the changing volatility of the short rate (σ_rolling) to
    show when the condition would be under stress.

    The key insight: σ is calibrated to the training period average.
    During high-volatility periods (2022 hike cycle), actual short-rate
    volatility was much higher, which would violate Feller if re-calibrated.
    """
    r = train_dataset.short_rate.values
    dates = train_dataset.df.index

    # Rolling empirical short-rate volatility (annualized)
    dt = 1 / 252
    returns = np.diff(r) / np.maximum(np.sqrt(r[:-1]), 1e-8)
    rolling_vol = pd.Series(returns).rolling(window).std().values / np.sqrt(dt)
    rolling_vol = np.concatenate([[np.nan], rolling_vol])  # align with dates

    # Rolling Feller margin using empirical σ instead of calibrated σ
    feller_margin = 2 * params.kappa * params.theta - rolling_vol**2

    df = pd.DataFrame({
        "date": dates,
        "r_t": r,
        "rolling_sigma": rolling_vol,
        "feller_margin_empirical": feller_margin,
        "feller_calibrated": params.feller_condition(),
    }).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Half-life interpretation
# ---------------------------------------------------------------------------

def half_life_analysis(params: CIRParams) -> str:
    """
    Interpret the mean-reversion speed in economic terms.
    """
    hl = params.half_life_years()
    lines = [
        f"Mean-reversion speed κ = {params.kappa:.4f}",
        f"  → Half-life = ln(2)/κ = {hl:.2f} years ({hl*12:.1f} months)",
        "",
    ]
    if hl < 0.5:
        lines.append("  Interpretation: Very fast reversion (<6mo). Rate shocks are transitory.")
        lines.append("  This could reflect: central bank policy, short-end rates near target.")
    elif hl < 2:
        lines.append("  Interpretation: Moderate reversion (6mo-2yr). Typical for short-rate models.")
        lines.append("  This is the most common empirical finding in CIR studies.")
    elif hl < 5:
        lines.append("  Interpretation: Slow reversion (2-5yr). Rate deviations are persistent.")
        lines.append("  This could reflect: structural rate regimes, secular stagnation.")
    else:
        lines.append(f"  Interpretation: Very slow reversion ({hl:.0f}yr). Near unit root.")
        lines.append("  WARNING: This makes CIR nearly indistinguishable from a random walk.")
        lines.append("  The model may be poorly identified — check calibration.")
    lines.append(f"  Long-run mean θ = {params.theta*100:.3f}%")
    return "\n".join(lines)
