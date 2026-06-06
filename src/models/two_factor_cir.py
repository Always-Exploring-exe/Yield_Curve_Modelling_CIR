"""
two_factor_cir.py — Longstaff-Schwartz (1992) Two-Factor CIR Model

This module implements the Two-Factor CIR model. 
Unlike the single-factor CIR which only tracks one state variable (meaning the whole curve moves together), 
the Two-Factor model decomposes the short rate into two independent square-root processes (level and slope).
This allows the model to naturally capture complex curve shapes like inversions.

For the full mathematical derivation and explanation of how the speed difference between factors 
generates inversions, please see the main `CIR_combined_Results.ipynb` notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from src.models.cir_math import CIRParams, cir_B, cir_ln_A, cir_yield
from src.data.preprocessing import CleanDataset, SHORT_RATE_COL


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class TwoFactorCIRParams:
    """
    Six parameters for the Longstaff-Schwartz two-factor model.

    Factor 1 (level factor):
        κ₁  — mean-reversion speed of r₁ (typically slow: κ₁ < κ₂)
        θ₁  — long-run level of r₁
        σ₁  — volatility of r₁

    Factor 2 (slope/spread factor):
        κ₂  — mean-reversion speed of r₂ (typically fast: κ₂ > κ₁)
        θ₂  — long-run level of r₂
        σ₂  — volatility of r₂

    alpha: fraction of total short rate attributed to factor 1.
           r₁ = alpha * r_t,  r₂ = (1-alpha) * r_t
           Estimated from training data cross-section.
    """
    kappa1: float
    theta1: float
    sigma1: float
    kappa2: float
    theta2: float
    sigma2: float
    alpha: float = 0.6   # r₁ share of total short rate (estimated during calibration)

    @property
    def params1(self) -> CIRParams:
        """Factor 1 as a single-factor CIRParams object."""
        return CIRParams(kappa=self.kappa1, theta=self.theta1,
                         sigma=self.sigma1, r0=0.0)

    @property
    def params2(self) -> CIRParams:
        """Factor 2 as a single-factor CIRParams object."""
        return CIRParams(kappa=self.kappa2, theta=self.theta2,
                         sigma=self.sigma2, r0=0.0)

    def feller1(self) -> float:
        """Feller condition for factor 1: 2κ₁θ₁ - σ₁² ≥ 0."""
        return 2 * self.kappa1 * self.theta1 - self.sigma1 ** 2

    def feller2(self) -> float:
        """Feller condition for factor 2: 2κ₂θ₂ - σ₂² ≥ 0."""
        return 2 * self.kappa2 * self.theta2 - self.sigma2 ** 2

    def __str__(self) -> str:
        f1 = "OK" if self.feller1() >= 0 else "VIOLATED"
        f2 = "OK" if self.feller2() >= 0 else "VIOLATED"
        return (
            f"TwoFactorCIRParams:\n"
            f"  Factor 1 (level):  k1={self.kappa1:.4f}  th1={self.theta1*100:.3f}%  "
            f"sig1={self.sigma1:.4f}  Feller: {self.feller1():.4f} [{f1}]\n"
            f"  Factor 2 (slope):  k2={self.kappa2:.4f}  th2={self.theta2*100:.3f}%  "
            f"sig2={self.sigma2:.4f}  Feller: {self.feller2():.4f} [{f2}]\n"
            f"  Factor allocation: alpha={self.alpha:.3f}  "
            f"(r1 = {self.alpha*100:.1f}% of r_t,  r2 = {(1-self.alpha)*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Core pricing formulas (build on single-factor building blocks)
# ---------------------------------------------------------------------------

def two_factor_yield(
    tau: np.ndarray,
    r1: float,
    r2: float,
    params: TwoFactorCIRParams,
) -> np.ndarray:
    """
    Two-factor CIR yield curve: y(τ) = [B₁(τ)r₁ + B₂(τ)r₂ - lnA₁(τ) - lnA₂(τ)] / τ

    Because the two factors are independent, the log bond price is additive:
        ln P(τ) = ln A₁(τ) - B₁(τ)r₁  +  ln A₂(τ) - B₂(τ)r₂

    This linearity is the key mathematical property that makes two-factor CIR
    tractable — it's simply two single-factor CIR models superimposed.

    Parameters
    ----------
    tau    : array of maturities in years, e.g. [0.25, 0.5, ..., 30]
    r1     : current value of factor 1 (level factor)
    r2     : current value of factor 2 (slope factor)
    params : TwoFactorCIRParams
    """
    tau = np.asarray(tau, dtype=float)
    # Each factor contributes additively to the log bond price
    ln_A1 = cir_ln_A(tau, params.params1)
    ln_A2 = cir_ln_A(tau, params.params2)
    B1    = cir_B(tau, params.params1)
    B2    = cir_B(tau, params.params2)

    # y(τ) = -ln(P)/τ = -(ln A₁ + ln A₂ - B₁·r₁ - B₂·r₂) / τ
    log_P = ln_A1 - B1 * r1 + ln_A2 - B2 * r2
    with np.errstate(divide="ignore", invalid="ignore"):
        yields = -log_P / tau
    return yields


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_two_factor(
    dataset: CleanDataset,
    n_starts: int = 5,
    verbose: bool = True,
    seed: int = 42,
) -> Tuple[TwoFactorCIRParams, dict]:
    """
    Calibrate the Longstaff-Schwartz two-factor CIR model on training data.

    Strategy
    --------
    We minimise the sum of squared errors between model-predicted yields and
    observed yields across all maturities and dates — a cross-sectional
    least-squares calibration. This is feasible because:

    1. The bond pricing formula is still closed-form (no simulation needed).
    2. We treat the factor decomposition as part of the optimisation:
       for each parameter set (κ₁,θ₁,σ₁,κ₂,θ₂,σ₂), we estimate the optimal
       α (factor allocation) via a simple 1D grid search.

    Identification Note
    -------------------
    Without observing r₁ and r₂ directly, we cannot uniquely identify all 6
    parameters from yield curve data alone (the model is under-identified).
    We impose a constraint: κ₁ < κ₂ (factor 1 is the slow/level factor,
    factor 2 is the fast/slope factor). This is economically motivated and
    standard in the literature.

    Parameters
    ----------
    dataset  : CleanDataset from training data
    n_starts : number of random restarts to avoid local minima
    verbose  : print optimisation progress
    seed     : random seed for reproducibility
    """
    yield_cols = [c for c in dataset.df.columns if c != "stress_event"
                  and c in ["3M", "6M", "9M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]]
    # Use maturity-ordered subset
    from src.models.prediction import YIELD_COLS, MATURITIES
    yield_cols = [c for c in YIELD_COLS if c in yield_cols]
    taus       = MATURITIES[:len(yield_cols)]

    Y = dataset.df[yield_cols].values   # shape (T, M)
    r_t = dataset.short_rate.values     # shape (T,)
    T, M = Y.shape

    rng = np.random.default_rng(seed)

    def model_yields_given_alpha(kappa1, theta1, sigma1, kappa2, theta2, sigma2, alpha):
        """Compute predicted yields for all T dates given alpha."""
        p = TwoFactorCIRParams(kappa1, theta1, sigma1, kappa2, theta2, sigma2, alpha)
        r1 = alpha * r_t
        r2 = (1 - alpha) * r_t
        # Ensure positivity
        r1 = np.maximum(r1, 1e-8)
        r2 = np.maximum(r2, 1e-8)
        preds = np.array([two_factor_yield(taus, r1[t], r2[t], p) for t in range(T)])
        return preds

    def objective(x):
        """Sum of squared errors across all dates × maturities."""
        kappa1, theta1, sigma1, kappa2, theta2, sigma2, alpha = x
        # Hard constraints
        if any(v <= 0 for v in [kappa1, theta1, sigma1, kappa2, theta2, sigma2]):
            return 1e15
        if kappa1 >= kappa2:   # enforce factor ordering
            return 1e15
        if not (0.01 <= alpha <= 0.99):
            return 1e15
        try:
            preds = model_yields_given_alpha(kappa1, theta1, sigma1,
                                             kappa2, theta2, sigma2, alpha)
            sse = np.sum((Y - preds) ** 2)
            return float(sse) if np.isfinite(sse) else 1e15
        except Exception:
            return 1e15

    # ── Multi-start optimisation ────────────────────────────────────────────
    # Parameter order: [kappa1, theta1, sigma1, kappa2, theta2, sigma2, alpha]
    bounds = [
        (1e-4, 1.0),   # kappa1: slow mean-reversion
        (1e-4, 0.15),  # theta1: long-run rate 0–15%
        (1e-4, 0.20),  # sigma1: volatility
        (1e-4, 5.0),   # kappa2: fast mean-reversion
        (1e-4, 0.15),  # theta2: long-run rate 0–15%
        (1e-4, 0.30),  # sigma2: volatility
        (0.01, 0.99),  # alpha: factor 1 share
    ]

    # Start with economically sensible initial guess
    x0_default = [0.08, 0.03, 0.06, 0.80, 0.02, 0.10, 0.60]

    best_result = None
    best_obj    = np.inf

    starts = [x0_default] + [
        [rng.uniform(lo, hi) for lo, hi in bounds]
        for _ in range(n_starts - 1)
    ]

    for i, x0 in enumerate(starts):
        # Enforce kappa1 < kappa2 in starting guess
        if x0[0] >= x0[3]:
            x0[3] = x0[0] * 5.0
        try:
            res = optimize.minimize(
                objective, x0=x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            if res.fun < best_obj:
                best_obj    = res.fun
                best_result = res
                if verbose:
                    print(f"  Restart {i+1}/{n_starts}: SSE={res.fun:.6f} "
                          f"{'(new best)' if i > 0 else ''}")
        except Exception as e:
            if verbose:
                print(f"  Restart {i+1} failed: {e}")

    kappa1, theta1, sigma1, kappa2, theta2, sigma2, alpha = best_result.x
    params = TwoFactorCIRParams(
        kappa1=float(kappa1), theta1=float(theta1), sigma1=float(sigma1),
        kappa2=float(kappa2), theta2=float(theta2), sigma2=float(sigma2),
        alpha=float(alpha),
    )

    # ── In-sample metrics ───────────────────────────────────────────────────
    preds_train = model_yields_given_alpha(kappa1, theta1, sigma1,
                                           kappa2, theta2, sigma2, alpha)
    ss_res = np.sum((Y - preds_train) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_insample = float(1 - ss_res / ss_tot)
    rmse_insample = float(np.sqrt(np.mean((Y - preds_train) ** 2)) * 10000)  # bps

    diagnostics = {
        "r2_insample":    r2_insample,
        "rmse_insample_bps": rmse_insample,
        "sse":            float(best_obj),
        "converged":      best_result.success,
        "feller1":        params.feller1(),
        "feller2":        params.feller2(),
    }

    if verbose:
        print(f"\n{str(params).encode('ascii', 'replace').decode()}")
        print(f"  In-sample R2:   {r2_insample:.4f}")
        print(f"  In-sample RMSE: {rmse_insample:.2f} bps")
        f1_ok = "OK" if params.feller1() >= 0 else "VIOLATED"
        f2_ok = "OK" if params.feller2() >= 0 else "VIOLATED"
        print(f"  Feller factor1: {params.feller1():.4f} [{f1_ok}]")
        print(f"  Feller factor2: {params.feller2():.4f} [{f2_ok}]")

    return params, diagnostics


# ---------------------------------------------------------------------------
# Prediction on test set
# ---------------------------------------------------------------------------

@dataclass
class TwoFactorPredictionResult:
    """Results from two-factor CIR prediction."""
    method: str
    dates: pd.DatetimeIndex
    predicted: np.ndarray    # shape (T, M)
    actual: np.ndarray       # shape (T, M)
    r_t_used: np.ndarray     # 3M rate used as input
    yield_cols: list
    r_squared: float
    rmse_by_maturity: dict


def predict_two_factor(
    test_dataset: CleanDataset,
    train_dataset: CleanDataset,
    params: TwoFactorCIRParams,
) -> TwoFactorPredictionResult:
    """
    Two-factor CIR prediction on the test set.

    On each test day, only the 3M yield is available. We allocate:
        r₁_t = params.alpha * r_t_3M
        r₂_t = (1 - params.alpha) * r_t_3M

    where alpha was estimated during calibration. This is the natural
    decomposition under the 3M-only constraint: we cannot observe the two
    factors separately, so we use their average training-period ratio.

    Important limitation of the constant-alpha approach
    ----------------------------------------------------
    Because r₁ = alpha * r_t and r₂ = (1-alpha) * r_t, both factors are
    always proportional to the same 3M rate at test time. They move in the
    same direction and in a fixed ratio -- there is no independent slope
    factor movement. The real advantage of the Longstaff-Schwartz model
    (that r₁ and r₂ can move independently) is not available when only
    the 3M rate is observed.

    So where does the performance gain come from?
    The predicted curve is:
        y(τ) = [alpha*B₁(τ)*r_t + (1-alpha)*B₂(τ)*r_t - lnA₁(τ) - lnA₂(τ)] / τ
    This is a different parametric curve family than single-factor CIR
    (different effective B shape and different A terms from 6 free parameters
    vs 3). The two-factor model fits a more flexible curve shape during
    calibration, which is what produces the better out-of-sample performance.
    It's not hollow -- the curve shape IS genuinely different and more
    flexible -- but it's not the full theoretical advantage of a 2-factor model.

    Why this still handles inversions better than single-factor
    -----------------------------------------------------------
    In single-factor CIR, the curve shape for a given r_t is entirely
    determined by three parameters. With two factors and six parameters,
    the combined effective B(τ) = alpha*B₁(τ) + (1-alpha)*B₂(τ) can
    produce a wider range of slope profiles, including ones that match
    the relatively flat or mildly inverted 2024 curve better than
    single-factor CIR's constrained shape.
    """
    from src.models.prediction import YIELD_COLS, MATURITIES

    # Use only the maturities present in test data
    yield_cols = [c for c in YIELD_COLS if c in test_dataset.df.columns
                  and c != "stress_event"]
    taus       = MATURITIES[:len(yield_cols)]

    actual    = test_dataset.df[yield_cols].values
    r_t_series = test_dataset.short_rate.values
    T         = len(r_t_series)

    # Decompose short rate into two factors using trained alpha
    r1 = np.maximum(params.alpha * r_t_series, 1e-8)
    r2 = np.maximum((1 - params.alpha) * r_t_series, 1e-8)

    predicted = np.array([
        two_factor_yield(taus, r1[t], r2[t], params)
        for t in range(T)
    ])

    # ── Compute metrics ─────────────────────────────────────────────────────
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2_oos = float(1 - ss_res / ss_tot)

    rmse_by_maturity = {}
    for j, col in enumerate(yield_cols):
        residuals = actual[:, j] - predicted[:, j]
        rmse_by_maturity[col] = float(np.sqrt(np.mean(residuals ** 2)))

    return TwoFactorPredictionResult(
        method="Two-Factor CIR (Longstaff-Schwartz 1992)",
        dates=test_dataset.df.index,
        predicted=predicted,
        actual=actual,
        r_t_used=r_t_series,
        yield_cols=yield_cols,
        r_squared=r2_oos,
        rmse_by_maturity=rmse_by_maturity,
    )


# ---------------------------------------------------------------------------
# Qualitative analysis helpers
# ---------------------------------------------------------------------------

def compare_curve_shapes(
    params_1f: "CIRParams",
    params_2f: TwoFactorCIRParams,
    r_t_values: list,
    taus: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare yield curve shapes from single-factor vs two-factor model
    at different short-rate levels.

    Single-factor CIR shape depends on where r_t sits relative to the model's
    long-run limit. When r_t < θ the curve is typically upward-sloping (rates
    expected to rise). When r_t > θ the curve can slope downward, but the
    term premium partially offsets this, so the shape is a balance between the
    two forces rather than a clean inversion.

    Two-factor CIR has more freedom: because the two factors mean-revert at
    different speeds (κ₁ < κ₂ by construction), they contribute differently
    to short vs long maturities. When factor 2 (the fast one) is large, it
    lifts short yields more than long yields, producing an inverted shape even
    when the total short rate r_t = r₁ + r₂ is not particularly extreme.
    This is the key qualitative difference the two-factor model adds.
    """
    if taus is None:
        taus = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])

    records = []
    for r_t in r_t_values:
        # Single factor
        y1f = cir_yield(taus, r_t, params_1f)
        # Two factor
        r1 = params_2f.alpha * r_t
        r2 = (1 - params_2f.alpha) * r_t
        y2f = two_factor_yield(taus, max(r1, 1e-8), max(r2, 1e-8), params_2f)
        for i, tau in enumerate(taus):
            records.append({
                "r_t": f"{r_t*100:.1f}%",
                "tau": tau,
                "y_1factor": y1f[i] * 100,
                "y_2factor": y2f[i] * 100,
            })
    return pd.DataFrame(records)


def two_factor_metrics(result: TwoFactorPredictionResult) -> pd.DataFrame:
    """Per-maturity metrics for the two-factor prediction result."""
    records = []
    for j, col in enumerate(result.yield_cols):
        residuals = result.actual[:, j] - result.predicted[:, j]
        records.append({
            "maturity":  col,
            "bias_bps":  float(np.mean(residuals) * 10000),
            "rmse_bps":  float(np.sqrt(np.mean(residuals**2)) * 10000),
            "mae_bps":   float(np.mean(np.abs(residuals)) * 10000),
            "r2":        float(1 - np.sum(residuals**2) /
                               np.sum((result.actual[:, j] -
                                       result.actual[:, j].mean())**2)),
        })
    return pd.DataFrame(records).set_index("maturity").round(4)
