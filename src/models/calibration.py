"""
calibration.py — Find the CIR parameters (κ, θ, σ) that best fit the data.

Three methods are implemented here:
  1. OLS -- Fast but fails badly on this dataset due to near-unit-root behaviour.
  2. MLE -- Uses exact CIR transition density, but still struggles with identification.
  3. EKF -- The optimal approach. Treats r_t as a hidden variable and calibrates
            across the entire yield curve (all 9 maturities) simultaneously.

For a detailed explanation of why EKF succeeds where OLS and MLE fail,
please see the main `CIR_combined_Results.ipynb` notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import warnings
from scipy import optimize, stats

from src.models.cir_math import CIRParams, cir_B, cir_yield, check_feller
from src.data.preprocessing import CleanDataset


# Column names and maturity array used throughout
YIELD_COLS = ["3M", "6M", "9M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]
MATURITIES = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

# Physical bounds on parameters — anything outside these makes no economic sense.
# κ > 20 would mean a half-life under 2 weeks. θ > 20% would be extreme.
PARAM_BOUNDS = [(1e-4, 20.0), (1e-4, 0.20), (1e-4, 2.0)]


@dataclass
class CalibrationResult:
    """Everything produced by a calibration run."""
    method: str
    params: CIRParams
    feller_satisfied: bool
    feller_value: float
    log_likelihood: Optional[float]
    r_squared_insample: float
    rmse_by_maturity: Dict[str, float]
    convergence_message: str
    n_observations: int
    filtered_states: Optional[np.ndarray] = None  # EKF only: the estimated r_t series

    def __str__(self) -> str:
        lines = [
            f"{'='*60}",
            f"  CALIBRATION RESULT [{self.method}]",
            f"{'='*60}",
            f"  κ (mean-reversion speed) : {self.params.kappa:.6f}",
            f"  θ (long-run mean)        : {self.params.theta*100:.3f}%",
            f"  σ (volatility)           : {self.params.sigma:.6f}",
            f"  Feller 2κθ-σ²            : {self.feller_value:.6f} → {'✓' if self.feller_satisfied else '✗'}",
            f"  Half-life                : {self.params.half_life_years():.2f} years",
            f"  In-sample R²             : {self.r_squared_insample:.4f}",
            f"  Convergence              : {self.convergence_message}",
            f"{'='*60}",
            "  RMSE by maturity (basis points):",
        ]
        for mat, rmse in self.rmse_by_maturity.items():
            lines.append(f"    {mat:5s}: {rmse*10000:.2f} bps")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_insample_metrics(
    params: CIRParams,
    r_t_series: np.ndarray,
    observed_yields: np.ndarray,
    maturities: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute R² and per-maturity RMSE on training data.
    r_t_series is either the raw 3M yields (OLS/MLE) or EKF-filtered states.
    """
    T, M = observed_yields.shape
    predicted = np.array([cir_yield(maturities, r, params) for r in r_t_series])
    residuals = observed_yields - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((observed_yields - observed_yields.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    rmse = {col: float(np.sqrt(np.mean(residuals[:, j]**2))) for j, col in enumerate(YIELD_COLS)}
    return r2, rmse


def _project_to_feller(params: CIRParams) -> CIRParams:
    """
    If the Feller condition is violated, reduce σ to the maximum value that
    just barely satisfies it: σ_max = √(2κθ).
    We only change σ — κ and θ have clearer economic meaning and we preserve them.
    """
    if params.feller_satisfied():
        return params
    sigma_max = np.sqrt(2 * params.kappa * params.theta)
    warnings.warn(
        f"Feller violated. Reducing σ from {params.sigma:.4f} → {sigma_max:.4f}.",
        RuntimeWarning,
    )
    return CIRParams(params.kappa, params.theta, sigma_max, params.r0)


# ===========================================================================
# METHOD 1: OLS
# ===========================================================================

def calibrate_ols(dataset: CleanDataset, enforce_feller: bool = True) -> CalibrationResult:
    """
    Calibrate CIR by turning the SDE into a linear regression.

    The CIR SDE discretised over a small dt:
        Δr_t  ≈  κθ·dt  -  κ·r_t·dt  +  σ√r_t·ε·√dt

    Divide everything by √r_t to equalise variance across observations
    (the CIR noise has variance proportional to r_t, not constant):
        Δr_t/√r_t  =  κθ·(dt/√r_t)  -  κ·(r_t·dt/√r_t)  +  σ·ε·√dt

    Now it's a plain linear regression: target = a·x1 + b·x2 + noise
        target = Δr_t/√r_t
        x1     = dt/√r_t        → coefficient a = κθ
        x2     = r_t·dt/√r_t   → coefficient b = -κ

    From a and b we recover κ = b and θ = a/b.
    σ is estimated from the residual variance.

    WHY THIS FAILS ON THIS DATASET:
    The 3M rate sat near zero for 5 years (2016-2021) then spiked to 4.3%.
    OLS cannot tell the difference between "slow mean-reversion toward a new
    level" and "no mean-reversion at all." It returns κ≈0, θ≈878% — nonsense.
    We include OLS as the starting point for MLE/EKF, not as a final answer.
    """
    r  = dataset.short_rate.values
    dt = dataset.dt_series.values

    # We need pairs (r_t, r_{t+1}) so drop the first row which has no predecessor
    r_t    = r[:-1]
    r_next = r[1:]
    dt_t   = dt[1:]
    delta_r = r_next - r_t

    # Weighted OLS: divide by sqrt(r_t) to account for heteroskedastic noise
    sqrt_r = np.sqrt(np.maximum(r_t, 1e-8))
    y  = delta_r / sqrt_r
    x1 = dt_t / sqrt_r
    x2 = -r_t * dt_t / sqrt_r

    coeffs, *_ = np.linalg.lstsq(np.column_stack([x1, x2]), y, rcond=None)
    kappa_theta, kappa = coeffs[0], coeffs[1]

    # Guard against degenerate / negative solutions
    kappa = max(kappa, 1e-4)
    theta = max(kappa_theta / kappa, 1e-4)

    # Estimate σ from how large the residuals are relative to r_t
    predicted_y = x1 * kappa_theta + x2 * (-kappa)
    resid = y - predicted_y
    sigma = max(float(np.std(resid) / np.sqrt(dt_t.mean())), 1e-4)

    params = CIRParams(kappa=kappa, theta=theta, sigma=sigma, r0=r[0])
    if enforce_feller:
        params = _project_to_feller(params)

    obs_yields = dataset.df[[c for c in YIELD_COLS if c in dataset.df.columns]].values
    r2, rmse = _compute_insample_metrics(params, r, obs_yields, MATURITIES)

    return CalibrationResult(
        method="OLS", params=params,
        feller_satisfied=params.feller_satisfied(),
        feller_value=params.feller_condition(),
        log_likelihood=None, r_squared_insample=r2, rmse_by_maturity=rmse,
        convergence_message="Closed-form (no iteration)",
        n_observations=len(r_t),
    )


# ===========================================================================
# METHOD 2: MLE
# ===========================================================================

def calibrate_mle(
    dataset: CleanDataset,
    init_params: Optional[CIRParams] = None,
    enforce_feller: bool = True,
) -> CalibrationResult:
    """
    Calibrate CIR by maximum likelihood using the exact CIR transition density.

    OLS assumed the noise in the SDE is Gaussian. It's not — it's noncentral
    chi-squared. Here's where that comes from:

    The CIR SDE can be written so that r_{t+dt} (scaled by a constant c) follows
    a noncentral chi-squared distribution χ²(d, λ) where:
        c = 4κ / (σ²(1 - e^{-κdt}))     ← scaling factor
        d = 4κθ / σ²                      ← degrees of freedom (> 0 iff Feller holds)
        λ = c · r_t · e^{-κdt}           ← noncentrality (shifts the distribution away from 0)

    We maximise the sum of log-likelihoods of observing each r_{t+1} given r_t
    under this distribution. scipy.stats.ncx2.logpdf does the heavy lifting.

    Still limited: treats the 3M yield as a perfect proxy for r_t (it isn't).
    """
    r  = dataset.short_rate.values
    dt = dataset.dt_series.values
    r_t, r_next, dt_t = r[:-1], r[1:], dt[1:]

    # Start from OLS parameters if none provided
    if init_params is None:
        init_params = calibrate_ols(dataset, enforce_feller=False).params
    p0 = [init_params.kappa, init_params.theta, init_params.sigma]

    def neg_log_likelihood(params_vec):
        kappa, theta, sigma = params_vec
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10

        # Build the noncentral chi-squared parameters for each daily transition
        c   = np.clip(4*kappa / (sigma**2 * (1 - np.exp(-kappa*dt_t))), 1e-10, None)
        d   = 4 * kappa * theta / sigma**2
        lam = np.clip(c * r_t * np.exp(-kappa*dt_t), 1e-10, None)
        u   = np.clip(c * r_next, 1e-10, None)

        if d <= 0:
            return 1e10

        try:
            # logpdf gives log-probability of observing u ~ χ²(d, λ)
            # plus log(c) for the Jacobian of the change of variables r→u=c·r
            log_lik = np.sum(stats.ncx2.logpdf(u, df=d, nc=lam) + np.log(c))
            return -log_lik if np.isfinite(log_lik) else 1e10
        except Exception:
            return 1e10

    result = optimize.minimize(
        neg_log_likelihood, x0=p0, method="L-BFGS-B",
        bounds=PARAM_BOUNDS, options={"maxiter": 2000, "ftol": 1e-12},
    )

    kappa, theta, sigma = result.x
    params = CIRParams(kappa=float(kappa), theta=float(theta), sigma=float(sigma), r0=r[0])
    if enforce_feller:
        params = _project_to_feller(params)

    obs_yields = dataset.df[[c for c in YIELD_COLS if c in dataset.df.columns]].values
    r2, rmse = _compute_insample_metrics(params, r, obs_yields, MATURITIES)
    log_lik  = -float(result.fun) if result.success else None
    msg      = "Converged" if result.success else f"WARNING: {result.message}"

    return CalibrationResult(
        method="MLE", params=params,
        feller_satisfied=params.feller_satisfied(),
        feller_value=params.feller_condition(),
        log_likelihood=log_lik, r_squared_insample=r2, rmse_by_maturity=rmse,
        convergence_message=msg, n_observations=len(r_t),
    )


# ===========================================================================
# METHOD 3: EXTENDED KALMAN FILTER (EKF)
# ===========================================================================

def calibrate_ekf(
    dataset: CleanDataset,
    init_params: Optional[CIRParams] = None,
    obs_noise_std: float = 0.001,
    enforce_feller: bool = True,
) -> CalibrationResult:
    """
    Calibrate CIR by treating r_t as a hidden variable and using all 9 yields.

    The core idea -- why this is better than OLS and MLE:
      You never observe r_t directly. Every yield you see is r_t viewed through
      a noisy lens. The 3M yield is one lens, the 30Y yield is another. Each
      lens has a different sensitivity to r_t, given by B(τ)/τ from the
      CIR yield formula.

      The Kalman filter fuses all 9 of these lenses optimally at each time step,
      giving a much better estimate of the true r_t than just using the 3M rate
      alone. On top of that, calibrating against 9 × 1976 = 17,784 data points
      instead of just 1,975 daily 3M transitions gives the optimiser a much
      stronger signal to work with -- which is why EKF recovers κ = 0.136
      (half-life ~5.1 years) and θ = 2.5%, while OLS gets κ ≈ 0.0001 and
      θ ≈ 1447% (nonsense).

    How the EKF works -- two steps every trading day:

    PREDICT: Using yesterday's best r_t estimate and CIR dynamics, predict
             where r_t is today before seeing any yields.
        r̂_pred = r̂_prev · e^{-κdt}  +  θ · (1 - e^{-κdt})
        P_pred  = e^{-2κdt} · P_prev  +  Q   (Q = CIR process variance over dt)

    UPDATE: See today's 9 yields. Compute how far off our predictions were
            (the 'innovation'). Adjust r̂ using the exact batch Kalman gain.

        H = B(τ)/τ for each maturity      <- how sensitive yield is to r_t
        S = H² · P_pred + R               <- innovation variance per maturity
        Log-likelihood contribution: -0.5 * sum(log(2πS) + innovation²/S)

        Batch update (information form -- exact for M independent observations):
        P_new   = 1 / (1/P_pred + sum(H_j²/R))   <- always positive, no approx
        K_j     = P_new · H_j / R                  <- optimal gain per maturity
        r̂      = r̂_pred + K · (actual - predicted yields)

        Note: the information form P update is the exact posterior variance
        for M independent Gaussian observations. The simpler P = (1-K@H)*P_pred
        can go negative with many maturities and was replaced.

    The outer optimisation finds κ, θ, σ, R by maximising the total
    log-likelihood of the innovations across all 1976 training days.
    """
    yield_cols = [c for c in YIELD_COLS if c in dataset.df.columns]
    Y   = dataset.df[yield_cols].values   # shape (T, 9)
    dt  = dataset.dt_series.values        # shape (T,)
    T, M = Y.shape
    taus = MATURITIES[:M]

    if init_params is None:
        init_params = calibrate_ols(dataset, enforce_feller=False).params

    def run_ekf(kappa, theta, sigma, obs_var):
        """
        One full forward pass of the EKF.
        Returns (total log-likelihood, array of filtered r_t estimates).
        """
        r_hat = Y[0, 0]   # initialise with first 3M observation
        P     = 0.01      # initial uncertainty about r_t (fairly uncertain)
        R     = obs_var   # observation noise variance (same for all maturities)

        total_ll = 0.0
        filtered = np.zeros(T)
        filtered[0] = r_hat

        for t in range(1, T):
            dt_t = dt[t]

            # ── PREDICT ───────────────────────────────────────────────────
            e_kdt  = np.exp(-kappa * dt_t)
            r_pred = max(r_hat * e_kdt + theta * (1 - e_kdt), 1e-8)

            # ── CIR CONDITIONAL VARIANCE ───────────────────────────────────
            # Exact formula from Brigo & Mercurio (2001) for Var[r_{t+dt} | r_t]:
            #   Var = r_t * σ² * e^{-κdt} * (1 - e^{-κdt}) / κ
            #       + θ * σ² * (1 - e^{-κdt})² / (2κ)
            one_minus_ekdt = 1 - e_kdt
            Q = max(
                r_hat * sigma**2 * e_kdt * one_minus_ekdt / kappa
                + theta * sigma**2 * one_minus_ekdt**2 / (2 * kappa),
                1e-12
            )
            P_pred = e_kdt**2 * P + Q

            # ── WHAT YIELDS WOULD WE EXPECT given r_pred? ─────────────────
            y_pred = cir_yield(taus, r_pred, CIRParams(kappa, theta, sigma))

            # ── JACOBIAN: how sensitive is each yield to r_t? ─────────────
            # dy(τ)/dr_t = B(τ)/τ  (the linearisation step in the EKF)
            H = cir_B(taus, CIRParams(kappa, theta, sigma)) / taus

            # ── INNOVATION: gap between actual and predicted yields ────────
            innovation = Y[t] - y_pred

            # ── INNOVATION COVARIANCE and LOG-LIKELIHOOD ───────────────────
            S = np.maximum(H**2 * P_pred + R, 1e-12)
            total_ll += -0.5 * np.sum(np.log(2*np.pi*S) + innovation**2 / S)

            # ── KALMAN GAIN and UPDATE ─────────────────────────────────────
            # For scalar state + M independent observations, the exact batch
            # update uses the information form:
            #   P_new = 1 / (1/P_pred + sum(H_j^2 / R))   <- always positive
            #   K_j   = P_new * H_j / R                    <- optimal batch gain
            # This avoids the (1 - K@H) approximation that can go negative.
            info_update = np.sum(H**2) / R
            P     = max(1.0 / (1.0 / P_pred + info_update), 1e-12)
            K     = P * H / R   # optimal batch Kalman gain per maturity
            r_hat = max(r_pred + float(K @ innovation), 1e-8)
            filtered[t] = r_hat

        return total_ll, filtered

    def neg_ll(params_vec):
        kappa, theta, sigma, log_obs_var = params_vec
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10
        try:
            ll, _ = run_ekf(kappa, theta, sigma, np.exp(log_obs_var))
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    # Use OLS params as starting point; log-transform obs_var so it stays positive
    p0 = [init_params.kappa, init_params.theta, init_params.sigma,
          np.log(obs_noise_std**2)]
    bounds = PARAM_BOUNDS + [(-20.0, -4.0)]  # last entry: log obs variance bounds

    result = optimize.minimize(
        neg_ll, x0=p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
    )

    kappa, theta, sigma, log_obs_var = result.x
    params = CIRParams(kappa=float(kappa), theta=float(theta),
                       sigma=float(sigma), r0=float(Y[0, 0]))
    if enforce_feller:
        params = _project_to_feller(params)

    # Final EKF pass with optimal params to get the filtered r_t series
    log_lik, filtered_states = run_ekf(params.kappa, params.theta, params.sigma,
                                       float(np.exp(log_obs_var)))
    r2, rmse = _compute_insample_metrics(params, filtered_states, Y, taus)
    msg = "Converged" if result.success else f"WARNING: {result.message}"

    return CalibrationResult(
        method="EKF", params=params,
        feller_satisfied=params.feller_satisfied(),
        feller_value=params.feller_condition(),
        log_likelihood=float(log_lik), r_squared_insample=r2,
        rmse_by_maturity=rmse, convergence_message=msg,
        n_observations=T, filtered_states=filtered_states,
    )
