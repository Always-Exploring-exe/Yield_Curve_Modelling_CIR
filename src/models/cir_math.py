"""
cir_math.py — Pure CIR mathematics. No data, no optimisation, just formulas.

The CIR model says the short rate r_t evolves as:
    dr_t = κ(θ - r_t)dt  +  σ√r_t dW_t

The closed-form bond price (derived by solving a PDE) is:
    P(t,T) = A(τ) × exp(-B(τ) × r_t)    where τ = T - t

And the yield at maturity τ is:
    y(τ) = [B(τ)×r_t  -  ln(A(τ))] / τ

This file computes A, B, yields, bond prices, and simulates paths.
Everything takes numpy arrays in and numpy arrays out — easy to test.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np


class CIRParams(NamedTuple):
    """
    The three numbers that define a CIR model, plus an optional starting rate.

    kappa : how fast rates snap back to theta. Half-life = ln(2)/kappa years.
    theta : the long-run average rate (e.g. 0.03 = 3%).
    sigma : how noisy the rate process is.
    r0    : starting rate, used only for simulation.
    """
    kappa: float
    theta: float
    sigma: float
    r0: float = 0.0

    def feller_condition(self) -> float:
        """
        2κθ - σ². Must be ≥ 0 for rates to stay strictly positive.

        Why: the noise term σ√r_t shrinks to zero as r_t→0, so the upward
        drift κ(θ-r_t) can push rates back up before they hit zero — but
        only if 2κθ ≥ σ². Otherwise the noise wins near zero.
        """
        return 2 * self.kappa * self.theta - self.sigma ** 2

    def feller_satisfied(self) -> bool:
        return self.feller_condition() >= 0

    def half_life_years(self) -> float:
        """Time (years) for a rate shock to decay to half its original size."""
        return np.log(2) / self.kappa

    def __str__(self) -> str:
        feller = self.feller_condition()
        status = "✓ SATISFIED" if feller >= 0 else "✗ VIOLATED"
        return (
            f"CIRParams(κ={self.kappa:.6f}, θ={self.theta*100:.3f}%, σ={self.sigma:.6f})\n"
            f"  Feller 2κθ-σ² = {feller:.6f} → {status}\n"
            f"  Half-life = {self.half_life_years():.2f} years"
        )


def cir_gamma(params: CIRParams) -> float:
    """
    γ = √(κ² + 2σ²). An intermediate constant used in every A/B formula.

    Appears because the CIR PDE solution involves combining mean-reversion
    (κ) and volatility (σ) into a single effective rate. Think of it as
    the 'adjusted speed' in the risk-neutral world.
    """
    return np.sqrt(params.kappa ** 2 + 2 * params.sigma ** 2)


def cir_B(tau: np.ndarray, params: CIRParams) -> np.ndarray:
    """
    B(τ) — the sensitivity of bond log-price to the short rate r_t.

    Formula:  B(τ) = 2(e^γτ - 1) / [(γ+κ)(e^γτ - 1) + 2γ]

    Intuition:
      - B(0) = 0: a bond maturing right now doesn't depend on today's rate.
      - B(∞) = 2/(γ+κ): very long bonds have bounded, finite rate sensitivity.
      - B increases with τ but at a decreasing rate — long bonds are less
        sensitive per unit of extra maturity than short bonds.

    B appears in the yield formula as: y(τ) = [B(τ)×r_t - ln(A(τ))] / τ
    The ∂y/∂r_t = B(τ)/τ — this is also the Jacobian used in the EKF.
    """
    tau = np.asarray(tau, dtype=float)
    gamma = cir_gamma(params)
    exp_gt = np.exp(gamma * tau)

    numerator   = 2 * (exp_gt - 1)
    denominator = (gamma + params.kappa) * (exp_gt - 1) + 2 * gamma

    # For very large tau, exp_gt overflows. The limit of the ratio is 2/(γ+κ).
    limit = 2 / (gamma + params.kappa)
    with np.errstate(over="ignore", invalid="ignore"):
        B = np.where(
            np.isfinite(denominator) & (denominator > 0),
            numerator / denominator,
            limit,
        )
    return B


def cir_ln_A(tau: np.ndarray, params: CIRParams) -> np.ndarray:
    """
    ln(A(τ)) — the log of the 'level' factor in the bond price formula.

    Raw formula:  A(τ) = [2γ·exp((κ+γ)τ/2) / ((γ+κ)(e^γτ-1) + 2γ)]^(2κθ/σ²)

    Why log? The exponent 2κθ/σ² is typically 10–50. Raising a small fraction
    to that power gives underflow on a computer. Working in log space is safe
    and we need ln(A) anyway since the yield formula uses ln(P) = ln(A) - B·r_t.
    """
    tau = np.asarray(tau, dtype=float)
    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    gamma = cir_gamma(params)
    exp_gt = np.exp(gamma * tau)
    power  = 2 * kappa * theta / (sigma ** 2)  # the exponent

    # log of the numerator inside the bracket: log(2γ) + (κ+γ)τ/2
    ln_num = np.log(2 * gamma) + (kappa + gamma) * tau / 2
    # log of the denominator: log((γ+κ)(e^γτ-1) + 2γ)
    denom = (gamma + kappa) * (exp_gt - 1) + 2 * gamma

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_A = power * (ln_num - np.log(denom))
    return ln_A


def cir_yield(tau: np.ndarray, r_t: float, params: CIRParams) -> np.ndarray:
    """
    The core prediction formula: given today's short rate r_t, return yields
    at all maturities in tau.

        y(τ) = [B(τ)×r_t  -  ln(A(τ))] / τ

    This is the entire CIR yield curve in one line. The beauty of the model:
    three calibrated numbers (κ, θ, σ) and today's short rate give you
    the complete yield curve for any maturity you care about.
    """
    tau  = np.asarray(tau, dtype=float)
    B    = cir_B(tau, params)
    ln_A = cir_ln_A(tau, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        yields = (B * r_t - ln_A) / tau
    return yields


def cir_bond_price(tau: np.ndarray, r_t: float, params: CIRParams) -> np.ndarray:
    """
    Zero-coupon bond prices P = A(τ)·exp(-B(τ)·r_t).
    A price of 0.95 means you pay 95 today to receive 100 at maturity.
    All prices should be in (0, 1).
    """
    tau  = np.asarray(tau, dtype=float)
    B    = cir_B(tau, params)
    ln_A = cir_ln_A(tau, params)
    return np.exp(ln_A - B * r_t)


def cir_simulate(
    params: CIRParams,
    n_steps: int,
    dt: float,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate CIR rate paths using the Milstein scheme.

    Why Milstein instead of the simpler Euler method?
    Euler:    r_{t+1} = r_t + κ(θ-r_t)dt + σ√r_t·ε·√dt
    This can produce negative r values when r_t is small and ε is large negative.

    Milstein adds one correction term (derivative of σ√r_t w.r.t. r_t = σ/(2√r_t)):
              + (σ²/4)(ε²-1)dt
    This nudges the path away from zero and reduces negative-value occurrences.

    Returns array of shape (n_steps+1, n_paths). First row is r0.
    """
    rng = np.random.default_rng(seed)
    kappa, theta, sigma, r0 = params.kappa, params.theta, params.sigma, params.r0

    if not params.feller_satisfied():
        warnings.warn(
            f"Feller condition violated ({params.feller_condition():.4f}). "
            "Paths may hit zero — applying a small floor.",
            RuntimeWarning,
        )

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = r0
    floor = 1e-8  # tiny positive floor so sqrt never sees a negative number

    for i in range(n_steps):
        r   = np.maximum(paths[i], floor)
        eps = rng.standard_normal(n_paths)

        drift      = kappa * (theta - r) * dt
        diffusion  = sigma * np.sqrt(r * dt) * eps
        correction = 0.25 * sigma**2 * dt * (eps**2 - 1)  # Milstein term

        paths[i + 1] = np.maximum(r + drift + diffusion + correction, floor)

    return paths


def check_feller(params: CIRParams, verbose: bool = True) -> dict:
    """Print and return a Feller condition diagnostic."""
    val       = params.feller_condition()
    satisfied = val >= 0
    margin    = val / (params.sigma ** 2)  # margin expressed as fraction of σ²

    if verbose:
        status = "✓ SATISFIED" if satisfied else "✗ VIOLATED"
        print(f"Feller: 2κθ - σ² = {val:.6f}  →  {status}")
        print(f"  2κθ = {2*params.kappa*params.theta:.6f}")
        print(f"  σ²  = {params.sigma**2:.6f}")
        if satisfied:
            print(f"  Comfortable margin: {margin*100:.1f}% of σ²")
        else:
            print(f"  Shortfall: {abs(margin)*100:.1f}% of σ² — rates can hit zero")

    return {"feller_value": val, "satisfied": satisfied,
            "margin_fraction": margin,
            "kappa": params.kappa, "theta": params.theta, "sigma": params.sigma}
