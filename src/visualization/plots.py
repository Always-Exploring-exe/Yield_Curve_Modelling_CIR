"""
Visualization
=============
All plots for the Colab notebook. Each function is self-contained:
takes data in, returns (fig, ax) or fig so the notebook can call
plt.show() or fig.savefig() as needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings

from src.models.prediction import PredictionResult, YIELD_COLS, MATURITIES
from src.models.cir_math import CIRParams, cir_yield
from src.data.preprocessing import CleanDataset


# Consistent color palette throughout
COLORS = {
    "actual":     "#2C3E50",
    "cir_base":   "#E74C3C",
    "cir_pp":     "#2ECC71",
    "ekf":        "#3498DB",
    "naive":      "#95A5A6",
    "stress":     "#E67E22",
    "feller_ok":  "#27AE60",
    "feller_bad": "#C0392B",
}


# ---------------------------------------------------------------------------
# 1. Data overview: yield curve history
# ---------------------------------------------------------------------------

def plot_yield_history(dataset: CleanDataset, title: str = "Yield Curve History") -> plt.Figure:
    """
    Heatmap of yield levels over time (dates × maturities).
    Shows rate regimes visually — low-rate era, hike cycle, current level.
    """
    yield_cols = [c for c in YIELD_COLS if c in dataset.df.columns]
    df_yields = dataset.df[yield_cols] * 100   # convert to percent

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: line chart of selected maturities
    ax = axes[0]
    for col, color in zip(["3M", "2Y", "10Y", "30Y"], ["#E74C3C","#F39C12","#2ECC71","#3498DB"]):
        if col in df_yields.columns:
            ax.plot(df_yields.index, df_yields[col], label=col, color=color, linewidth=1.2)
    ax.set_ylabel("Yield (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Bottom: heatmap
    ax2 = axes[1]
    im = ax2.imshow(
        df_yields.values.T,
        aspect="auto",
        cmap="RdYlGn_r",
        origin="lower",
        extent=[0, len(df_yields), 0, len(yield_cols)],
    )
    ax2.set_yticks(np.arange(len(yield_cols)) + 0.5)
    ax2.set_yticklabels(yield_cols)
    ax2.set_xlabel("Time")
    ax2.set_title("Yield Level Heatmap (% — darker = higher rate)")
    plt.colorbar(im, ax=ax2, label="Yield (%)")

    # Add regime annotations
    dates = df_yields.index
    for yr_str, label in [("2020-02", "COVID"), ("2022-03", "Fed hikes")]:
        try:
            idx = dates.searchsorted(pd.Timestamp(yr_str))
            frac = idx / len(dates)
            ax.axvline(dates[idx], color="gray", linestyle="--", alpha=0.5)
            ax.text(dates[idx], ax.get_ylim()[1]*0.95, label,
                    fontsize=8, ha="left", color="gray")
        except Exception:
            pass

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Yield curve snapshot: predicted vs actual on selected dates
# ---------------------------------------------------------------------------

def plot_curve_snapshots(
    result: PredictionResult,
    n_snapshots: int = 6,
    title: str = None,
) -> plt.Figure:
    """
    Show predicted vs actual yield curve on N evenly-spaced test dates.

    This is the most direct visual test of the model: can the predicted
    curve match the actual shape given only the 3M rate?
    """
    yield_cols = list(result.rmse_by_maturity.keys())
    mats = MATURITIES[:len(yield_cols)]
    n = len(result.dates)
    indices = np.linspace(0, n - 1, n_snapshots, dtype=int)

    ncols = 3
    nrows = int(np.ceil(n_snapshots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten()

    title = title or f"Predicted vs Actual Yield Curve — {result.method}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for i, idx in enumerate(indices):
        ax = axes[i]
        actual = result.actual[idx] * 100
        predicted = result.predicted[idx] * 100
        date = result.dates[idx]

        ax.plot(mats, actual, "o-", color=COLORS["actual"], label="Actual", linewidth=2)
        ax.plot(mats, predicted, "s--", color=COLORS["cir_pp"], label="Predicted", linewidth=2)

        rmse = np.sqrt(np.mean((result.actual[idx] - result.predicted[idx])**2)) * 10000
        ax.set_title(f"{date.date()}\nRMSE={rmse:.1f} bps", fontsize=9)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_snapshots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Residuals over time by maturity
# ---------------------------------------------------------------------------

def plot_residuals_over_time(result: PredictionResult) -> plt.Figure:
    """
    Plot prediction errors (in bps) over the test period for each maturity.
    Reveals whether errors are systematic (bias in one direction) or random.
    Systematic patterns → model misspecification.
    """
    yield_cols = list(result.rmse_by_maturity.keys())
    residuals = (result.actual - result.predicted) * 10000   # bps

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f"Prediction Residuals Over Time — {result.method}", fontsize=12, fontweight="bold")

    for i, col in enumerate(yield_cols[:9]):
        ax = axes[i]
        ax.plot(result.dates, residuals[:, i], color=COLORS["cir_pp"], linewidth=0.8, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(np.mean(residuals[:, i]), color="red", linewidth=1.2,
                   linestyle="-", label=f"bias={np.mean(residuals[:, i]):.1f}bps")
        ax.fill_between(result.dates, residuals[:, i], 0,
                        alpha=0.15, color=COLORS["cir_pp"])
        ax.set_title(f"{col} maturity", fontsize=9)
        ax.set_ylabel("Error (bps)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Method comparison bar chart
# ---------------------------------------------------------------------------

def plot_method_comparison(results: List[PredictionResult]) -> plt.Figure:
    """
    Bar chart comparing RMSE by maturity across methods.
    Makes it immediately clear which method is best at which maturities.
    """
    yield_cols = list(results[0].rmse_by_maturity.keys())
    n_mats = len(yield_cols)
    n_methods = len(results)
    x = np.arange(n_mats)
    width = 0.8 / n_methods

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    method_colors = [COLORS["cir_base"], COLORS["cir_pp"], COLORS["ekf"], COLORS["naive"]]

    # RMSE by maturity
    for i, (res, color) in enumerate(zip(results, method_colors)):
        rmse_vals = [res.rmse_by_maturity.get(c, 0) * 10000 for c in yield_cols]
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax1.bar(x + offset, rmse_vals, width * 0.9,
                       label=res.method, color=color, alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(yield_cols)
    ax1.set_ylabel("RMSE (basis points)")
    ax1.set_title("Out-of-Sample RMSE by Maturity and Method")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.axhline(10, color="gray", linestyle=":", alpha=0.7, label="10 bps reference")

    # R² comparison
    method_names = [r.method for r in results]
    r2_vals = [r.r_squared for r in results]
    bar_colors = [
        COLORS["feller_ok"] if v >= 0.85 else COLORS["feller_bad"]
        for v in r2_vals
    ]
    bars = ax2.barh(method_names, r2_vals, color=bar_colors, alpha=0.8)
    ax2.axvline(0.85, color="black", linestyle="--", linewidth=1.5, label="0.85 threshold")
    ax2.set_xlabel("R² (out-of-sample)")
    ax2.set_title("Overall R² by Method (green = passes threshold)")
    ax2.legend()
    ax2.set_xlim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Feller condition tracking
# ---------------------------------------------------------------------------

def plot_feller_analysis(
    feller_df: pd.DataFrame,
    params: CIRParams,
) -> plt.Figure:
    """
    Show Feller condition margin over time.
    Red shading = periods where empirical volatility would violate Feller.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: short rate history
    ax = axes[0]
    ax.plot(feller_df.index, feller_df["r_t"] * 100,
            color=COLORS["actual"], linewidth=1.0, label="3M rate")
    ax.set_ylabel("3M Rate (%)")
    ax.set_title(f"Short Rate History (θ = {params.theta*100:.2f}% marked)")
    ax.axhline(params.theta * 100, color="orange", linestyle="--",
               label=f"θ = {params.theta*100:.2f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: Feller margin
    ax2 = axes[1]
    margin = feller_df["feller_margin_empirical"].dropna()
    dates_margin = feller_df.index[feller_df["feller_margin_empirical"].notna()]

    ax2.plot(dates_margin, margin, color=COLORS["cir_pp"], linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=1.5, linestyle="--",
                label="Feller boundary")
    ax2.fill_between(dates_margin, margin, 0,
                     where=(margin < 0),
                     color=COLORS["feller_bad"], alpha=0.3,
                     label="Feller violated (empirical σ)")
    ax2.fill_between(dates_margin, margin, 0,
                     where=(margin >= 0),
                     color=COLORS["feller_ok"], alpha=0.2,
                     label="Feller satisfied")
    ax2.axhline(params.feller_condition(), color="blue", linestyle=":",
                label=f"Calibrated: 2κθ-σ²={params.feller_condition():.4f}")
    ax2.set_ylabel("2κθ - σ²_empirical")
    ax2.set_title("Feller Condition Margin (empirical rolling σ vs calibrated params)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. CIR++ phi visualisation
# ---------------------------------------------------------------------------

def plot_phi_correction(phi: np.ndarray, params: CIRParams) -> plt.Figure:
    """
    Visualize the CIR++ shift φ(τ) — the structural correction term.

    Where φ is large, the CIR model is systematically biased at that maturity.
    This is a direct diagnostic of the model's limitations.
    """
    yield_cols = YIELD_COLS[:len(phi)]
    mats = MATURITIES[:len(phi)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["feller_ok"] if v >= 0 else COLORS["feller_bad"] for v in phi]
    bars = ax.bar(yield_cols, phi * 10000, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_ylabel("φ(τ) shift (basis points)")
    ax.set_xlabel("Maturity")
    ax.set_title(
        "CIR++ Shift φ(τ) — Structural Gap Between CIR Model and Market\n"
        "Green = CIR underestimates (φ adds yield), Red = CIR overestimates"
    )
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, phi * 10000):
        ypos = val + (2 if val >= 0 else -4)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Calibration parameter comparison across methods
# ---------------------------------------------------------------------------

def plot_param_comparison(param_dict: Dict[str, CIRParams]) -> plt.Figure:
    """
    Bar chart comparing κ, θ, σ across OLS, MLE, EKF.
    Visually demonstrates how different methods give radically different parameters.
    """
    methods = list(param_dict.keys())
    kappas = [p.kappa for p in param_dict.values()]
    thetas = [p.theta * 100 for p in param_dict.values()]
    sigmas = [p.sigma for p in param_dict.values()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    mc = [COLORS["cir_base"], COLORS["cir_pp"], COLORS["ekf"]][:len(methods)]

    for ax, vals, label, unit in zip(
        axes,
        [kappas, thetas, sigmas],
        ["κ (mean-reversion speed)", "θ (long-run mean)", "σ (volatility)"],
        ["", "%", ""],
    ):
        bars = ax.bar(methods, vals, color=mc, alpha=0.8, edgecolor="black")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(f"Value{' ('+unit+')' if unit else ''}")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{val:.4f}{unit}", ha="center", fontsize=9)

    fig.suptitle("Calibrated Parameters: OLS vs MLE vs EKF\n"
                 "OLS/MLE treat 3M as directly observed r_t; EKF treats r_t as latent",
                 fontsize=10)
    plt.tight_layout()
    return fig
