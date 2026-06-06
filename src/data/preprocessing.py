"""
preprocessing.py — Load and clean the raw yield data.

Reads the CSV, fixes column names, clips negative yields, and computes
the exact time gap (dt) between trading days to ensure likelihoods are accurate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name mapping: raw CSV name → (human name, maturity in years)
# The raw columns have a leading space, e.g. " ZC025YR". We strip that.
# ---------------------------------------------------------------------------
MATURITY_MAP: Dict[str, Tuple[str, float]] = {
    " ZC025YR":  ("3M",  0.25),
    " ZC050YR":  ("6M",  0.50),
    " ZC075YR":  ("9M",  0.75),
    " ZC100YR":  ("1Y",  1.00),
    " ZC200YR":  ("2Y",  2.00),
    " ZC500YR":  ("5Y",  5.00),
    " ZC1000YR": ("10Y", 10.00),
    " ZC2000YR": ("20Y", 20.00),
    " ZC3000YR": ("30Y", 30.00),
}

SHORT_RATE_COL = "3M"   # the column used as r_t input during prediction
MATURITIES_YR  = np.array([m for _, m in MATURITY_MAP.values()])

# All possible human-readable yield column names in maturity order
ALL_YIELD_COLS  = [name for name, _ in MATURITY_MAP.values()]
ALL_MATURITIES  = MATURITIES_YR  # alias kept for backward compat


# ---------------------------------------------------------------------------
# Data classes — structured containers so callers don't pass loose variables
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingReport:
    """Printed summary of everything that happened during cleaning."""
    n_rows_raw: int
    n_rows_clean: int
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    n_date_gaps: int        # number of transitions where gap > 3 calendar days
    max_gap_days: int
    outliers_flagged: Dict[str, int]
    large_jumps_flagged: Dict[str, int]
    feller_eligible: bool
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 60, "  PREPROCESSING REPORT", "=" * 60,
            f"  Raw rows         : {self.n_rows_raw}",
            f"  Clean rows       : {self.n_rows_clean}",
            f"  Date range       : {self.date_start.date()} -> {self.date_end.date()}",
            f"  Date gaps (>3d)  : {self.n_date_gaps} (max {self.max_gap_days} days)",
            f"  Outliers flagged : {sum(self.outliers_flagged.values())} total",
        ]
        for col, n in self.outliers_flagged.items():
            if n > 0:
                lines.append(f"    {col}: {n} rows")
        lines.append(f"  Large jumps      : {sum(self.large_jumps_flagged.values())} total")
        for col, n in self.large_jumps_flagged.items():
            if n > 0:
                lines.append(f"    {col}: {n} rows")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  NOTE: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class CleanDataset:
    """Everything the modelling pipeline needs, bundled together."""
    df: pd.DataFrame      # DatetimeIndex, one column per maturity + stress_event
    maturities: np.ndarray          # maturities present, in years (e.g. [0.25, 0.5, ..., 30.0])
    short_rate: pd.Series           # the 3M column, pulled out for convenience
    dt_series: pd.Series            # time gap to the PREVIOUS row, in years
    report: PreprocessingReport


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def load_and_clean(
    csv_path: str,
    iqr_multiplier: float = 3.0,
    jump_multiplier: float = 20.0,
    verbose: bool = True,
    require_all_maturities: bool = False,
) -> CleanDataset:
    """
    Load the raw CSV and return a clean, ready-to-use CleanDataset.

    Parameters
    ----------
    csv_path               : path to train_data.csv or test_data.csv
    iqr_multiplier         : how many IQRs away from Q1/Q3 counts as an outlier.
                             3.0 is conservative — we want to flag, not remove.
    jump_multiplier        : a day-over-day change larger than this × median daily
                             change gets flagged as a stress event (not removed).
    verbose                : print the preprocessing report when done.
    require_all_maturities : if True, raise if any of the 9 maturities are missing.
                             Default False — accepts any subset (e.g. test_data.csv
                             with only 5 maturities).
    """

    # ── 1. Read raw CSV ─────────────────────────────────────────────────────
    raw = pd.read_csv(csv_path)
    n_rows_raw = len(raw)

    # ── 2. Strip leading spaces from column names, use human-readable labels ─
    rename_map = {"Date": "Date"}
    for raw_col, (clean_name, _) in MATURITY_MAP.items():
        rename_map[raw_col] = clean_name
    raw.rename(columns=rename_map, inplace=True)

    # ── 3. Parse dates, sort oldest→newest, set as index ───────────────────
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw.sort_values("Date", inplace=True)
    raw.set_index("Date", inplace=True)

    # ── 4. Determine which maturities are present ───────────────────────────
    #  Accept any subset of the full 9. Build maturity list in canonical order.
    expected_cols = ALL_YIELD_COLS  # full list of 9 human names
    present_cols  = [c for c in expected_cols if c in raw.columns]
    missing_cols  = [c for c in expected_cols if c not in raw.columns]

    if require_all_maturities and missing_cols:
        raise ValueError(
            f"Missing maturity columns: {missing_cols}. "
            "Pass require_all_maturities=False to allow partial datasets."
        )

    if missing_cols and verbose:
        warnings.warn(
            f"Partial dataset: {len(present_cols)}/{len(expected_cols)} maturities found. "
            f"Missing: {missing_cols}. This is normal for test_data.csv.",
            UserWarning,
            stacklevel=2,
        )

    if SHORT_RATE_COL not in present_cols:
        raise ValueError(
            f"The short-rate column '{SHORT_RATE_COL}' (3M) is required but missing."
        )

    df = raw[present_cols].copy()

    # Build the maturity array for the columns we actually have
    col_to_mat = {name: mat for name, mat in MATURITY_MAP.values()}
    maturities = np.array([col_to_mat[c] for c in present_cols])

    # ── 5. Handle missing values ─────────────────────────────────────────────
    # Forward-fill gaps ≤2 days first (handles single NaN cleanly),
    # then interpolate anything longer, then backfill the very start.
    n_missing = df.isnull().sum().sum()
    notes = []
    if n_missing > 0:
        notes.append(f"{n_missing} missing values imputed (ffill → time interpolate → bfill).")
        df.ffill(limit=2, inplace=True)
        df.interpolate(method="time", inplace=True)
        df.bfill(inplace=True)
    assert df.isnull().sum().sum() == 0, "NaN values remain after imputation"

    # ── 6. Clip non-positive yields ──────────────────────────────────────────
    # CIR requires r_t > 0 because of the sqrt(r_t) term.
    # Near-zero negative yields can appear in some markets. Clip them.
    n_nonpositive = (df <= 0).sum().sum()
    if n_nonpositive > 0:
        notes.append(f"{n_nonpositive} non-positive yields clipped to column minimum.")
        for col in df.columns:
            min_pos = df[col][df[col] > 0].min()
            df[col] = df[col].clip(lower=min_pos)

    # ── 7. Flag IQR outliers — but do NOT remove them ───────────────────────
    # The 2022 rate-hike cycle looks like an outlier statistically but is real.
    # Removing it would make the model look good in-sample and fail on the
    # test set — the exact regime we care about.
    outliers_flagged: Dict[str, int] = {}
    for col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - iqr_multiplier * IQR) | (df[col] > Q3 + iqr_multiplier * IQR)
        outliers_flagged[col] = int(mask.sum())
    n_outlier_rows = sum(outliers_flagged.values())
    if n_outlier_rows > 0:
        notes.append(
            f"{n_outlier_rows} rows flagged as outliers (IQR×{iqr_multiplier}). "
            "RETAINED — they are real market events."
        )

    # ── 8. Flag large day-over-day jumps as stress events ───────────────────
    # These are real (COVID, rate hikes). We tag them so we can study model
    # performance on stress days vs calm days separately.
    daily_change = df.diff().abs()
    large_jumps_flagged: Dict[str, int] = {}
    stress_mask = pd.Series(False, index=df.index)
    for col in df.columns:
        median_chg = daily_change[col].median()
        threshold = jump_multiplier * median_chg
        col_mask = daily_change[col] > threshold
        large_jumps_flagged[col] = int(col_mask.sum())
        stress_mask |= col_mask
    df["stress_event"] = stress_mask.fillna(False).astype(bool)
    n_stress = int(df["stress_event"].sum())
    if n_stress > 0:
        notes.append(
            f"{n_stress} stress event rows flagged (jump >{jump_multiplier}× median). RETAINED."
        )

    # ── 9. Compute dt = actual time gap to previous row, in years ──────────
    # We use real calendar differences, not a fixed 1/252.
    # A Fri→Mon gap = 3 days = 3/365.25 years, not 1/252.
    # This is critical for the MLE and EKF likelihood computations.
    dt_days = pd.Series(df.index, index=df.index).diff().dt.days.fillna(1)
    dt_series = (dt_days / 365.25).rename("dt")

    # ── 10. Audit date gaps ─────────────────────────────────────────────────
    n_date_gaps = int((dt_days > 3).sum())
    max_gap_days = int(dt_days.max())

    report = PreprocessingReport(
        n_rows_raw=n_rows_raw,
        n_rows_clean=len(df),
        date_start=df.index[0],
        date_end=df.index[-1],
        n_date_gaps=n_date_gaps,
        max_gap_days=max_gap_days,
        outliers_flagged=outliers_flagged,
        large_jumps_flagged=large_jumps_flagged,
        feller_eligible=True,
        notes=notes,
    )
    if verbose:
        print(report)

    return CleanDataset(
        df=df,
        maturities=maturities,
        short_rate=df[SHORT_RATE_COL],
        dt_series=dt_series,
        report=report,
    )


def train_test_split_by_date(
    dataset: CleanDataset,
    test_start: str,
) -> Tuple[CleanDataset, CleanDataset]:
    """
    Split a CleanDataset into train and test by date.
    test_start: e.g. '2023-01-01' — first date in the test set.
    """
    split = pd.Timestamp(test_start)
    df_train = dataset.df[dataset.df.index < split].copy()
    df_test  = dataset.df[dataset.df.index >= split].copy()

    def _subset(df_sub: pd.DataFrame) -> CleanDataset:
        dt_days = pd.Series(df_sub.index, index=df_sub.index).diff().dt.days.fillna(1)
        # Preserve only the yield columns (not stress_event) for maturities lookup
        yield_cols = [c for c in df_sub.columns if c != "stress_event"]
        col_to_mat = {name: mat for name, mat in MATURITY_MAP.values()}
        mats = np.array([col_to_mat[c] for c in yield_cols if c in col_to_mat])
        return CleanDataset(
            df=df_sub,
            maturities=mats,
            short_rate=df_sub[SHORT_RATE_COL],
            dt_series=(dt_days / 365.25).rename("dt"),
            report=PreprocessingReport(
                n_rows_raw=len(df_sub), n_rows_clean=len(df_sub),
                date_start=df_sub.index[0], date_end=df_sub.index[-1],
                n_date_gaps=int((dt_days > 3).sum()),
                max_gap_days=int(dt_days.max()),
                outliers_flagged={}, large_jumps_flagged={},
                feller_eligible=True,
                notes=[f"Subset from split at {split.date()}"],
            ),
        )

    train_ds, test_ds = _subset(df_train), _subset(df_test)
    print(f"Train: {len(df_train)} rows  ({df_train.index[0].date()} -> {df_train.index[-1].date()})")
    print(f"Test : {len(df_test)} rows  ({df_test.index[0].date()} -> {df_test.index[-1].date()})")
    return train_ds, test_ds


def yield_curve_stats(dataset: CleanDataset) -> pd.DataFrame:
    """Return mean, std, min, max per maturity. For the EDA section of the notebook."""
    yield_cols = [c for c in dataset.df.columns if c != "stress_event"]
    return dataset.df[yield_cols].describe().T.round(6)
