# CIR Yield Curve Model

> [!IMPORTANT]
> **Recommendation:** Please check the `CIR_combined_Results.ipynb` notebook for the complete results and abstracted code. It acts as the central conductor for the whole project, importing source code from different folders in this repository to tell one combined story. **This README is just for reference.**

Finance Club IIT Roorkee · Open Project 2026

---

## What Is This Project? (Simple Version)

When you lend money to someone, you earn interest. The longer you lend, the higher the interest rate you usually demand (because you're taking more risk by waiting longer). The same logic applies to government bonds:

- Lend to the government for **3 months** → you earn one interest rate
- Lend for **30 years** → you earn a higher (usually) interest rate

If you draw all those rates on a graph — with *time horizon* on the x-axis and *interest rate* on the y-axis — you get something called the **yield curve**. It's a snapshot of what borrowing costs look like at every time horizon on a given day.

**The central question this project answers:**

> Given *only* today's 3-month interest rate, can we mathematically predict what the rest of the yield curve looks like — at 6 months, 1 year, 5 years, all the way to 30 years?

This project answers that question using the **CIR model** (Cox-Ingersoll-Ross, 1985) — one of the classic mathematical models used in finance to describe how interest rates move over time.

---

## The Data We Used

- **Training data**: ~1,976 daily observations (May 2016 → April 2024)
  - Each day has yields at **9 maturities**: 3M, 6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
- **Test data**: ~495 daily observations (April 2024 → April 2026)
  - Each day only has **5 maturities**: 3M, 6M, 9M, 1Y, 2Y (the longer ones aren't available)

The data is in *decimal* form: 0.05 means 5%.

### Why is the test set harder, even though it has fewer maturities?

You might think: *"Predicting only 5 maturities instead of 9 sounds easier!"* You're right that there's less to predict. But the difficulty isn't about the number of maturities — it's about the **market conditions** the model has to deal with.

Think of it like training to drive in a parking lot, then being tested on a highway in the rain.

Here's what happened in the data:

| Period | What rates were doing |
|--------|-----------------------|
| 2016–2021 (training) | Near 0% for years — the "free money" era after the 2008 financial crisis |
| 2022 (training, late) | The US Federal Reserve hiked rates extremely fast to fight COVID-era inflation |
| 2023–2026 (test set) | Rates are high and the curve is *inverted* (short-term rates are *higher* than long-term rates) |

**An "inverted" yield curve** is unusual and problematic for our model — normally short rates < long rates, but in 2023-2024 it was the opposite. The CIR model (in its basic form) can produce inversions when the current rate `r` is above the long-run mean `θ` (because investors expect rates to fall, so long yields are lower). However, because the whole curve is controlled by just *one* number (`r`), the model cannot independently control both the level and the slope of the curve at the same time — meaning the shape of the inversion it produces often doesn't match reality. So testing in this era is genuinely harder.

Additionally, the model was trained mostly on near-zero rates, then tested on high rates. A model that "memorises" what low-rate curves look like will struggle badly in a high-rate world.

---

## Project File Layout

```
cir_project/
│
├── data/
│   ├── train_data.csv          ← 1,976 days of training data (9 maturities)
│   ├── test_data.csv           ← 495 days of test data (5 maturities, up to 2Y)
│   └── test_data_3M.csv        ← Challenge input: only 3M rates, no other maturities
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    ← Cleans and prepares the raw data
│   │
│   ├── models/
│   │   ├── cir_math.py         ← The core CIR formulas (A, B, yield calculation)
│   │   ├── two_factor_cir.py   ← A more powerful 2-factor extension of CIR
│   │   ├── calibration.py      ← Finds the best κ, θ, σ values from training data
│   │   └── prediction.py       ← Predicts yield curves from the 3M rate
│   │
│   ├── evaluation/
│   │   └── metrics.py          ← Measures accuracy (R², RMSE, bias per maturity)
│   │
│   └── visualization/
│       └── plots.py            ← Creates all the charts and graphs
│
├── CIR_combined_Results.ipynb  ← THE MAIN SUBMISSION — The unified notebook that imports all modules below.
├── outputs/
│   └── results/                ← Auto-generated CSVs of final model metrics (Leaderboard & Per-Maturity Errors)
├── requirements.txt            ← Python packages needed to run the project
└── README.md                   ← This reference file
```

---

---

## How to Run This Project

1. **Install Requirements**: Ensure you have Python 3.10+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Notebook**: Open `CIR_combined_Results.ipynb` in Jupyter Notebook, VS Code, or Google Colab. Run all cells from top to bottom. The notebook will automatically pull the data, calibrate all models using the code in `src/`, and output the final results into the `outputs/results/` folder.

---

## How the Project Works — Step by Step

---

### Step 1: Clean the Data (`preprocessing.py`)

Before doing any modelling, we clean up the raw CSV files. A few quirks had to be fixed:

**Problem 1 — Column names had accidental spaces**

The CSV had column names like `" ZC025YR"` (with a space at the start). Python treats `" ZC025YR"` and `"ZC025YR"` as completely different column names, so we strip those spaces off first.

**Problem 2 — The time gap between rows isn't always one day**

The dataset only includes *trading days* (no weekends, no public holidays). So a Monday row follows a Friday row in the file — but that's actually a 3-calendar-day gap, not 1. A row after Christmas week might be a 5-day gap.

Why does this matter? Our model's math uses time intervals. If we wrongly assume every row = 1 day, we're feeding incorrect numbers into probability calculations. So instead, for every row we compute the actual gap:

```
dt = (today's calendar date − yesterday's calendar date) / 365.25 days
```

A normal weekday gets `dt ≈ 1/365`. A Monday gets `dt ≈ 3/365`. A post-holiday gets more. This makes the math more accurate.

**Problem 3 — The model breaks if any rate is zero or negative**

The CIR model has a square root of the interest rate (`√r`) in its formula. You can't take the square root of zero or a negative number (well, you can, but it breaks the model's assumptions). Any zero or negative value in the data gets nudged to a tiny positive number before we use it.

**Decision: We kept the big market events (COVID crash, 2022 rate hikes)**

Statistically, the huge rate moves in 2020 (COVID) and 2022 (rate hikes) look like "outliers" that you might want to delete to make the data cleaner. We deliberately kept them. Why?

- The test set is *from* that high-rate era. If we delete the training examples of big rate moves, the model has never seen anything like the test set. It would fail badly.
- These aren't data errors — they're real events that the model must be able to handle.

We *flag* these events in the data (mark them as `stress_event = True`) but don't remove them.

---

### Step 2: The CIR Math — How One Rate Predicts All Others (`cir_math.py`)

This is the core of the project. Here's how the CIR model works, step by step.

#### 2a. The CIR Equation (What it says about how rates move)

The CIR model says: interest rates don't just randomly wander in any direction. They tend to drift back toward some long-run average. Think of it like a rubber band — the further rates get from their "natural" level, the harder they get pulled back.

The equation that captures this is:

```
dr = κ(θ - r)dt  +  σ√r dW
```

In plain English, each part means:

| Symbol | What it is | Plain meaning |
|--------|-----------|---------------|
| `r` | Today's interest rate | The current short rate |
| `dr` | The tiny change in rate | How much the rate moves in a tiny time step |
| `θ` (theta) | Long-run average | The rate level that `r` tends to drift toward over time |
| `κ` (kappa) | Mean-reversion speed | How fast rates snap back to θ — higher κ = faster snapback |
| `σ` (sigma) | Volatility | How noisy/jumpy rates are day-to-day |
| `√r` | Square root of rate | This makes the noise smaller when rates are near zero — so rates can't go negative |
| `dW` | Random noise | The unpredictable, random part of daily rate moves |

The key insight: the `κ(θ - r)dt` part is a *pull*. If today's rate `r` is above θ, this term is negative (pulling r down). If below θ, it's positive (pulling r up). The strength of the pull scales with `κ`.

#### 2b. From the Rate Equation to a Bond Price Formula

This is where the clever maths happens. Mathematicians solved the equation above to find a *closed-form* formula for bond prices. A zero-coupon bond is simply: you pay some amount today, and you get £1 at maturity. The CIR formula gives you the price of that bond for any maturity τ:

```
P(τ) = A(τ) × exp(−B(τ) × r)
```

Where:
- `τ` (tau) is the time to maturity in years (e.g. τ = 0.5 for a 6-month bond, τ = 10 for a 10-year bond)
- `r` is today's short rate (we use the 3M yield as a proxy for this)
- `A(τ)` and `B(τ)` are functions that depend only on κ, θ, σ, and τ — once you know the three parameters, they're just arithmetic

#### 2c. From Bond Prices to Yields — The Actual Prediction Formula

A bond's **yield** is just the annualised return you get from buying it. It's calculated as:

```
yield(τ) = −ln(P) / τ
```

Substituting in the bond price formula:

```
yield(τ) = [B(τ) × r  −  ln(A(τ))]  /  τ
```

**This single formula is the entire yield curve prediction.** Here's how to read it:

1. You have κ, θ, σ (found during training — see Step 3)
2. You compute B(τ) and ln(A(τ)) — these are just plugging numbers into formulas, pure arithmetic
3. You plug in today's 3M rate as `r`
4. You repeat for each maturity: τ = 0.5 (6M), 0.75 (9M), 1.0 (1Y), 2.0 (2Y), etc.
5. Out comes the predicted yield at each maturity

That's the whole prediction — a handful of arithmetic operations repeated for each maturity.

#### Worked Example (Approximate Numbers)

Suppose our calibrated parameters are κ = 0.136, θ = 0.025, σ = 0.036, and today's 3M rate = 5.0% (= 0.05).

For a 2-year bond (τ = 2.0):
- Compute B(2.0) using the formula → gives approximately 1.84
- Compute ln(A(2.0)) using the formula → gives approximately −0.027
- Predicted yield = (1.84 × 0.05 − (−0.027)) / 2.0 = (0.092 + 0.027) / 2.0 ≈ **5.95%**

You repeat this arithmetic for every maturity and you get the full curve.

#### 2d. Why We Use ln(A) Instead of A Directly

A(τ) involves raising a small number to a large power (around 10–50). In a computer, `(small number)^50` can become so tiny it rounds down to zero — a problem called "floating point underflow." Since the yield formula uses `ln(A)` anyway, we just compute that directly and never compute A itself. Problem avoided.

#### 2e. The Feller Condition — A Safety Check

There's a mathematical condition that prevents the model from producing zero or negative rates:

```
2κθ ≥ σ²
```

Intuitively: the upward pull of mean reversion (`κθ`) must be strong enough to overpower the noise (`σ`) near zero. If this is violated, the random noise can push rates to zero (and the `√r` term breaks).

We check this after every calibration. If it's violated, we reduce σ to the maximum value that still satisfies the condition, because σ is the parameter that can be adjusted without breaking the economic interpretation of κ and θ.

#### 2f. The Half-Life of Rate Shocks

With κ = 0.136, you can calculate: if rates jump above (or below) their long-run average by some amount, it takes `ln(2) / 0.136 ≈ 5.1 years` for that gap to shrink by half. This is the "half-life" of a rate shock.

5 years is economically reasonable — central bank rate cycles do tend to take several years to play out.

---

### Step 3: Finding κ, θ, σ From the Data (`calibration.py`)

We have 1,976 days of training data and need to find the three parameter values that make the CIR model fit that data as well as possible. We tried three methods.

---

#### Method 1: OLS (Ordinary Least Squares — Simple Linear Regression)

The CIR equation, when you write it out for daily data steps, looks like a linear regression:

```
change_in_rate = κθ × dt  −  κ × rate × dt  +  noise
```

- The "target" variable is: how much did the rate change today?
- The "features" are: the time step `dt` and yesterday's rate `rate × dt`
- Linear regression finds the coefficients — which tell you `κ` and `κθ`, and from those you get θ

We also divide all variables by `√rate` first (this is called weighted OLS) because the noise in the CIR model isn't constant — it scales with the square root of the rate.

**Result: Total failure.** OLS produced κ ≈ 0.0001 and θ ≈ 878% (a long-run average rate of 878%!?). Obviously nonsense.

**Why it failed:** The 3M rate sat near 0% for 5 straight years (2016–2021), then jumped to ~4.3% in 2022. To OLS, this doesn't look like mean reversion at all — it looks like a process that barely moves, with no clear "average." This is a known problem in statistics called *weak identification*: there isn't enough signal in a single short-rate time series to reliably estimate κ when the data is nearly stuck in one place for years.

---

#### Method 2: MLE (Maximum Likelihood Estimation)

OLS assumes the daily rate changes follow a bell-curve (Gaussian) distribution. That's actually not the right distribution for the CIR model — the mathematically correct distribution is something called a *noncentral chi-squared distribution*.

MLE uses the correct distribution. For each daily observation, it asks: "given yesterday's rate, what's the probability of seeing today's rate under the CIR model?" It then finds the parameters κ, θ, σ that maximise the probability of seeing *all* of the training data.

This is more statistically rigorous than OLS. But it still fails for the same fundamental reason: both OLS and MLE only look at the 3M rate's evolution over time. One single time series, stuck near zero for years, just doesn't have enough information to identify the parameters well.

---

#### Method 3: Extended Kalman Filter (EKF) — The Main Approach

This is the most sophisticated method and the one we actually use. It fixes the fundamental problem with OLS and MLE.

**The key insight that motivates EKF:**

In OLS and MLE, we treated the 3M yield we observe in the market as if it *were* the true short rate `r` in the model. But that's not quite right. The 3M yield is a *noisy measurement* of the underlying true rate. So is every other yield on the curve (6M, 1Y, 10Y, etc.).

Think of it like this: you're trying to track how hot a room is (the true temperature = `r`). You have 9 thermometers, each giving you a slightly different reading because they all have small measurement errors. Each thermometer is a different maturity's yield. You'd want to use *all* of them together to estimate the true temperature — not just one.

**What a Kalman Filter does:**

The Kalman Filter is an algorithm (originally invented for tracking spacecraft) that solves exactly this kind of problem. It combines:

1. A **model of how the thing you're tracking evolves** (here: the CIR equation — rates mean-revert toward θ)
2. **Noisy measurements** of that thing (here: the 9 yield observations every day)

...and outputs the *best possible estimate* of the hidden state (the true `r`) given all available information.

It does this in two steps, every single day:

**Predict step:** Before seeing today's yields, use the CIR equation to predict where the rate probably is today based on yesterday's estimate:

```
r_predicted = r_yesterday × e^(−κ×dt)  +  θ × (1 − e^(−κ×dt))
```

This is just mean reversion in action: yesterday's rate fades toward θ at speed κ. We also track our *uncertainty* in this prediction (how confident we are). The longer the time gap, the less certain we are.

**Update step:** Now today's 9 yields arrive. We ask: how different are they from what we predicted? That gap is called the *innovation* (how surprised we were). We then decide: how much should we revise our estimate of `r` based on this surprise?

The **Kalman Gain** `K` is the key calculation here. It's automatically computed as:

```
K = (our uncertainty in r) × (how sensitive yields are to r) / (total uncertainty)
```

- If our model prediction was very uncertain (big P), K is large → we trust the new yield data more
- If the yield measurements are very noisy (big R), K is small → we trust our model dynamics more
- The filter balances these two sources of information automatically, every day

The final updated estimate is:

```
r_today = r_predicted  +  K × (actual yields − predicted yields)
```

**Why "Extended" Kalman Filter?**

The standard Kalman Filter only works when all the relationships are linear (i.e., straight lines, no exponents). Our yield formula involves exponentials, so it's technically nonlinear.

The "Extended" version handles this by approximating the nonlinear formula with a straight line at each step (using calculus). Conveniently, our specific CIR yield formula `y(τ) = [B(τ) × r − ln(A(τ))] / τ` is actually *already linear in `r`* — the slope is just `B(τ)/τ`. So the approximation is exact in our case, which is a nice property.

**Why EKF gives much better parameters than OLS:**

- OLS used ~1,975 data points (just the daily 3M rate changes)
- EKF uses 9 maturities × 1,976 days = **~17,784 data points** all contributing to parameter estimation
- More data = better, more stable parameter estimates
- The cross-sectional shape of the yield curve (how rates differ across maturities) contains a lot of information about κ, θ, σ that single-time-series methods miss entirely

**EKF result:** κ = 0.136, θ = 2.5%, σ = 3.6% — all economically reasonable.

---

### Step 4: Predicting the Full Yield Curve From Just the 3M Rate (`prediction.py`)

The constraint: on each of the 495 test days, we only get the 3M yield as input. We predict the other maturities.

---

#### Approach 1: Base CIR (Simple and Honest)

Take today's 3M rate, call it `r`. Plug into the formula for each maturity τ:

```
yield(τ) = [B(τ) × r  −  ln(A(τ))] / τ
```

Repeat for τ = 0.5, 0.75, 1.0, 2.0 years (the 4 test maturities we don't have as inputs).

That's the entire method. It's just arithmetic — 4 formula evaluations per day.

**Result: R² = 0.9143, RMSE ≈ 38.7 bps**

Works reasonably well on the 3M–2Y range because CIR fits the short end of the curve better. Struggles when the curve is heavily inverted — CIR can produce inversions when `r > θ`, but with only one factor driving the whole curve, it cannot independently match both the degree and shape of the inversion that the market actually shows.

#### Approach 2: CIR++ (CIR With a Correction Term)

**The problem:** Even a perfectly calibrated CIR model has a structural bias. It CAN produce inverted curves when `r > θ` (current rate above the long-run mean), but the entire curve shape is controlled by a single number (`r`). This means the level of rates and the slope of the curve are not independently controllable — they move together in a fixed way dictated by κ, θ, σ. In 2022–2024, the market showed inversions of a specific shape that single-factor CIR couldn't match well. So CIR will have a predictable, systematic error in those conditions.

**What is CIR++ (the real academic definition)?**

The proper CIR++ model (Brigo & Mercurio) was invented to solve two problems with plain CIR:

1. **CIR forces rates ≥ 0.** Post-2008, countries like Japan and the Eurozone had *negative* interest rates. A standard CIR process can never go below zero. CIR++ fixes this by defining:
   ```
   r_t = x_t + ϕ(t)
   ```
   where `x_t` is a standard CIR process (always ≥ 0) and `ϕ(t)` is a deterministic (non-random) time-dependent shift. If `ϕ(t)` is negative enough, `r_t` can go negative even though `x_t` stays positive.

2. **CIR's 3 parameters (κ, θ, σ) can't exactly fit today's market yield curve.** Any curve the model produces is constrained by those 3 numbers. `ϕ(t)` provides extra degrees of freedom to match the market's actual term structure exactly — it's calibrated analytically to today's observed bond prices.

**What this project actually does (simplified version):**

We don't implement the full time-dependent `ϕ(t)` shift to the short rate process. Instead, we take a simpler, empirical approach: measure the *average error* CIR makes at each maturity over the last 21 training days, and add that as a fixed correction:

```
yield_predicted(τ) = CIR_yield(τ, r)  +  ϕ(τ)
```

`ϕ(τ)` here is one constant number per maturity (not time-dependent), estimated from training data errors rather than derived analytically. It's inspired by the same idea — add a deterministic correction to fix a known model gap — but is a simplified, empirical version rather than the full academic CIR++ model.

`ϕ` is computed entirely from training data before the test period starts. It never gets updated during testing. The only input on any test day is still the 3M rate.

**Why 21 days, not 1 day?**

A single day might be unusual — a quarter-end, a panic day, a Fed meeting day. One unusual day gives a noisy, unrepresentative correction. Averaging over 21 trading days (≈1 calendar month) smooths out the noise and gives a stable estimate of the *persistent* structural gap between CIR and reality. Using 1 day gives R² ≈ 0.84; using 21 days gives R² ≈ 0.8655.

**Result: R² = 0.8655, RMSE ≈ 51.6 bps**

Note: This is *lower* than Base CIR. The correction was calibrated on training data where the full 30Y curve was inverted. The actual test set only goes to 2Y, and by April 2024 the short end of the curve was starting to normalize. So CIR++ slightly overcorrects for the specific test data we have.

---

#### Approach 3: EKF Filter + CIR++

Instead of plugging the raw 3M yield straight into the CIR formula, we first run a Kalman filter step to get a cleaner estimate of the true latent `r`:

1. **Predict step:** use CIR dynamics on yesterday's filtered estimate
2. **Update step:** observe today's 3M yield (just this one number), correct the estimate
3. **Predict full curve:** plug the filtered r into CIR++ formula

The idea: the raw 3M yield is a noisy observation. The filter smooths out some of that noise before we use it to predict the other maturities.

**Result: R² = 0.8607, RMSE ≈ 51.8 bps** — similar to CIR++ alone. The extra filtering step helps marginally but not dramatically.

---

#### The Method We Rejected: Rolling CIR++

An earlier version updated the correction term ϕ every test day using *yesterday's actual observed yields*. This seemed amazing — R² ≈ 0.9981.

But we realised it was essentially cheating. Rewritten, the formula becomes:

```
predicted_today ≈ actual_yields_yesterday  +  tiny_CIR_adjustment
```

The "model" was barely contributing anything. It was just copying yesterday's full yield curve and adding a tiny adjustment. The naive baseline — "tomorrow's curve = today's curve" — achieves R² = 0.9981 by itself. The CIR model was adding nothing.

**This violated the constraint**: we're not allowed to observe yesterday's 6M, 1Y, 2Y yields during testing. Only the 3M rate. Rolling CIR++ secretly used all 9 maturities from yesterday. We cut it entirely.

**Key lesson:** Always compare against the naive "predict tomorrow = today" baseline. If your model barely beats it, you haven't built a model — you've just exploited the fact that interest rates move slowly.

---

#### Approach 4: Two-Factor CIR — The Best Approach

The single-factor CIR model has one source of randomness (`r`). When `r` moves, the *entire* yield curve is forced to respond in the same way — you can't independently tilt or invert the curve. Single-factor CIR CAN produce inversions when `r > θ` (investors expect rates to fall, so long yields are lower), but the level of the curve and the shape of the inversion are coupled. You can't tune one without affecting the other, which means the model often gets the direction of inversion right but the shape wrong.

The Two-Factor CIR model (Longstaff & Schwartz, 1992) adds a second independent CIR process:

```
r_total = r₁ + r₂
```

- `r₁` captures the **level** of interest rates (roughly, the general up/down movement)
- `r₂` captures the **slope** (the difference between short and long rates)

Because `r₁` and `r₂` are independent, the total bond price is just the product of two individual CIR bond prices. Both `r₁` and `r₂` are CIR processes — they stay non-negative at all times. The inversion capability comes from the **speed difference**: factor 2 has a much faster mean-reversion speed (κ₂ > κ₁ by design). This means `B₂(τ)/τ` — how much factor 2 contributes to yield at each maturity — drops off more quickly with maturity than factor 1's contribution. When `r₂` is large relative to `r₁`, short yields get a bigger boost than long yields, naturally producing an inverted curve without either factor needing to go negative.

**Result: R² = 0.9270, RMSE ≈ 34.8 bps** — our best model. It achieves better R² than Base CIR and much lower error (34.8 bps vs 38.7 bps) by correctly modelling curve shape, not just level.

**Honest note on the constant-alpha decomposition:** During the test phase, we can't observe r₁ and r₂ separately (we only see the 3M rate = r₁ + r₂). So we use `r₁ = α·r_t` and `r₂ = (1-α)·r_t` where α was estimated from training data. This means both factors always move together at test time. The real advantage of having two independent factors (level moves separately from slope) isn't fully captured during prediction — what we gain is a *more flexible curve shape* from 6 parameters instead of 3. The improvement is real and meaningful, but it comes from parametric flexibility, not from the full theoretical benefit of independent factors.

---

### Step 5: Measuring Accuracy (`metrics.py`)

We measure how well the model performs using several metrics:

**R² (R-squared):**
- Ranges from −∞ to 1.0
- 1.0 = perfect predictions
- 0.0 = model is no better than just predicting the average yield every day
- Negative = model is *worse* than predicting the average
- Our models: 0.86–0.93

**RMSE (Root Mean Squared Error):**
- Average size of prediction errors, measured in **basis points** (bps)
- 1 basis point = 0.01%, so RMSE of 35 bps means predictions are off by ~0.35% on average
- Easier to interpret than R² for understanding practical accuracy

**Bias per maturity:**
- Positive bias = model systematically predicts *too high* at that maturity
- Negative bias = model systematically predicts *too low*
- Short maturities (3M–1Y) tend to have the biggest bias during the inverted curve period
- Long maturities (20Y–30Y) are easiest to predict — CIR fits the long flat end well

**Feller condition tracking over time:**
- Even though calibrated parameters satisfy Feller globally, during the extreme rate moves of 2022, the *observed* volatility was so high that if you re-calibrated only on those 6 months, Feller would be violated
- We plot this over time to show when the model's assumptions are under stress — a useful diagnostic

---

## Final Results

**The task:** Predict yields at maturities 6M, 9M, 1Y, 2Y using *only* the 3M rate on test days (April 2024 → April 2026).

| Method | R² | Avg Error (2Y RMSE) | Uses forbidden data? |
|--------|-----|---------------------|----------------------|
| Naive baseline (tomorrow = today) | 0.9981 | ~5 bps | Reference — uses yesterday's yields |
| Base CIR | 0.9143 | 38.7 bps | ✅ No — only 3M rate |
| CIR++ (21-day correction) | 0.8655 | 51.6 bps | ✅ No — only 3M rate |
| EKF filter + CIR++ | 0.8607 | 51.8 bps | ✅ No — only 3M rate |
| **Two-Factor CIR** | **0.9270** | **34.8 bps** | **✅ No — only 3M rate** |

**Two-Factor CIR is our best honest model**, beating Base CIR on both metrics and handling the inverted curve problem structurally rather than with a correction patch.

### Why Base CIR Scores Better Than CIR++ Here

In an earlier version of the project (tested on a different time window), Base CIR had R² ≈ 0.59. It jumped to 0.91 on the real test set. Two reasons:

1. The earlier test evaluated the full curve up to 30 years, where CIR fails badly at the inverted long end. The real test only goes to 2 years — the short end where CIR performs better.
2. By April 2024 (start of the test period), the curve had begun to "normalize" (the inversion was easing). So CIR's structural inability to model inversions mattered less.

Meanwhile CIR++'s correction was calibrated on a period of deep inversion. Applied to a normalizing curve, it overcorrects.

### Calibrated Parameters (EKF)

| Parameter | Value | What it means |
|-----------|-------|----------------|
| κ (kappa) | 0.136 | Rate shocks fade to half their size in ~5.1 years |
| θ (theta) | 2.5% | Long-run average rate that rates drift toward |
| σ (sigma) | 3.6% | Annual noise/volatility in the rate process |
| Feller check | 2κθ − σ² > 0 | ✅ Satisfied — rates stay positive |

---

## How to Run It

```bash
# Install required packages
pip install -r requirements.txt

# Generate the submission notebook from source
python build_notebook.py
```

Then open `CIR_Yield_Model.ipynb` in Google Colab to see the full analysis and charts.
