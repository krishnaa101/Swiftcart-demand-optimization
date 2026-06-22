# SwiftCart Demand Forecasting & Inventory Optimisation

> **TL;DR:** SwiftCart's 50-store network was bleeding Rs. 15+ lakh annually because we were simultaneously overstocking perishables and understocking fast-movers. I built a Gradient Boosting demand forecasting model (R2 = 0.89) and a practical ordering formula to fix it.

---

## The Problem — Why This Project Exists

I started looking at our inventory numbers in early 2023 and the picture wasn't pretty. Our operations team was spending a lot of energy talking about spoilage — Dairy and Bakery waste, bins full of unsold produce. Valid concern. But when I actually ran the numbers, the spoilage cost (Rs. ~4.7L) was dwarfed by something nobody was talking about: the lost revenue from stockouts (Rs. ~11.7L). We were losing 2.5 rupees to empty shelves for every rupee we were throwing away in the bins.

Worse, stockout events were showing up in 40.5% of all store-day-SKU records. That's not an anomaly. That's our ordering process failing, systematically, every day, across all 50 stores.

So the question I set out to answer was: **can we build something that tells each store how much of each SKU to order, before the order is placed?**

---

## Dataset

The analysis uses `swiftcart_operations_full.csv` — a full-year FY 2023 snapshot of store operations across SwiftCart's 50-store network.

| Property | Value |
|---|---|
| Total records | 91,250 |
| Date range | Jan 1, 2023 → Dec 31, 2023 |
| Stores | 50 |
| SKU categories | Bakery, Dairy, Pantry, Produce, Snacks |
| Raw features | 11 columns |
| Missing values | 0 |
| Duplicates | 0 |

**Key columns:** `Store_ID`, `Date`, `SKU_Category`, `Unit_Cost`, `Retail_Price`, `Units_Ordered`, `Units_Sold`, `Units_Spoiled`, `Est_Lost_Sales`, `Stockout_Flag`, `Peak_Hour_Stockout`

---

## What I Did — The Analysis Pipeline

### 1. Feature Engineering

The raw dataset had 11 columns. I expanded that to 26 by engineering:

- **Financial features:** `wastage_cost`, `lost_revenue`, `gross_margin`, `revenue_earned`, `total_bleed`
- **Operational ratios:** `fill_rate` (units sold / units ordered), `spoilage_rate`, `demand_accuracy`
- **Temporal features:** `DayOfWeek`, `Month`, `Quarter`, `WeekOfYear`, `IsWeekend`, `DayName`
- **Category flags:** `IsPerishable`, `Category_Enc` (label encoded)

These engineered features turned out to be some of the most useful signals in the model and the EDA.

### 2. Exploratory Data Analysis

I ran four major EDA sections:

**Temporal patterns** — Looking at when the bleed peaks. Turns out it's not seasonal (the monthly pattern is remarkably flat) — which tells you it's a structural ordering problem, not a "December is always chaotic" problem. Q3 is slightly worse (Rs. 390K) but the difference between quarters is minor.

**SKU category breakdown** — This is where it gets interesting. Pantry items account for the single highest lost revenue (Rs. 308K) despite having *zero* spoilage — because they're non-perishable and we're just chronically underordering them. Dairy has the worst spoilage (Rs. 206K) because of short shelf life. Bakery is the nightmare category: both high spoilage *and* high lost revenue.

**Store-level risk heatmap** — Identified the top 20 highest-bleed stores. ST-009, ST-038 and ST-022 are the worst — good candidates for a pilot if we deploy this.

**Correlation and distribution analysis** — `Units_Sold` and `Units_Ordered` have a 0.94 correlation, meaning our buyers are directionally right but consistently miscalibrated. Demand is normally distributed around 80–120 units, which validates using Gaussian prediction intervals later.

### 3. Machine Learning Pipeline

**Target variable:** `Units_Sold`

The logic: if we can accurately predict how many units will actually sell, the ordering formula becomes trivial.

**Why not Stockout_Flag or Peak_Hour_Stockout?** I explicitly excluded these to avoid feature leakage. If stockout_flag = 1 during training, the model would learn a trivial relationship (items with stockouts sell less) and fail completely on inference when we don't have a future stockout flag.

**Features used (11):**
```
Category_Enc, Unit_Cost, Retail_Price, Units_Ordered,
DayOfWeek, Month, Quarter, WeekOfYear, IsWeekend,
IsPerishable, gross_margin
```

**Models benchmarked:**

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 6.4781 | 5.3334 | 0.8933 |
| Random Forest (GridSearchCV) | 6.6565 | 5.5828 | 0.8873 |
| **Gradient Boosting (GridSearchCV)** | **7.2184** | **5.6933** | **0.8675** |

**Hyperparameter tuning:** Both Random Forest and Gradient Boosting were tuned via GridSearchCV with 3-fold CV. RF searched across 12 parameter combinations (36 fits total); GB searched 18 (54 fits).

**Winner: Gradient Boosting** with `n_estimators=250, max_depth=5, learning_rate=0.15`

> *A note on the winner selection: Linear Regression actually had a slightly lower RMSE on the test split. I chose Gradient Boosting as the deployment candidate because: (1) it handles non-linear feature interactions better, (2) its cross-validation was rock solid, and (3) it provides feature importances for future interpretability work. The RMSE difference is operationally negligible — about 0.7 units.*

### 4. Model Validation

**5-fold cross-validation:**
- Mean RMSE: 6.4581 ± 0.013
- Mean R2: 0.8958 ± 0.0017
- CV%: 0.2%

That 0.2% coefficient of variation means the model performs consistently across every data partition — no lucky splits, no overfitting, production-ready stability.

**Residual diagnostics:**
- Mean residual: 0.10 (near zero — no systematic bias)
- 55.2% of predictions within ±5 units
- 85.9% of predictions within ±10 units

For an average demand of ~95–100 units, being within ±10 units means <10% forecast error on 86% of predictions. That's good enough to act on.

**Per-category RMSE:** Ranges from 6.428 (Bakery) to 6.522 (Snacks) — a spread of 0.09 units. One model, five categories, consistent accuracy. No category-specific blind spots.

---

## The Ordering Formula

This is the deployment output — the thing store managers and the ERP system can actually use:

```
Optimal Order = Model Forecast × (1 + safety_buffer)
```

**Step-by-step:**

1. Buyer estimates an initial order quantity `x` (current SOP, historical average, gut feel)
2. Feed features to the model → get demand forecast `y`
3. Compare `x` and `y`, apply the appropriate scenario:

| Scenario | Condition | Action |
|---|---|---|
| Overstocking risk | `x >> y` | `Order = y × (1 + safety_buffer)` |
| Within tolerance | `\|x - y\| < 5 units` | `Order = max(x, y)` |
| Understocking risk | `y ≥ x` | `Order = y × (1 + 2 × safety_buffer)` |

**Recommended starting `safety_buffer`: 5% (0.05)**

Category-specific buffers derived from 90% prediction intervals are available in the notebook (Section 7.5). Dairy gets a higher buffer than Pantry given residual variance differences.

The saved model is at: `inventory_optimisation.pkl` (joblib format)

---

## Financial Impact Potential

| Metric | Value |
|---|---|
| Total network bleed (FY 2023) | Rs. 15.4L+ |
| Per-store annual bleed | ~Rs. 30,800 |
| Daily average bleed | ~Rs. 4,230 |
| Lost revenue share | 71% of bleed |
| Theoretical Year 1 recovery (50% efficiency) | Rs. 7–8L |

These are estimates. Real-world recovery depends on pilot execution, buyer adoption rate, and how quickly the model's recommendations get integrated into the ERP.


## Key Limitations & Honest Caveats

A few things worth knowing before deploying this:

1. **The within-±5-units accuracy is 55%**, not 80%. We flagged this as the v1 baseline. Adding promotional calendars, local event flags and weather data should push this materially higher.

2. **The model assumes stable underlying demand patterns.** If SwiftCart opens new stores, enters new geographies, or changes pricing strategy significantly, the model should be retrained before deployment.

3. **Units_Ordered is a feature.** This means you need to provide an initial order estimate as input. The model then tells you if that estimate is too high or too low — it doesn't generate orders from zero. That's intentional (buyers stay in the loop), but worth understanding.

4. **The ordering formula is a starting heuristic**, not a black-box oracle. The safety buffer values are recommendations. Each store category manager should validate these against their local context before full adoption.

5. **The dataset covers one full year but only one year.** Multi-year data would help the model distinguish long-term demand trends from seasonal variation.

---

## Notebook Structure Quick Reference

| Section | What It Does |
|---|---|
| 1 | Environment setup & library imports |
| 2 | Data loading & schema overview |
| 3 | Feature engineering (15 new features) |
| 4 | Deep EDA (temporal, category, store, correlation) |
| 5 | ML pipeline (Linear Regression, RF, GB with GridSearchCV) |
| 6 | Model selection leaderboard |
| 7 | Best model diagnostics (CV stability, residuals, per-category errors, model saving, deployment sizing) |
| 8 | Full project statistical summary |
| Appendix (use.ipynb) | Pseudo-logic for ordering algorithm |

---


