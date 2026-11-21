"""Stage 4: festive peaks + weekly demand forecast per category.

Run:  python -m src.forecast

What it does
  1. Builds a complete weekly table (units, revenue, orders, avg price) per category with SQL.
  2. Finds each category's festive ramp-up and peak.
  3. Forecasts weekly units at two horizons: 1 week ahead and 4 weeks ahead
     (4 weeks is what a stock planner really needs).
  4. Tests on the festive window (from config.TEST_START) against simple baselines.

Honesty rules baked in
  * Features for week t only use data from week t-h or earlier (no peeking).
  * The model type is chosen on a validation window BEFORE the test window.
  * We report where the model loses to a baseline, not only where it wins.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from src.config import DB_PATH, OUT_DIR, TEST_START, LAST_COMPLETE_WEEK_END

VAL_START = "2011-07-04"   # validation window = [VAL_START, TEST_START)
HORIZONS = (1, 4)


# ------------------------------------------------------------------ 1. weekly table
def weekly_table(con=None) -> pd.DataFrame:
    own = con is None
    con = con or sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT c.category, s.week_start,
               SUM(s.quantity) AS units, SUM(s.revenue) AS revenue,
               COUNT(DISTINCT s.invoice) AS orders, AVG(s.price) AS avg_price
        FROM sales s JOIN product_category c USING (stockcode)
        GROUP BY c.category, s.week_start""", con, parse_dates=["week_start"])
    if own:
        con.close()
    weeks = pd.date_range("2009-11-30", pd.Timestamp(LAST_COMPLETE_WEEK_END) - pd.Timedelta(days=6), freq="W-MON")
    out = []
    for cat, g in df.groupby("category"):
        g = g.set_index("week_start").reindex(weeks)
        g["category"] = cat
        g[["units", "revenue", "orders"]] = g[["units", "revenue", "orders"]].fillna(0)
        g["avg_price"] = g["avg_price"].ffill().bfill()
        out.append(g.rename_axis("week_start").reset_index())
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ 2. festive peaks
def festive_peaks(wk: pd.DataFrame) -> pd.DataFrame:
    """For each category and year: when did the ramp-up start and when was the peak?
    baseline = median weekly revenue Jan-Aug of that year; ramp start = first week from Sep
    with revenue >= 1.5 x baseline."""
    rows = []
    for (cat, ), g in wk.groupby(["category"]):
        g = g.set_index("week_start")["revenue"]
        for year in (2010, 2011):
            base = g[(g.index >= f"{year}-01-04") & (g.index < f"{year}-09-01")].median()
            fall = g[(g.index >= f"{year}-09-01") & (g.index <= f"{year}-12-31")]
            if base <= 0 or fall.empty:
                continue
            hot = fall[fall >= 1.5 * base]
            rows.append({"category": cat, "year": year,
                         "baseline_weekly_revenue": round(base),
                         "ramp_start_week": hot.index[0].date() if len(hot) else None,
                         "peak_week": fall.idxmax().date(),
                         "peak_multiple_of_baseline": round(fall.max() / base, 1)})
    return pd.DataFrame(rows).sort_values(["category", "year"])


# ------------------------------------------------------------------ 3. features
def make_features(wk: pd.DataFrame, h: int) -> pd.DataFrame:
    """One row per (category, week). All inputs come from week t-h or earlier."""
    frames = []
    for cat, g in wk.groupby("category"):
        g = g.sort_values("week_start").reset_index(drop=True)
        y = g["units"]
        f = pd.DataFrame({"category": cat, "week_start": g["week_start"], "y": y})
        f["lag_h"] = y.shift(h)
        f["ma4"] = y.shift(h).rolling(4).mean()
        f["ly"] = y.shift(52).rolling(3, center=True).mean()   # same week last year (3-wk avg)
        recent = y.shift(h).rolling(8).sum()
        a_year_ago = y.shift(h + 52).rolling(8).sum()
        f["yoy_ratio"] = ((recent + 1) / (a_year_ago + 1)).clip(0.5, 2.0)
        f["price_h"] = g["avg_price"].shift(h)
        f["woy"] = g["week_start"].dt.isocalendar().week.astype(int)
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    f["baseline_ly"] = f["ly"]
    f["baseline_ly_growth"] = f["ly"] * f["yoy_ratio"]
    f["baseline_ma4"] = f["ma4"]
    return f.dropna(subset=["lag_h", "ma4", "ly", "yoy_ratio"]).reset_index(drop=True)


FEATURES = ["log_lag", "log_ma4", "log_ly", "log_yoy", "log_price", "woy_sin", "woy_cos"]


def design(f: pd.DataFrame, cats: list) -> pd.DataFrame:
    X = pd.DataFrame({
        "log_lag": np.log1p(f["lag_h"]), "log_ma4": np.log1p(f["ma4"]),
        "log_ly": np.log1p(f["ly"]), "log_yoy": np.log(f["yoy_ratio"]),
        "log_price": np.log(f["price_h"].clip(lower=0.01)),
        "woy_sin": np.sin(2 * np.pi * f["woy"] / 52), "woy_cos": np.cos(2 * np.pi * f["woy"] / 52),
    })
    for c in cats:   # one-hot category
        X["cat_" + c] = (f["category"] == c).astype(float)
    return X


def fit_predict(kind, train, test, cats):
    Xtr, Xte = design(train, cats), design(test, cats)
    ytr = np.log1p(train["y"])
    if kind == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=8, random_state=0)
    model.fit(Xtr, ytr)
    return np.clip(np.expm1(model.predict(Xte)), 0, None)


def wape(y, yhat):
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


# ------------------------------------------------------------------ 4. evaluation
def run():
    wk = weekly_table()
    wk.to_csv(OUT_DIR / "weekly_by_category.csv", index=False)
    festive_peaks(wk).to_csv(OUT_DIR / "festive_peaks.csv", index=False)
    cats = sorted(wk["category"].unique())
    results, per_cat, preds_all = [], [], []

    for h in HORIZONS:
        f = make_features(wk, h)
        val_start, test_start = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
        # --- choose the model type on the validation window (no test data touched)
        tr, va = f[f["week_start"] < val_start], f[(f["week_start"] >= val_start) & (f["week_start"] < test_start)]
        val_scores = {k: wape(va["y"].values, fit_predict(k, tr, va, cats)) for k in ("ridge", "hgb")}
        best = min(val_scores, key=val_scores.get)
        # --- final test: train on everything before the test window
        tr, te = f[f["week_start"] < test_start], f[f["week_start"] >= test_start].copy()
        te["model"] = fit_predict(best, tr, te, cats)
        te["horizon_weeks"] = h
        scores = {"model (" + best + ")": wape(te["y"].values, te["model"].values)}
        for b in ("baseline_ly", "baseline_ly_growth", "baseline_ma4"):
            scores[b] = wape(te["y"].values, te[b].values)
        # NOTE: this blend was added AFTER the first look at test results, so treat its score as
        # optimistic. It is reported separately and flagged in the README.
        te["blend"] = 0.5 * te["model"] + 0.5 * te["baseline_ly"]
        scores["blend 0.5*model+0.5*last-year (added after first look)"] = wape(te["y"].values, te["blend"].values)
        for name, s in scores.items():
            results.append({"horizon_weeks": h, "method": name, "test_WAPE_pct": round(100 * s, 1),
                            "test_weeks": te["week_start"].nunique(), "train_rows": len(tr)})
        for cat, g in te.groupby("category"):
            row = {"horizon_weeks": h, "category": cat,
                   "model_WAPE_pct": round(100 * wape(g["y"].values, g["model"].values), 1)}
            for b in ("baseline_ly", "baseline_ly_growth", "baseline_ma4"):
                row[b + "_WAPE_pct"] = round(100 * wape(g["y"].values, g[b].values), 1)
            per_cat.append(row)
        preds_all.append(te[["category", "week_start", "horizon_weeks", "y", "model", "blend", "baseline_ly",
                             "baseline_ly_growth", "baseline_ma4"]])
        print(f"h={h}: validation WAPE {val_scores} -> chose {best}")

    res, pc = pd.DataFrame(results), pd.DataFrame(per_cat)
    res.to_csv(OUT_DIR / "forecast_results.csv", index=False)
    pc.to_csv(OUT_DIR / "forecast_by_category.csv", index=False)
    pd.concat(preds_all).to_csv(OUT_DIR / "forecast_test_predictions.csv", index=False)
    print("\n=== Test-window results (lower is better) ===")
    print(res.to_string(index=False))
    print("\n=== By category ===")
    print(pc.to_string(index=False))
    return res, pc


if __name__ == "__main__":
    run()
