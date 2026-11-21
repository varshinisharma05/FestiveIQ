"""Stage 6: price sensitivity (elasticity) and a discount scenario simulator.

Price elasticity e = % change in units for a 1% change in price (negative: higher price, fewer units).
Estimated with a fixed-effects regression on SKU-week data:

    log(units) = e * log(price) + SKU effect + week effect + noise

SKU effects remove "this product is just popular"; week effects remove "this was the festive week".
What remains is how a product's units move when ITS OWN price differs from its usual price.

BIG CAVEATS (state these in interviews):
  * The data has no promotion flag. Price differences here are mostly wholesale quantity tiers
    and list-price changes, so the estimate is an ASSOCIATION, not a proven causal effect.
  * Few products change price at all. We only use products that do (>= 26 active weeks, price varies >= 3%).
  * Cost is not in the data. The simulator uses an ASSUMED margin from config.py, adjustable on screen.
So the simulator is a SCENARIO tool: it shows how profit reacts to a discount under stated assumptions.

Run:  python -m src.pricing
"""
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.config import DB_PATH, OUT_DIR, ASSUMED_MARGIN


# ------------------------------------------------------------------ estimation
def sku_week_panel(con) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT s.stockcode, c.category, s.week_start,
               SUM(s.quantity) AS units,
               SUM(s.revenue) * 1.0 / SUM(s.quantity) AS price_rev_per_unit,
               AVG(s.price) AS price_avg_line, MIN(s.price) AS price_min, MAX(s.price) AS price_max
        FROM sales s JOIN product_category c USING (stockcode)
        GROUP BY s.stockcode, s.week_start""", con)


def _demean_two_way(df, cols, g1, g2, iters=15):
    out = df[cols].copy()
    for _ in range(iters):
        out = out - out.groupby(df[g1]).transform("mean")
        out = out - out.groupby(df[g2]).transform("mean")
    return out


def estimate(panel: pd.DataFrame, min_weeks=26, min_price_cv=0.03):
    p = panel[(panel["units"] > 0) & (panel["price"] > 0)].copy()
    p["ly"], p["lp"] = np.log(p["units"]), np.log(p["price"])
    stats = p.groupby("stockcode").agg(weeks=("ly", "size"), cv=("lp", "std"))
    keep = stats[(stats["weeks"] >= min_weeks) & (stats["cv"] >= min_price_cv)].index
    p = p[p["stockcode"].isin(keep)].reset_index(drop=True)

    def fit(d):
        if d["stockcode"].nunique() < 15:
            return None
        z = _demean_two_way(d, ["ly", "lp"], "stockcode", "week_start")
        m = sm.OLS(z["ly"], z[["lp"]]).fit(cov_type="cluster", cov_kwds={"groups": d["stockcode"]})
        lo, hi = m.conf_int().loc["lp"]
        return {"elasticity": round(float(m.params["lp"]), 2), "ci_low": round(float(lo), 2),
                "ci_high": round(float(hi), 2), "skus_used": d["stockcode"].nunique(), "obs": len(d)}

    rows = [dict(category="ALL (pooled)", **fit(p))]
    for cat, d in p.groupby("category"):
        r = fit(d)
        if r:
            rows.append(dict(category=cat, **r))
    kept_share = len(keep) / panel["stockcode"].nunique()
    return pd.DataFrame(rows), kept_share


# ------------------------------------------------------------------ scenario maths
def simulate(units0: float, price0: float, discount: float, elasticity: float, margin: float):
    """Constant-elasticity scenario. discount in 0..1. Cost per unit = price0 * (1 - margin)."""
    price = price0 * (1 - discount)
    units = units0 * (1 - discount) ** elasticity
    cost = price0 * (1 - margin)
    return {"price": price, "units": units, "revenue": units * price, "profit": units * (price - cost)}


def breakeven_elasticity(margin: float) -> float:
    """A small discount raises profit only if |elasticity| > 1/margin (derivation in README)."""
    return -1.0 / margin


def best_discount(units0, price0, elasticity, margin, max_discount=0.5):
    grid = np.arange(0, max_discount + 1e-9, 0.01)
    profits = [simulate(units0, price0, d, elasticity, margin)["profit"] for d in grid]
    i = int(np.argmax(profits))
    return float(grid[i]), profits[i]


def main():
    con = sqlite3.connect(DB_PATH)
    panel = sku_week_panel(con)
    con.close()
    base = panel[["stockcode", "category", "week_start", "units"]]
    # PRIMARY: only weeks where the product had ONE list price (no wholesale-tier mixing)
    single = panel[panel["price_min"] == panel["price_max"]]
    res, kept = estimate(single.assign(price=single["price_avg_line"])[list(base.columns) + ["price"]])
    res.insert(1, "variant", "PRIMARY: single-list-price weeks")
    # SENSITIVITY: other ways to define price
    sens = []
    for name, col in [("sensitivity: average of line prices", "price_avg_line"),
                      ("sensitivity: revenue per unit (inflated by wholesale tiers)", "price_rev_per_unit")]:
        r, _ = estimate(panel.assign(price=panel[col])[list(base.columns) + ["price"]])
        sens.append(r.iloc[[0]].assign(variant=name))
    res = pd.concat([res] + sens, ignore_index=True)
    res.to_csv(OUT_DIR / "price_elasticity.csv", index=False)
    print(f"PRIMARY variant uses {kept:.0%} of all products")
    print(res.to_string(index=False))
    print(f"\nBreak-even elasticity at {ASSUMED_MARGIN:.0%} margin: {breakeven_elasticity(ASSUMED_MARGIN):.2f}")
    for e in (-0.5, -1.0, -2.0, -4.0):
        d, pr = best_discount(100, 10, e, ASSUMED_MARGIN)
        base = simulate(100, 10, 0, e, ASSUMED_MARGIN)["profit"]
        print(f"  elasticity {e:>5}: profit-best discount {d:.0%}  (profit {pr:.0f} vs {base:.0f} at no discount)")


if __name__ == "__main__":
    main()
