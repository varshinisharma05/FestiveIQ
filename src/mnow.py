"""Stage 5: which products deserve scarce fast-delivery (M-Now style) shelf space?

A dark store holds only a few thousand products, so each slot must earn its place.
We score every product on four things we can measure from sales data:

  velocity      units sold per week                         (sells fast)
  consistency   share of weeks with at least one sale       (sells steadily, not one big order)
  low returns   1 - returned units / sold units             (customers keep it)
  revenue       revenue within its category                 (worth the space)

score = 0.35*velocity_rank + 0.25*consistency + 0.25*(1 - return_rate) + 0.15*revenue_rank
(ranks are percentile ranks inside the category, 0..1). The weights are a judgement call,
not learned, which is why the formula is shown on the dashboard.

BACKTEST FINDING (see backtest()): this composite score did NOT beat the simple rule
"top products by past units per week", so the default picks use the simple rule and the
score is kept only as a documented alternative. Run:  python -m src.mnow
"""
import sqlite3
import numpy as np
import pandas as pd
from src.config import DB_PATH, OUT_DIR, LAST_COMPLETE_WEEK_END

WEIGHTS = {"velocity": 0.35, "consistency": 0.25, "low_returns": 0.25, "revenue": 0.15}
TOP_N = 10


def _window(con, start, end):
    """Per-SKU stats for weeks in [start, end)."""
    q = """
    SELECT s.stockcode, c.category, p.description,
           SUM(s.quantity) AS units, SUM(s.revenue) AS revenue,
           COUNT(DISTINCT s.week_start) AS active_weeks
    FROM sales s
    JOIN product_category c USING (stockcode)
    JOIN products p USING (stockcode)
    WHERE s.week_start >= ? AND s.week_start < ?
    GROUP BY s.stockcode"""
    sales = pd.read_sql_query(q, con, params=(start, end))
    ret = pd.read_sql_query(
        "SELECT stockcode, SUM(quantity) AS returned_units FROM returns "
        "WHERE week_start >= ? AND week_start < ? GROUP BY stockcode", con, params=(start, end))
    return sales.merge(ret, on="stockcode", how="left").fillna({"returned_units": 0})


def score(as_of: str, lookback_weeks: int = 52, con=None) -> pd.DataFrame:
    """Score every product using only data before `as_of`."""
    own = con is None
    con = con or sqlite3.connect(DB_PATH)
    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(weeks=lookback_weeks)
    d = _window(con, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if own:
        con.close()
    d["units_per_week"] = d["units"] / lookback_weeks
    d["consistency"] = d["active_weeks"] / lookback_weeks
    d["return_rate"] = (d["returned_units"] / d["units"]).clip(0, 1)
    g = d.groupby("category")
    d["velocity_rank"] = g["units_per_week"].rank(pct=True)
    d["revenue_rank"] = g["revenue"].rank(pct=True)
    d["score"] = (WEIGHTS["velocity"] * d["velocity_rank"] + WEIGHTS["consistency"] * d["consistency"]
                  + WEIGHTS["low_returns"] * (1 - d["return_rate"]) + WEIGHTS["revenue"] * d["revenue_rank"])
    # eligibility: must have sold in at least a quarter of the weeks (filters one-off bulk buys)
    d["eligible"] = d["consistency"] >= 0.25
    return d


def pick(d: pd.DataFrame, n: int = TOP_N, rule: str = "score") -> pd.DataFrame:
    """rule='units' = top products by units per week (the rule that won the backtest, see README);
    rule='score'  = the composite score described at the top of this file."""
    key = "units_per_week" if rule == "units" else "score"
    e = d[d["eligible"]].sort_values(["category", key], ascending=[True, False])
    e["rank_in_category"] = e.groupby("category").cumcount() + 1
    return e[e["rank_in_category"] <= n]


def backtest(as_of="2011-10-03", end=None, n=TOP_N):
    """Pick with data before `as_of`; judge on the weeks from `as_of` to `end` (the festive window).
    Compare with the naive rule 'just take the top-n by past revenue'."""
    end = end or (pd.Timestamp(LAST_COMPLETE_WEEK_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    con = sqlite3.connect(DB_PATH)
    d = score(as_of, con=con)
    future = _window(con, as_of, end).rename(columns={"units": "f_units", "revenue": "f_revenue",
                                                       "active_weeks": "f_active_weeks",
                                                       "returned_units": "f_returned"})
    con.close()
    n_weeks = future["f_active_weeks"].max()
    fut = future.set_index("stockcode")
    cat_rev = future.groupby("category")["f_revenue"].sum()
    total_units = future["f_units"].sum()

    def evaluate(picked: pd.DataFrame, label: str):
        f = fut.reindex(picked["stockcode"]).fillna(0)
        f["category"] = picked["category"].values
        cover = f.groupby("category")["f_revenue"].sum().reindex(cat_rev.index).fillna(0).sum() / cat_rev.sum()
        units = f["f_units"].sum()
        return {"selection_rule": label, "products_picked": len(picked),
                "festive_revenue_captured_pct": round(100 * cover, 1),
                "festive_units_captured_pct": round(100 * units / total_units, 1),
                "festive_return_rate_pct": round(100 * f["f_returned"].sum() / max(units, 1), 2),
                "still_selling_pct": round(100 * (f["f_active_weeks"] >= 0.5 * n_weeks).mean(), 1)}

    mine = pick(d, n, "score")
    elig = d[d["eligible"]].sort_values(["category", "revenue"], ascending=[True, False]).copy()
    elig["r"] = elig.groupby("category").cumcount() + 1
    rev_only = elig[elig["r"] <= n]
    u_only = d[d["eligible"]].sort_values(["category", "units_per_week"], ascending=[True, False]).copy()
    u_only["r"] = u_only.groupby("category").cumcount() + 1
    # Rule added AFTER seeing the first backtest (so its score is optimistic): rank by units,
    # but only among products that sell steadily and are rarely returned.
    g = d[d["eligible"] & (d["consistency"] >= 0.5) & (d["return_rate"] <= 0.05)]
    g = g.sort_values(["category", "units_per_week"], ascending=[True, False]).copy()
    g["r"] = g.groupby("category").cumcount() + 1
    res = pd.DataFrame([evaluate(mine, "M-Now score (velocity, consistency, low returns, revenue)"),
                        evaluate(rev_only, "top by past revenue only"),
                        evaluate(u_only[u_only["r"] <= n], "top by past units only"),
                        evaluate(g[g["r"] <= n], "units rank + guardrails (steady >=50% of weeks, returns <=5%) [added after first look]")])
    return res


def main():
    final = score(LAST_COMPLETE_WEEK_END)
    picks = pick(final, rule="units")   # default = simple rule that won the backtest
    picks[["category", "rank_in_category", "stockcode", "description", "units_per_week", "consistency",
           "return_rate", "revenue", "score"]].round(3).to_csv(OUT_DIR / "mnow_picks.csv", index=False)
    print(picks.groupby("category").head(2)[["category", "description", "units_per_week", "consistency",
                                              "return_rate", "score"]].round(2).to_string(index=False))
    bt = backtest()
    bt.to_csv(OUT_DIR / "mnow_backtest.csv", index=False)
    print("\n=== Backtest: picked with data before 3 Oct 2011, judged on 3 Oct - 4 Dec 2011 ===")
    print(bt.to_string(index=False))


if __name__ == "__main__":
    main()
