"""Stage 2: the business metrics, written in plain SQL.

Run:  python -m src.queries      (saves CSVs into outputs/)
Definitions used everywhere:
  revenue      = quantity * price on sales lines (gross, before returns)
  return rate  = returned revenue / sales revenue
  repeat rate  = share of identified customers with 2+ distinct invoices
"""
import sqlite3
import pandas as pd
from src.config import DB_PATH, OUT_DIR

QUERIES = {
    # One-row summary for the dashboard header
    "kpis": """
        SELECT
          ROUND(SUM(revenue), 0)                                   AS total_revenue,
          COUNT(DISTINCT invoice)                                  AS orders,
          ROUND(SUM(revenue) * 1.0 / COUNT(DISTINCT invoice), 2)   AS avg_order_value,
          COUNT(DISTINCT customer_id)                              AS identified_customers,
          (SELECT ROUND(100.0 * SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) / COUNT(*), 1)
             FROM (SELECT customer_id, COUNT(DISTINCT invoice) AS n
                   FROM sales WHERE customer_id IS NOT NULL GROUP BY customer_id)) AS repeat_rate_pct,
          (SELECT ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM sales), 2) FROM returns)
                                                                   AS return_rate_pct
        FROM sales
    """,

    # Monthly revenue with month-over-month growth (window function LAG)
    "monthly_revenue": """
        WITH m AS (
          SELECT month, ROUND(SUM(revenue), 0) AS revenue,
                 COUNT(DISTINCT invoice) AS orders,
                 ROUND(SUM(revenue) * 1.0 / COUNT(DISTINCT invoice), 2) AS aov
          FROM sales GROUP BY month)
        SELECT month, revenue, orders, aov,
               ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
                     / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
        FROM m ORDER BY month
    """,

    "top_products": """
        SELECT s.stockcode, p.description,
               SUM(s.quantity) AS units, ROUND(SUM(s.revenue), 0) AS revenue
        FROM sales s JOIN products p USING (stockcode)
        GROUP BY s.stockcode ORDER BY revenue DESC LIMIT 20
    """,

    "revenue_by_country": """
        SELECT country, ROUND(SUM(revenue), 0) AS revenue,
               ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM sales), 1) AS share_pct
        FROM sales GROUP BY country ORDER BY revenue DESC LIMIT 10
    """,

    # Products with high return rates (only products with meaningful sales)
    "return_rate_by_product": """
        SELECT s.stockcode, p.description,
               ROUND(SUM(s.revenue), 0) AS sales_revenue,
               ROUND(COALESCE(r.ret_rev, 0), 0) AS returned_revenue,
               ROUND(100.0 * COALESCE(r.ret_rev, 0) / SUM(s.revenue), 1) AS return_rate_pct
        FROM sales s
        JOIN products p USING (stockcode)
        LEFT JOIN (SELECT stockcode, SUM(revenue) AS ret_rev FROM returns GROUP BY stockcode) r
               USING (stockcode)
        GROUP BY s.stockcode
        HAVING SUM(s.revenue) > 5000
        ORDER BY return_rate_pct DESC LIMIT 20
    """,

    # How much of each year's revenue arrives in the festive quarter (Oct-Dec)
    "festive_share": """
        SELECT CASE WHEN month < '2010-10' THEN 'before Oct 2010'
                    WHEN month < '2011-01' THEN 'Oct-Dec 2010'
                    WHEN month < '2011-10' THEN 'Jan-Sep 2011'
                    ELSE 'Oct-Dec 2011 (to 9 Dec)' END AS period,
               ROUND(SUM(revenue), 0) AS revenue
        FROM sales GROUP BY period ORDER BY MIN(month)
    """,
}


def run(name: str, con=None) -> pd.DataFrame:
    own = con is None
    con = con or sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(QUERIES[name], con)
    finally:
        if own:
            con.close()


def main():
    con = sqlite3.connect(DB_PATH)
    for name in QUERIES:
        df = run(name, con)
        df.to_csv(OUT_DIR / f"metric_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(df.head(12).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
