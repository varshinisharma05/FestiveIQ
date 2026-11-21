"""Stage 1: clean the raw CSV and load it into SQLite.

Run:  python -m src.clean_load
Every row removed is counted and written to outputs/cleaning_log.csv,
so each cleaning rule can be defended with a number.
"""
import sqlite3
import pandas as pd
from src.config import RAW_CSV, DB_PATH, OUT_DIR

PRODUCT_CODE_RE = r"^\d{5}[A-Za-z]{0,2}$"   # real products look like 85048 or 79323P
MEGA_QTY = 10000   # a cancellation this large is checked against a matching sale


def clean(df: pd.DataFrame):
    """Return (sales, returns, log). Pure function, so it is easy to unit-test."""
    log = []

    def step(name, before, after):
        log.append({"step": name, "rows_before": before, "rows_removed": before - after,
                    "rows_after": after})

    df = df.rename(columns={"Invoice": "invoice", "StockCode": "stockcode",
                            "Description": "description", "Quantity": "quantity",
                            "InvoiceDate": "invoice_date", "Price": "price",
                            "Customer ID": "customer_id", "Country": "country"})
    df["invoice"] = df["invoice"].astype(str)
    df["stockcode"] = df["stockcode"].astype(str).str.strip().str.upper()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    n = len(df)
    log.append({"step": "raw rows", "rows_before": n, "rows_removed": 0, "rows_after": n})

    # 1. exact duplicate lines
    d = df.drop_duplicates()
    step("drop exact duplicate rows", n, len(d)); df, n = d, len(d)

    # 2. non-product codes (postage, bank charges, manual adjustments, samples...)
    d = df[df["stockcode"].str.match(PRODUCT_CODE_RE)]
    step("drop non-product stock codes (POST, DOT, M, BANK CHARGES...)", n, len(d)); df, n = d, len(d)

    # 3. zero / negative price (stock write-offs, 'adjust bad debt')
    d = df[df["price"] > 0]
    step("drop rows with price <= 0", n, len(d)); df, n = d, len(d)

    df = df.copy()
    df["is_cancel"] = df["invoice"].str.startswith("C")

    # 4. negative quantity that is NOT a cancellation = stock adjustment, not a sale
    d = df[(df["is_cancel"]) | (df["quantity"] > 0)]
    step("drop negative-quantity rows that are not cancellations", n, len(d)); df = d

    returns = df[df["is_cancel"]].copy()
    returns["quantity"] = returns["quantity"].abs()
    sales = df[~df["is_cancel"]].copy()

    # 5. mistaken mega-orders: a huge sale that is cancelled within a day, same
    #    customer, same product, same quantity. This is an entry error, not demand.
    #    (e.g. 80,995 units of one product ordered and cancelled 12 minutes later)
    big = returns[returns["quantity"] >= MEGA_QTY]
    drop_sale_idx, drop_ret_idx = [], []
    for ridx, r in big.iterrows():
        m = sales[(sales["stockcode"] == r["stockcode"]) & (sales["customer_id"] == r["customer_id"])
                  & (sales["quantity"] == r["quantity"])
                  & ((r["invoice_date"] - sales["invoice_date"]).abs() <= pd.Timedelta(days=1))]
        if len(m):
            drop_sale_idx += list(m.index)
            drop_ret_idx.append(ridx)
    before = len(sales) + len(returns)
    sales = sales.drop(index=drop_sale_idx)
    returns = returns.drop(index=drop_ret_idx)
    log.append({"step": "drop mistaken mega-order + cancellation pairs",
                "rows_before": before, "rows_removed": before - len(sales) - len(returns),
                "rows_after": len(sales) + len(returns)})

    for t in (sales, returns):
        t["revenue"] = t["quantity"] * t["price"]
        t["week_start"] = (t["invoice_date"].dt.normalize()
                           - pd.to_timedelta(t["invoice_date"].dt.weekday, unit="D"))
        t["month"] = t["invoice_date"].dt.strftime("%Y-%m")
    log.append({"step": "final: sales lines kept", "rows_before": len(sales) + len(returns),
                "rows_removed": len(returns), "rows_after": len(sales)})
    log.append({"step": "final: return lines kept (separate table)",
                "rows_before": len(sales) + len(returns),
                "rows_removed": len(sales), "rows_after": len(returns)})
    return sales, returns, pd.DataFrame(log)


def product_table(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode with its most frequent description."""
    s = sales.dropna(subset=["description"]).copy()
    s["description"] = s["description"].str.strip().str.upper()
    mode = (s.groupby("stockcode")["description"]
            .agg(lambda x: x.value_counts().index[0]).rename("description"))
    return mode.reset_index()


def main():
    df = pd.read_csv(RAW_CSV)
    sales, returns, log = clean(df)
    products = product_table(sales)
    for t in (sales, returns):
        t["invoice_date"] = t["invoice_date"].astype(str)
        t["week_start"] = t["week_start"].dt.strftime("%Y-%m-%d")
    cols = ["invoice", "stockcode", "quantity", "invoice_date", "price",
            "customer_id", "country", "revenue", "week_start", "month"]
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    sales[cols].to_sql("sales", con, index=False)
    returns[cols].to_sql("returns", con, index=False)
    products.to_sql("products", con, index=False)
    con.executescript("""
        CREATE INDEX idx_sales_sku ON sales(stockcode);
        CREATE INDEX idx_sales_week ON sales(week_start);
        CREATE INDEX idx_sales_cust ON sales(customer_id);
        CREATE INDEX idx_ret_sku ON returns(stockcode);
    """)
    con.commit()
    con.close()
    log.to_csv(OUT_DIR / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nproducts: {len(products)}  ->  {DB_PATH}")


if __name__ == "__main__":
    main()
