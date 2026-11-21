"""Stage 7a: "Ask the Analyst": an English question -> SQL -> answer, with guardrails.

Design rule: the AI NEVER does the maths. It only WRITES a SQL query. SQLite runs it.
Guardrails (all in this file, all unit-tested):
  1. validate_sql(): one single SELECT/WITH statement, no write/DDL keywords.
  2. The database is opened read-only (mode=ro), so even a bug cannot change data.
  3. Results are capped at 200 rows.
  4. The generated SQL is shown to the user next to the answer.

Run the accuracy test (needs GEMINI_API_KEY):  python -m src.analyst
The 15 test questions have golden SQL written by hand; an answer counts as correct when every
value of the golden result appears in the model's result.
"""
import os
import re
import sqlite3
import pandas as pd
from src.config import DB_PATH, OUT_DIR

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b", re.I)
MAX_ROWS = 200

SCHEMA_NOTES = """Tables (SQLite):
sales(invoice, stockcode, quantity, invoice_date 'YYYY-MM-DD HH:MM:SS', price, customer_id (may be NULL), country, revenue, week_start 'YYYY-MM-DD' (Monday), month 'YYYY-MM')
returns(same columns as sales; quantity is positive; these are cancelled/returned lines)
products(stockcode, description)
product_category(stockcode, category, method)  -- categories: Christmas & seasonal, Lighting & candles, Jewellery & accessories, Party & wrapping, Stationery & paper, Toys & games, Garden & outdoor, Bath & personal care, Kitchen & dining, Bags & storage, Home decor, Gifts & novelties
Definitions: revenue = SUM(sales.revenue). return rate = SUM(returns.revenue) / SUM(sales.revenue).
Data covers 2009-12-01 to 2011-12-09. The last week (from 2011-12-05) is incomplete."""

EXAMPLES = """Q: What was total revenue in March 2010?
SQL: SELECT ROUND(SUM(revenue),0) AS revenue FROM sales WHERE month = '2010-03'
Q: Top 3 countries by revenue?
SQL: SELECT country, ROUND(SUM(revenue),0) AS revenue FROM sales GROUP BY country ORDER BY revenue DESC LIMIT 3"""


def validate_sql(sql: str) -> str:
    """Return a safe version of the SQL or raise ValueError."""
    s = sql.strip().strip("`").strip()
    s = re.sub(r"^sql\s*\n", "", s, flags=re.I).strip()
    s = s.rstrip(";").strip()
    if ";" in s:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise ValueError("Query contains a forbidden keyword.")
    return s


def run_readonly(sql: str, db_path=DB_PATH) -> pd.DataFrame:
    safe = validate_sql(sql)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(f"SELECT * FROM ({safe}) LIMIT {MAX_ROWS}", con)
    finally:
        con.close()


def generate_sql(question: str) -> str:
    """Ask Gemini to write the SQL. Needs `pip install google-genai` and GEMINI_API_KEY in .env."""
    from dotenv import load_dotenv
    from google import genai
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")   # check the current model name
    prompt = (f"You write SQLite queries for a retail analytics database.\n{SCHEMA_NOTES}\n\n"
              f"Examples:\n{EXAMPLES}\n\nReturn ONLY one SQLite SELECT query, no explanation, no markdown.\n"
              f"Q: {question}\nSQL:")
    return client.models.generate_content(model=model, contents=prompt).text


def ask(question: str):
    sql = validate_sql(generate_sql(question))
    return sql, run_readonly(sql)


# ----------------------------------------------------------------- 15 test questions + golden SQL
GOLDEN = [
    ("What is the total revenue?", "SELECT ROUND(SUM(revenue),0) FROM sales"),
    ("How many orders were placed in total?", "SELECT COUNT(DISTINCT invoice) FROM sales"),
    ("Which 3 countries have the highest revenue?", "SELECT country FROM sales GROUP BY country ORDER BY SUM(revenue) DESC LIMIT 3"),
    ("Which month had the highest revenue?", "SELECT month FROM sales GROUP BY month ORDER BY SUM(revenue) DESC LIMIT 1"),
    ("What was the average order value in November 2011?", "SELECT ROUND(SUM(revenue)*1.0/COUNT(DISTINCT invoice),2) FROM sales WHERE month='2011-11'"),
    ("Which product sold the most units?", "SELECT p.description FROM sales s JOIN products p USING(stockcode) GROUP BY s.stockcode ORDER BY SUM(s.quantity) DESC LIMIT 1"),
    ("Which category has the highest revenue?", "SELECT c.category FROM sales s JOIN product_category c USING(stockcode) GROUP BY c.category ORDER BY SUM(s.revenue) DESC LIMIT 1"),
    ("What is the overall return rate as a percentage of revenue?", "SELECT ROUND(100.0*(SELECT SUM(revenue) FROM returns)/(SELECT SUM(revenue) FROM sales),2)"),
    ("How many identified customers ordered in 2011?", "SELECT COUNT(DISTINCT customer_id) FROM sales WHERE invoice_date >= '2011-01-01' AND customer_id IS NOT NULL"),
    ("What was the revenue in Germany?", "SELECT ROUND(SUM(revenue),0) FROM sales WHERE country='Germany'"),
    ("Which week had the highest revenue?", "SELECT week_start FROM sales GROUP BY week_start ORDER BY SUM(revenue) DESC LIMIT 1"),
    ("Which product earned the most revenue in the Christmas & seasonal category?", "SELECT p.description FROM sales s JOIN product_category c USING(stockcode) JOIN products p USING(stockcode) WHERE c.category='Christmas & seasonal' GROUP BY s.stockcode ORDER BY SUM(s.revenue) DESC LIMIT 1"),
    ("How many customers placed more than one order?", "SELECT COUNT(*) FROM (SELECT customer_id FROM sales WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING COUNT(DISTINCT invoice) > 1)"),
    ("Which category has the highest return rate by revenue?", "SELECT c.category FROM product_category c JOIN sales s USING(stockcode) LEFT JOIN (SELECT pc.category, SUM(r.revenue) AS rr FROM returns r JOIN product_category pc USING(stockcode) GROUP BY pc.category) x ON x.category=c.category GROUP BY c.category ORDER BY COALESCE(MAX(x.rr),0)/SUM(s.revenue) DESC LIMIT 1"),
    ("What was total revenue in December 2010?", "SELECT ROUND(SUM(revenue),0) FROM sales WHERE month='2010-12'"),
]


def _cells(df: pd.DataFrame) -> list:
    out = []
    for v in df.to_numpy().ravel():
        try:
            out.append(round(float(v), 1))
        except (TypeError, ValueError):
            out.append(str(v).strip().lower())
    return out


def answer_matches(golden: pd.DataFrame, got: pd.DataFrame) -> bool:
    """Correct when every golden value appears somewhere in the model's result."""
    g, m = _cells(golden), _cells(got)
    return all(x in m for x in g)


def evaluate():
    ok, rows = 0, []
    for q, gold in GOLDEN:
        g = run_readonly(gold)
        try:
            sql, got = ask(q)
            good = answer_matches(g, got)
        except Exception as e:   # a failed or unsafe query counts as wrong
            sql, good = f"ERROR: {e}", False
        ok += good
        rows.append({"question": q, "correct": good, "generated_sql": sql})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "analyst_eval.csv", index=False)
    print(df[["question", "correct"]].to_string(index=False))
    print(f"\nAccuracy: {ok}/{len(GOLDEN)}")


if __name__ == "__main__":
    evaluate()
