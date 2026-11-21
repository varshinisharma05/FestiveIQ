"""Stage 7b: auto-generated weekly brief for a category manager.

Python computes every fact (build_facts). The writer (template or Gemini) only phrases them.
check_numbers() then verifies that every number in the text exists in the facts; if the LLM
invented or altered a number, the brief is rejected and we fall back to the template version.

Run:  python -m src.brief
"""
import json
import os
import re
import sqlite3
import pandas as pd
from src.config import DB_PATH, OUT_DIR, ASSUMED_MARGIN, LAST_COMPLETE_WEEK_END


def build_facts(as_of_end=LAST_COMPLETE_WEEK_END, weeks=4, db_path=DB_PATH) -> dict:
    end = pd.Timestamp(as_of_end)
    start = end - pd.Timedelta(days=7 * weeks - 1)             # Monday of the first week
    ly_start, ly_end = start - pd.Timedelta(weeks=52), end - pd.Timedelta(weeks=52)
    f = lambda t: t.strftime("%Y-%m-%d")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = lambda sql, p: pd.read_sql_query(sql, con, params=p)

    tot = q("SELECT ROUND(SUM(revenue),0) AS r FROM sales WHERE week_start>=? AND week_start<=?", (f(start), f(end)))["r"][0]
    ly = q("SELECT ROUND(SUM(revenue),0) AS r FROM sales WHERE week_start>=? AND week_start<=?", (f(ly_start), f(ly_end)))["r"][0]
    cats = q("""SELECT c.category, ROUND(SUM(s.revenue),0) AS revenue FROM sales s
                JOIN product_category c USING(stockcode)
                WHERE s.week_start>=? AND s.week_start<=? GROUP BY c.category ORDER BY revenue DESC LIMIT 3""",
             (f(start), f(end)))
    cat_ly = q("""SELECT c.category, ROUND(SUM(s.revenue),0) AS revenue FROM sales s
                  JOIN product_category c USING(stockcode)
                  WHERE s.week_start>=? AND s.week_start<=? GROUP BY c.category""", (f(ly_start), f(ly_end)))
    ly_map = dict(zip(cat_ly["category"], cat_ly["revenue"]))
    top_cats = [{"category": r.category, "revenue": float(r.revenue),
                 "yoy_pct": round(100 * (r.revenue - ly_map.get(r.category, 0)) / ly_map[r.category], 1)
                 if ly_map.get(r.category) else None} for r in cats.itertuples()]
    ret = q("""SELECT c.category, ROUND(100.0*SUM(r.revenue)/
                 (SELECT SUM(s.revenue) FROM sales s JOIN product_category c2 USING(stockcode)
                   WHERE c2.category=c.category AND s.week_start>=? AND s.week_start<=?),1) AS return_rate_pct
               FROM returns r JOIN product_category c USING(stockcode)
               WHERE r.week_start>=? AND r.week_start<=? GROUP BY c.category
               ORDER BY return_rate_pct DESC LIMIT 1""", (f(start), f(end), f(start), f(end)))
    top_sku = q("""SELECT p.description, SUM(s.quantity) AS units FROM sales s JOIN products p USING(stockcode)
                   WHERE s.week_start>=? AND s.week_start<=? GROUP BY s.stockcode ORDER BY units DESC LIMIT 1""",
                (f(start), f(end)))
    con.close()
    return {
        "period": f"{f(start)} to {f(end)}", "weeks": weeks,
        "revenue": float(tot), "revenue_same_weeks_last_year": float(ly),
        "yoy_pct": round(100 * (tot - ly) / ly, 1),
        "top_categories": top_cats,
        "highest_return_rate_category": {"category": ret["category"][0], "return_rate_pct": float(ret["return_rate_pct"][0])},
        "best_selling_product": {"name": top_sku["description"][0], "units": int(top_sku["units"][0])},
        "assumed_margin_pct": round(ASSUMED_MARGIN * 100),
    }


def template_brief(f: dict) -> str:
    c = f["top_categories"]
    lines = [
        f"Revenue for {f['period']} was {f['revenue']:,.0f}, {f['yoy_pct']}% versus {f['revenue_same_weeks_last_year']:,.0f} in the same weeks a year earlier.",
        "Top categories: " + "; ".join(
            f"{x['category']} ({x['revenue']:,.0f}" + (f", {x['yoy_pct']}% year on year)" if x["yoy_pct"] is not None else ")") for x in c) + ".",
        f"The best-selling product was {f['best_selling_product']['name']} with {f['best_selling_product']['units']:,} units.",
        f"Watch returns in {f['highest_return_rate_category']['category']}, which had the highest return rate at {f['highest_return_rate_category']['return_rate_pct']}%.",
    ]
    return "\n".join(lines)


def llm_brief(f: dict) -> str:
    from dotenv import load_dotenv
    from google import genai
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    prompt = ("Write a 4-6 sentence weekly brief for a retail category manager using ONLY the facts in this JSON. "
              "Copy every number exactly as given (do not round, convert or compute new numbers). "
              "Add one practical suggestion that does not need any new number.\n\n" + json.dumps(f, indent=1))
    return client.models.generate_content(model=model, contents=prompt).text.strip()


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text: str):
    out = []
    for t in _NUM.findall(text):
        try:
            out.append(float(t.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def check_numbers(text: str, facts: dict, tol=0.005):
    """Return the numbers in `text` that do not appear in `facts` (within a small tolerance)."""
    allowed = _numbers(json.dumps(facts)) + [2009, 2010, 2011, 2012]
    bad = []
    for n in _numbers(text):
        if not any(abs(n - a) <= tol * max(abs(a), 1) for a in allowed):
            bad.append(n)
    return bad


def make_brief():
    """Return (text, source, unverified_numbers). Falls back to the template if no key or numbers fail."""
    facts = build_facts()
    if os.environ.get("GEMINI_API_KEY") or _env_has_key():
        try:
            text = llm_brief(facts)
            bad = check_numbers(text, facts)
            if not bad:
                return text, "Gemini (all numbers verified against computed facts)", []
            return template_brief(facts), f"template (Gemini text rejected, unverified numbers: {bad})", bad
        except Exception as e:
            return template_brief(facts), f"template (Gemini unavailable: {type(e).__name__})", []
    return template_brief(facts), "template (no GEMINI_API_KEY set)", []


def _env_has_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.environ.get("GEMINI_API_KEY"))
    except Exception:
        return False


if __name__ == "__main__":
    facts = build_facts()
    (OUT_DIR / "weekly_facts.json").write_text(json.dumps(facts, indent=1))
    text, source, bad = make_brief()
    print(f"[{source}]\n{text}")
