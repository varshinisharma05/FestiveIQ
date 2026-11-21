"""Unit tests for the parts that must never silently break. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.clean_load import clean
from src.analyst import validate_sql, answer_matches
from src.categorize import rule_category, parse_llm_json, CATEGORIES
from src.pricing import simulate, breakeven_elasticity, best_discount
from src.forecast import make_features, wape
from src.brief import check_numbers


def raw(rows):
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- cleaning
def test_clean_removes_duplicates_nonproducts_zero_price():
    df = raw([
        ["1", "85048", "A", 2, "2011-01-03 10:00:00", 1.0, 1.0, "UK"],
        ["1", "85048", "A", 2, "2011-01-03 10:00:00", 1.0, 1.0, "UK"],      # exact duplicate
        ["2", "POST", "Postage", 1, "2011-01-03 10:00:00", 5.0, 1.0, "UK"],  # not a product
        ["3", "85049", "B", 1, "2011-01-03 10:00:00", 0.0, 1.0, "UK"],      # price 0
    ])
    sales, returns, log = clean(df)
    assert len(sales) == 1 and len(returns) == 0


def test_clean_splits_returns_and_makes_quantity_positive():
    df = raw([["10", "85048", "A", 5, "2011-01-03 10:00:00", 2.0, 1.0, "UK"],
              ["C11", "85048", "A", -2, "2011-01-04 10:00:00", 2.0, 1.0, "UK"]])
    sales, returns, _ = clean(df)
    assert len(sales) == 1 and len(returns) == 1 and returns["quantity"].iloc[0] == 2


def test_clean_drops_mega_order_cancelled_minutes_later():
    df = raw([["20", "23843", "X", 80995, "2011-12-09 09:15:00", 2.0, 16446.0, "UK"],
              ["C21", "23843", "X", -80995, "2011-12-09 09:27:00", 2.0, 16446.0, "UK"],
              ["22", "85048", "A", 3, "2011-12-09 10:00:00", 2.0, 1.0, "UK"]])
    sales, returns, _ = clean(df)
    assert len(sales) == 1 and len(returns) == 0 and sales["stockcode"].iloc[0] == "85048"


# ---------------------------------------------------------------- SQL guardrails
@pytest.mark.parametrize("bad", ["DROP TABLE sales", "DELETE FROM sales", "select 1; select 2",
                                 "UPDATE sales SET price=0", "PRAGMA table_info(sales)",
                                 "SELECT 1; DROP TABLE sales"])
def test_validate_sql_blocks_unsafe(bad):
    with pytest.raises(ValueError):
        validate_sql(bad)


def test_validate_sql_allows_select_and_with():
    assert validate_sql("SELECT 1;").lower().startswith("select")
    assert validate_sql("WITH a AS (SELECT 1) SELECT * FROM a").lower().startswith("with")


def test_answer_matches_checks_values_not_column_names():
    gold = pd.DataFrame({"x": [1234.0]})
    assert answer_matches(gold, pd.DataFrame({"total": [1234.04], "other": ["a"]}))
    assert not answer_matches(gold, pd.DataFrame({"total": [999.0]}))


# ---------------------------------------------------------------- categorizer
def test_rule_categories():
    assert rule_category("PAPER CHAIN KIT 50'S CHRISTMAS") == "Christmas & seasonal"
    assert rule_category("PLAYING CARDS KEEP CALM") == "Toys & games"
    assert rule_category("JUMBO BAG RED RETROSPOT") == "Bags & storage"
    assert rule_category("WHITE HANGING HEART T-LIGHT HOLDER") == "Lighting & candles"
    assert rule_category("SOMETHING UNKNOWN") == "Gifts & novelties"


def test_parse_llm_json_handles_fences_and_invalid_categories():
    text = '```json\n{"0": "Toys & games", "1": "Not a category"}\n```'
    out = parse_llm_json(text, 3)
    assert out[0] == "Toys & games" and out[1] is None and out[2] is None
    assert parse_llm_json("garbage", 2) == {0: None, 1: None}
    assert all(c in CATEGORIES for c in ["Toys & games"])


# ---------------------------------------------------------------- pricing maths
def test_simulate_no_discount_is_identity():
    s = simulate(100, 10, 0.0, -2.0, 0.3)
    assert s["units"] == pytest.approx(100) and s["profit"] == pytest.approx(300)


def test_breakeven_elasticity_matches_best_discount():
    m = 0.3
    be = breakeven_elasticity(m)                       # -3.33
    assert best_discount(100, 10, be + 0.5, m)[0] == 0.0   # less sensitive than break-even: no discount
    assert best_discount(100, 10, be - 1.0, m)[0] > 0.0    # more sensitive: a discount pays


# ---------------------------------------------------------------- forecasting: no peeking
def test_features_do_not_use_current_or_future_weeks():
    weeks = pd.date_range("2010-01-04", periods=120, freq="W-MON")
    rng = np.random.default_rng(0)
    wk = pd.DataFrame({"category": "A", "week_start": weeks, "units": rng.integers(10, 100, 120).astype(float),
                       "revenue": 1.0, "orders": 1.0, "avg_price": 2.0})
    h = 4
    base = make_features(wk, h).set_index("week_start")
    t = weeks[100]
    wk2 = wk.copy()
    wk2.loc[wk2["week_start"] >= weeks[100 - h + 1], "units"] = 9999.0   # change weeks t-h+1 and later
    changed = make_features(wk2, h).set_index("week_start")
    cols = ["lag_h", "ma4", "ly", "yoy_ratio"]
    assert np.allclose(base.loc[t, cols].astype(float), changed.loc[t, cols].astype(float))


def test_wape():
    assert wape(np.array([10, 10]), np.array([8, 12])) == pytest.approx(0.2)


# ---------------------------------------------------------------- brief number check
def test_check_numbers_flags_invented_numbers():
    facts = {"revenue": 1342652.0, "yoy_pct": 1.1, "period": "2011-11-07 to 2011-12-04"}
    assert check_numbers("Revenue was 1,342,652, up 1.1%.", facts) == []
    assert check_numbers("Revenue was 1,342,652, up 5.7%.", facts) == [5.7]
