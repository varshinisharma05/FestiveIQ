"""FestiveIQ dashboard.  Run:  streamlit run app.py
Run Stages 1-6 first (see README) so data/retail.db and outputs/*.csv exist."""
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import zipfile
from src.config import DB_PATH, DATA_DIR, OUT_DIR, ASSUMED_MARGIN
from src import queries, pricing
from src.analyst import GOLDEN, run_readonly, ask
from src.brief import build_facts, make_brief

st.set_page_config(page_title="FestiveIQ", layout="wide")
st.title("FestiveIQ: festive-sale demand, pricing and assortment copilot")
st.caption("Built on a public UK gift-ware retail dataset (UCI Online Retail II) as a stand-in for fashion "
           "e-commerce data. Numbers are not Myntra's numbers.")

if not DB_PATH.exists():
    zip_path = DATA_DIR / "retail_db.zip"
    if zip_path.exists():
        with st.spinner("Preparing analytics database on first run..."):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
    else:
        st.error("data/retail.db not found. Put online_retail_II.csv in data/ and run: python -m src.clean_load")
        st.stop()


@st.cache_data
def csv(name):
    return pd.read_csv(OUT_DIR / name)


@st.cache_data
def metric(name):
    return queries.run(name)


tabs = st.tabs(["Overview", "Festive Planner", "M-Now Picker", "Pricing Simulator",
                "Ask the Analyst (AI)", "Weekly Brief (AI)"])

# ------------------------------------------------------------------ 1 Overview
with tabs[0]:
    k = metric("kpis").iloc[0]
    c = st.columns(5)
    c[0].metric("Revenue", f"{k.total_revenue:,.0f}")
    c[1].metric("Orders", f"{k.orders:,}")
    c[2].metric("Avg order value", f"{k.avg_order_value:,.0f}")
    c[3].metric("Repeat customers", f"{k.repeat_rate_pct}%")
    c[4].metric("Return rate (revenue)", f"{k.return_rate_pct}%")
    m = metric("monthly_revenue")
    st.plotly_chart(px.bar(m, x="month", y="revenue", title="Monthly revenue: the Oct-Dec festive ramp is clear"),
                    width="stretch")
    a, b = st.columns(2)
    a.subheader("Top products by revenue")
    a.dataframe(metric("top_products"), hide_index=True)
    b.subheader("Revenue by category")
    b.dataframe(csv("category_summary.csv"), hide_index=True)

# ------------------------------------------------------------------ 2 Festive planner
with tabs[1]:
    wk = csv("weekly_by_category.csv")
    wk["week_start"] = pd.to_datetime(wk["week_start"])
    cat = st.selectbox("Category", sorted(wk["category"].unique()), key="fp_cat")
    g = wk[wk["category"] == cat]
    pred = csv("forecast_test_predictions.csv")
    pred["week_start"] = pd.to_datetime(pred["week_start"])
    h = st.radio("Forecast horizon", [1, 4], horizontal=True, format_func=lambda x: f"{x} week(s) ahead")
    p = pred[(pred["category"] == cat) & (pred["horizon_weeks"] == h)]
    fig = go.Figure()
    fig.add_scatter(x=g["week_start"], y=g["units"], name="Actual units")
    fig.add_scatter(x=p["week_start"], y=p["model"], name="Model forecast", mode="lines+markers")
    fig.add_scatter(x=p["week_start"], y=p["baseline_ly"], name="Same week last year", mode="lines+markers")
    fig.update_layout(title=f"{cat}: weekly units (forecast shown for the Oct-Dec 2011 test window)")
    st.plotly_chart(fig, width="stretch")
    st.subheader("When do the festive ramp-up and peak start?")
    st.dataframe(csv("festive_peaks.csv").query("category == @cat"), hide_index=True)
    st.subheader("How good is the forecast? (honest test results)")
    st.dataframe(csv("forecast_results.csv"), hide_index=True)
    st.info("Lower error is better. At 1 week ahead the ML model does NOT beat simple baselines; the blend row "
            "was added after seeing test results, so treat it as optimistic. Only 8 test weeks, so results are noisy.")

# ------------------------------------------------------------------ 3 M-Now picker
with tabs[2]:
    picks = csv("mnow_picks.csv")
    cat = st.selectbox("Category", sorted(picks["category"].unique()), key="mn_cat")
    st.dataframe(picks[picks["category"] == cat].drop(columns=["score"]), hide_index=True)
    st.caption("Rule: top products by units per week over the last 52 weeks, among products that sold in at "
               "least a quarter of the weeks. Check return_rate before stocking a fast-delivery shelf.")
    st.subheader("Backtest: picked with data before 3 Oct 2011, judged on 3 Oct to 4 Dec 2011")
    st.dataframe(csv("mnow_backtest.csv"), hide_index=True)
    st.info("A more complex composite score was tested and lost to the simple units rule, so the simple rule is used.")

# ------------------------------------------------------------------ 4 Pricing simulator
with tabs[3]:
    el = csv("price_elasticity.csv")
    prim = el[el["variant"].str.startswith("PRIMARY")].set_index("category")
    wk = csv("weekly_by_category.csv")
    cat = st.selectbox("Category", sorted(wk["category"].unique()), key="ps_cat")
    recent = wk[wk["category"] == cat].tail(8)
    units0, price0 = float(recent["units"].mean()), float(recent["avg_price"].mean())
    default_e = float(prim.loc[cat, "elasticity"]) if cat in prim.index else float(prim.loc["ALL (pooled)", "elasticity"])
    c1, c2, c3 = st.columns(3)
    e = c1.slider("Price sensitivity (elasticity)", -5.0, -0.2, default_e, 0.05)
    margin = c2.slider("Assumed gross margin", 0.05, 0.60, float(ASSUMED_MARGIN), 0.01)
    disc = c3.slider("Discount", 0.0, 0.5, 0.10, 0.01)
    base = pricing.simulate(units0, price0, 0.0, e, margin)
    sim = pricing.simulate(units0, price0, disc, e, margin)
    d = st.columns(3)
    d[0].metric("Weekly units", f"{sim['units']:,.0f}", f"{100*(sim['units']/base['units']-1):+.1f}%")
    d[1].metric("Revenue", f"{sim['revenue']:,.0f}", f"{100*(sim['revenue']/base['revenue']-1):+.1f}%")
    d[2].metric("Profit", f"{sim['profit']:,.0f}", f"{100*(sim['profit']/base['profit']-1):+.1f}%")
    grid = np.arange(0, 0.51, 0.01)
    curve = pd.DataFrame({"discount": grid, "profit": [pricing.simulate(units0, price0, x, e, margin)["profit"] for x in grid],
                          "revenue": [pricing.simulate(units0, price0, x, e, margin)["revenue"] for x in grid]})
    st.plotly_chart(px.line(curve, x="discount", y=["profit", "revenue"], title="Profit and revenue vs discount"),
                    width="stretch")
    st.warning(f"Scenario tool, not a prediction. Break-even rule: a discount raises profit only if elasticity is below "
               f"{pricing.breakeven_elasticity(margin):.2f} at a {margin:.0%} margin. Cost is assumed (not in the data), "
               "elasticity is an association, and traffic or brand effects of a sale are not modelled.")
    st.dataframe(el, hide_index=True)

# ------------------------------------------------------------------ 5 Ask the analyst
with tabs[4]:
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    st.write("The AI writes the SQL; SQLite computes the answer. Queries are read-only and shown below.")
    if has_key:
        q = st.text_input("Ask a business question", "Which category had the highest revenue in November 2011?")
        if st.button("Ask"):
            try:
                sql, df = ask(q)
                st.code(sql, language="sql")
                st.dataframe(df, hide_index=True)
            except Exception as ex:
                st.error(f"Could not answer safely: {ex}")
    else:
        st.info("No GEMINI_API_KEY found, so free-text questions are off. Showing the 15 test questions with "
                "hand-written SQL instead. Add a key to a .env file to enable the AI.")
        q = st.selectbox("Question", [x[0] for x in GOLDEN])
        sql = dict(GOLDEN)[q]
        st.code(sql, language="sql")
        st.dataframe(run_readonly(sql), hide_index=True)

# ------------------------------------------------------------------ 6 Weekly brief
with tabs[5]:
    if st.button("Generate weekly brief"):
        text, source, bad = make_brief()
        st.write(text)
        st.caption(f"Source: {source}")
    with st.expander("Facts computed by Python (the only numbers the brief may use)"):
        st.json(build_facts())
