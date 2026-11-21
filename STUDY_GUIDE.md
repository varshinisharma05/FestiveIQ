# Study guide: be able to explain every stage

Read each stage's file next to this guide. Rule: do not put a stage on your resume until you can answer its questions without looking.

## Stage 1: clean_load.py
**Why:** raw data lies. Rows that are not real sales would distort every later number.
**Key ideas:** duplicates; non-product codes (POST, BANK CHARGES) are fees, not products; price 0 rows are write-offs;
invoices starting with C are cancellations, kept in a separate `returns` table; one sale cancelled minutes later is an entry error.
**Say:** "I counted every removed row in a cleaning log, and found two mega-orders that would have made January 2011 and December 2011 look like demand spikes."
**Q:** Why not just drop negatives? Because cancellations are information (return rate); I keep them separately.

## Stage 2: queries.py
**Why SQL:** the metrics are group-and-aggregate questions, which SQL is built for, and the database is a single file (SQLite).
**Know:** GROUP BY, JOIN, LEFT JOIN, subquery, and the window function LAG (compares a month with the previous month).
**Definitions:** AOV = revenue / orders. Repeat rate = share of identified customers with 2+ invoices. Return rate = returned revenue / sales revenue.
**Q:** What does LAG do? Looks one row back in the ordered result, so growth = (this month - last month) / last month.

## Stage 3: categorize.py
**Why:** no category column exists, but category-level planning is the whole point.
**How:** ordered keyword rules (first match wins), 12 categories, a default bucket. Optional Gemini path classifies in batches of 50 and falls back to the rules for invalid answers.
**Be honest:** accuracy is unmeasured until you label 50 products. Do that (15 minutes) and quote the real number.
**Q:** Why rules AND an LLM? Rules are free, instant and explainable; the LLM handles odd names; measuring both on the same labelled sample shows which is better.

## Stage 4: forecast.py
**Terms:** weekly demand per category; a *baseline* is a simple method the model must beat; WAPE = total absolute error / total actual units.
**Method:** features come only from weeks at least h weeks earlier (a unit test proves no peeking). Train before 3 Oct 2011, test on the festive weeks.
**Results:** the ML model loses at 1 week and is roughly level at 4 weeks. The blend that scored 18.9% was added after seeing test results.
**Say:** "Simple baselines were hard to beat with two years of data; a blend helped at 4 weeks but I flag that as optimistic."
**Q:** Why not deep learning? About 100 weekly points per category; a simple model is easier to defend and less likely to overfit.

## Stage 5: mnow.py
**Idea:** a fast-delivery store has little shelf space, so pick products that sell fast, steadily and are rarely returned.
**Backtest:** pick with data before 3 Oct, judge on the festive weeks. My composite score (19.1% of units) lost to "top by past units" (23.9%).
**Say:** "I tested an elaborate score against a simple rule, the simple rule won, so I shipped the simple one."

## Stage 6: pricing.py
**Elasticity:** % change in units for a 1% price change. Estimated with a fixed-effects regression: SKU effects remove "popular product", week effects remove "festive week".
**Problem:** wholesale customers get lower unit prices on bigger orders, which inflates the estimate. Using only weeks with one list price gives -1.7; other definitions give -1.5 to -2.0.
**Break-even rule:** with profit = units x (price - cost), a discount raises profit only if |elasticity| > 1 / margin (3.33 at 30%).
**Q:** Is -1.7 causal? No, it is an association; proving causation needs a price experiment.

## Stage 7: analyst.py and brief.py
**Principle:** the AI never does maths. It writes SQL; SQLite runs it. Guardrails: one SELECT only, forbidden keywords blocked, database opened read-only, 200-row cap, SQL shown to the user.
**Brief:** Python computes facts; the LLM only phrases them; `check_numbers` rejects any number not in the facts.
**Test:** 15 questions with hand-written golden SQL (`python -m src.analyst`). Run it once you have a key and quote your real score.

## Stage 8: app.py
Six tabs, pure Python (Streamlit). Know that each tab reads from the SQLite database or the CSVs in `outputs/`.

## Ten questions to rehearse
1. Why this dataset, and what is wrong with it? 2. What did you remove and why? 3. How did you test the forecast?
4. Where did the model fail? 5. Why SQLite? 6. What is elasticity and how reliable is yours? 7. When does a discount raise profit?
8. How do you stop the AI from giving wrong numbers? 9. What changes with real marketplace data? 10. What did AI help with, and what did you verify yourself?
