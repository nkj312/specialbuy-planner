# Special Buy Planner

**Group project prototype and report**  
**Authors:** Add group member names  
**Research paper:** Tschan, A. et al. (2025), [“Data-Driven Inventory Control and Integrated Employee Involvement for Special Buys at ALDI SÜD Germany”](https://doi.org/10.1287/inte.2024.0182), *INFORMS Journal on Applied Analytics*, 55(1), 22–35.

> This academic prototype uses generated data only. It is not ALDI SÜD software, data, or a reproduction of the company’s production model.

## 1. Business Requirements Document (BRD)

### Business problem

Special-buy products are ordered ahead of a short selling life cycle, with limited or no replenishment. Ordering too little can leave demand unmet; ordering too much ties up cash and creates stock-keeping, markdown-handling, and end-of-life disposal costs. The selected paper addresses this decision with demand prediction and a multiperiod mixed-integer order-quantity model.

### Goal and users

The prototype demonstrates how an operations planner can turn a synthetic sales history and editable product assumptions into weekly demand forecasts and carton-sized starting order recommendations. It displays the expected profit, sales, service, inventory movement, and leftover stock for business review.

### Scope

**Included:** synthetic historical data; a trained and holdout-evaluated demand regressor; an eight-week-or-shorter selling horizon; carton multiples; optional procurement budget; inventory flow; markdown, holding, procurement, handling, and disposal economics; result tables and charts.

**Not included:** live data or integrations; store allocation; replenishment; inter-store reallocation; markdown-price optimization; uncertain-demand optimization; the paper’s full proprietary operating environment.

### Success criteria

1. A user can inspect/edit key product assumptions and select a launch month, horizon, store count, and optional budget.
2. The app trains/evaluates a synthetic-data forecast and runs the order model from those forecasts.
3. Every recommendation is a valid carton multiple and inventory balances across weeks.
4. Results include forecast accuracy, demand service, recommended quantities, inventory projections, and expected profit.

## 2. Software Requirements Specification (SRS)

| ID | Requirement |
|---|---|
| FR-1 | Load reproducible synthetic product and weekly sales data without requiring an external service. |
| FR-2 | Train a demand model and report MAE, RMSE, and MAPE on later launch cohorts held out chronologically. |
| FR-3 | Forecast demand by product and selling week for a selected launch month and horizon of 1–8 weeks. |
| FR-4 | Recommend carton-sized orders within each product’s minimum and maximum. |
| FR-5 | Account for demand served, markdown-adjusted revenue, procurement, holding, markdown handling, fixed handling, and final disposal. |
| FR-6 | Apply an optional procurement budget and report product/week outcomes in tables and charts. |
| NFR-1 | Use synthetic inputs, deterministic data generation, and a local Streamlit interface. |
| NFR-2 | Show validation or solver failures explicitly; do not present a failed run as a successful recommendation. |

### Inputs and assumptions

| Input | Meaning |
|---|---|
| Product ID/category | Synthetic assortment identifier and demand-model features |
| Selling price / unit cost | Expected unit revenue before markdown and unit procurement cost |
| Carton size and order bounds | Feasible initial order multiples and limits |
| Markdown schedule | Assumed markdown rate by selling week; used in demand prediction and selling price |
| Holding / markdown / disposal / fixed handling costs | Per-unit-week stock cost, markdown-period stock cost, final leftover cost, and per-order handling cost |
| Launch month / horizon / stores | Forecast seasonality, modeled weeks, and scale for fixed store handling |
| Optional budget | Maximum initial procurement expenditure |

The synthetic generator creates 24 launches for eight product types (1,536 weekly observations). Demand is a seeded, noisy function of product identity, category, price, markdown, lifecycle week, and calendar season. It is designed to demonstrate the workflow, not to represent measured retail behavior. Forecast evaluation withholds the latest 20% of launch cohorts; the final forecasting model is then fitted on all synthetic history.

## 3. Technical Design Document (TDD)

### Predictive model

A scikit-learn random-forest regressor predicts weekly unit demand from product ID, category, price, unit cost, week of life, markdown, and cyclical month features. The chronological launch holdout limits leakage from future synthetic launches. Reported accuracy describes this generated dataset only and must not be interpreted as real-world accuracy.

### Order-quantity MILP

Let `i` be products and `t` weeks. Forecast demand is `d̂ᵢₜ`; price is `pᵢ`; markdown is `mᵢₜ`; purchase cost is `cᵢ`; carton size is `Pᵢ`; holding, markdown, disposal, and fixed handling costs are `hᵢ`, `gᵢ`, `dᵢ`, and `fᵢ`.

**Decision variables:** `nᵢ` integer cartons; `yᵢ` binary order activation; `Qᵢ` initial units; `Iᵢₜ` opening inventory; `xᵢₜ` expected sales; and `wᵢ` final leftover units.

**Objective — maximize expected profit:**

```text
ΣᵢΣₜ [pᵢ(1 − mᵢₜ)xᵢₜ − hᵢIᵢₜ − 1(mᵢₜ > 0)gᵢIᵢₜ]
− Σᵢ [cᵢQᵢ + dᵢwᵢ + S fᵢ yᵢ]
```

**Constraints:**

```text
Qᵢ = Pᵢ nᵢ                                  carton-sized orders
minCartonsᵢ yᵢ ≤ nᵢ ≤ maxCartonsᵢ yᵢ        order bounds and activation
Iᵢ1 = Qᵢ                                    initial stock
Iᵢ,t+1 = Iᵢₜ − xᵢₜ                         inventory balance
0 ≤ xᵢₜ ≤ d̂ᵢₜ;  xᵢₜ ≤ Iᵢₜ                  demand and stock limits
wᵢ = IᵢT − xᵢT                             end-of-life leftover
Σᵢ cᵢQᵢ ≤ budget                            optional budget
nᵢ, Qᵢ ≥ 0; yᵢ ∈ {0,1}; Iᵢₜ, xᵢₜ, wᵢ ≥ 0
```

`S` is the user-selected number of stores. The MILP extends the paper’s order-quantity and stock-cost structure with explicit carton/activation variables, inventory balance and sales variables, optional budget, and unmet demand. It is intentionally small enough to explain and run in a classroom prototype. PuLP builds the linear model and CBC solves it.

### Process logic

```text
Load synthetic product table and launch history
Split history by launch ID; train and score a chronological holdout
Refit the demand model on all history
Predict demand for each selected product/week
Build and solve the carton-constrained profit-maximizing MILP
Validate solver status; calculate and display product/week outcomes
```

## 4. Testing plan

Tests run with `pytest`. Automated checks verify repeatable data generation, later-launch holdout, forecast shape and nonnegative predictions, carton-sized orders, weekly inventory conservation, demand limits, zero-budget behavior, and invalid horizon handling.

| Case | Sample input | Expected result |
|---|---|---|
| TC-01: default run | 8 products, 24 generated launches, 8 weeks | Optimal solver result, forecast metrics, recommendations, and charts |
| TC-02: short horizon | 4-week selection | Forecast and inventory output contain only weeks 1–4 |
| TC-03: carton feasibility | Any valid product assortment | Every recommended order is divisible by its carton size |
| TC-04: stock balance | Any optimal run | Closing stock in week `t` equals opening stock in week `t+1`; sales do not exceed stock or forecast demand |
| TC-05: budget | Zero procurement budget | Zero order and zero expected sales |
| TC-06: invalid horizon | Horizon greater than 8 | Input validation error; no result is reported |
| TC-07: changed economics | Edit price, cost, or carton size | Forecast/order run reflects the submitted input or reports invalid input |

## 5. Deploy on Streamlit Community Cloud

The app's entry point is `app.py`. To publish it:

1. Create a GitHub repository named `specialbuy-planner` and push the app files from this project directory.
2. Keep the repository public (the app itself is publicly accessible).
3. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
4. Select the `specialbuy-planner` repository, branch `main`, and main file path `app.py`.
5. Deploy. Streamlit installs the runtime packages from `requirements.txt`; `runtime.txt` selects Python 3.12.

Only the app, model, synthetic data, tests, and supporting configuration need to be published. Course/research PDFs and unrelated workspace files are excluded from Git.

## 6. Run locally

From the project directory in PowerShell:

```powershell
python -m pip install -r requirements.txt
python data\generate_synthetic.py
pytest -q
streamlit run app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`. The web prototype requires no credentials or external data.

## 7. AI usage note

AI assistance was used to implement and document this prototype. The project team is responsible for reviewing the paper, checking the synthetic-data assumptions, validating the formulation and test results, and being able to explain every design choice in its own words.
