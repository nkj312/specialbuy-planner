"""Local Streamlit prototype for synthetic special-buy demand and ordering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from specialbuy_model import forecast_demand, optimize_orders


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Special Buy Planner", page_icon="📦", layout="wide")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(DATA_DIR / "synthetic_products.csv")
    history = pd.read_csv(DATA_DIR / "synthetic_history.csv")
    return products, history


def run_pipeline(
    products: pd.DataFrame,
    history: pd.DataFrame,
    horizon: int,
    launch_month: int,
    budget: float | None,
    stores: int,
) -> dict:
    forecast, metrics = forecast_demand(history, products, horizon, launch_month)
    solution = optimize_orders(products, forecast, budget=budget, stores=stores)
    return {"forecast": forecast, "metrics": metrics, "solution": solution}


if "products" not in st.session_state:
    st.session_state.products, st.session_state.history = load_inputs()

if "model_run" not in st.session_state:
    try:
        st.session_state.model_run = run_pipeline(
            st.session_state.products,
            st.session_state.history,
            horizon=8,
            launch_month=11,
            budget=None,
            stores=1,
        )
        st.session_state.run_error = None
    except (ValueError, RuntimeError) as error:
        st.session_state.model_run = None
        st.session_state.run_error = str(error)

st.title("Special Buy Planner")
st.caption(
    "Synthetic demand forecasting and multiperiod order optimization · "
    "Academic prototype based on Tschan et al. (2025)"
)
st.info(
    "All products and sales in this prototype are synthetic. It is an educational "
    "demonstration, not ALDI SÜD software, data, or a reproduction of its production model."
)

results_tab, inputs_tab, model_tab = st.tabs(
    ["Decision dashboard", "Inputs & synthetic data", "Model & testing"]
)

with inputs_tab:
    st.subheader("Decision inputs")
    st.write(
        "Edit the synthetic product economics, choose a selling horizon, and optionally "
        "apply a procurement budget. The demand model is evaluated on later synthetic "
        "product launches before being refit on all generated history."
    )
    with st.form("model_inputs"):
        edited_products = st.data_editor(
            st.session_state.products,
            hide_index=True,
            width="stretch",
            disabled=["product_id", "product_name", "category", "markdown_schedule"],
            column_config={
                "selling_price": st.column_config.NumberColumn(
                    "Selling price (€)", min_value=0.01, step=1.0, format="%.2f"
                ),
                "unit_cost": st.column_config.NumberColumn(
                    "Purchase cost (€)", min_value=0.0, step=1.0, format="%.2f"
                ),
                "pack_size": st.column_config.NumberColumn(
                    "Units per carton", min_value=1, step=1
                ),
                "max_order_qty": st.column_config.NumberColumn(
                    "Maximum order (units)", min_value=1, step=10
                ),
            },
        )
        control_columns = st.columns(4)
        horizon = control_columns[0].slider("Selling horizon (weeks)", 2, 8, 8)
        launch_month = control_columns[1].selectbox(
            "Launch month",
            options=range(1, 13),
            index=10,
            format_func=lambda month: pd.Timestamp(2026, month, 1).strftime("%B"),
        )
        stores = control_columns[2].number_input(
            "Stores represented", min_value=1, max_value=500, value=1, step=1
        )
        use_budget = control_columns[3].checkbox("Apply procurement budget")
        budget = st.number_input(
            "Procurement budget (€)",
            min_value=0.0,
            value=20_000.0,
            step=1_000.0,
            disabled=not use_budget,
        )
        submitted = st.form_submit_button(
            "Run forecast + optimize", type="primary", width="stretch"
        )

    if submitted:
        try:
            st.session_state.model_run = run_pipeline(
                edited_products,
                st.session_state.history,
                int(horizon),
                int(launch_month),
                float(budget) if use_budget else None,
                int(stores),
            )
            st.session_state.products = edited_products
            st.session_state.run_error = None
        except (ValueError, RuntimeError) as error:
            st.session_state.model_run = None
            st.session_state.run_error = str(error)

    if st.session_state.run_error:
        st.error(st.session_state.run_error)

    st.subheader("Synthetic historical sales used for training")
    history = st.session_state.history
    st.caption(
        f"{len(history):,} synthetic weekly observations across "
        f"{history['launch_id'].nunique()} historical launches and "
        f"{history['product_id'].nunique()} product types."
    )
    st.dataframe(history.head(20), hide_index=True, width="stretch")
    st.download_button(
        "Download synthetic training data",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_special_buy_history.csv",
        mime="text/csv",
    )

with results_tab:
    result = st.session_state.model_run
    if result is None:
        st.warning("Update the inputs and run the model to display results.")
    else:
        solution = result["solution"]
        metrics = result["metrics"]
        totals = solution["totals"]
        st.subheader("Recommended order plan")
        kpis = st.columns(4)
        kpis[0].metric("Products", len(solution["product_results"]))
        kpis[1].metric("Recommended order", f"{totals['order_qty']:,.0f} units")
        kpis[2].metric("Expected profit", f"€{totals['profit']:,.2f}")
        kpis[3].metric("Forecast holdout MAE", f"{metrics['MAE']:.1f} units")
        st.caption(
            f"Forecast test: {metrics['holdout_launches']} later synthetic launches · "
            f"RMSE {metrics['RMSE']:.1f} · MAPE {metrics['MAPE_pct']:.1f}% · "
            f"MILP solver: {solution['status']}"
        )

        product_results = solution["product_results"]
        weekly_results = solution["weekly_results"]
        chart_columns = st.columns(2)
        with chart_columns[0]:
            order_chart = product_results.melt(
                id_vars=["product_name"],
                value_vars=["forecast_demand", "order_qty"],
                var_name="Measure",
                value_name="Units",
            )
            fig = px.bar(
                order_chart,
                x="product_name",
                y="Units",
                color="Measure",
                barmode="group",
                title="Forecast demand vs recommended order",
                labels={"product_name": "Product"},
            )
            fig.update_layout(xaxis_title="", legend_title="")
            st.plotly_chart(fig, width="stretch")
        with chart_columns[1]:
            weekly = weekly_results.groupby("week", as_index=False)[
                ["forecast_demand", "expected_sales"]
            ].sum()
            fig = px.line(
                weekly,
                x="week",
                y=["forecast_demand", "expected_sales"],
                markers=True,
                title="Expected demand and sales across the life cycle",
                labels={"week": "Selling week", "value": "Units", "variable": ""},
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Product recommendations")
        st.dataframe(
            product_results[
                [
                    "product_name",
                    "forecast_demand",
                    "order_qty",
                    "cartons",
                    "expected_sales",
                    "unmet_demand",
                    "leftover_qty",
                    "expected_profit",
                    "service_level_pct",
                ]
            ].style.format(
                {
                    "forecast_demand": "{:,.1f}",
                    "order_qty": "{:,.0f}",
                    "expected_sales": "{:,.1f}",
                    "unmet_demand": "{:,.1f}",
                    "leftover_qty": "{:,.1f}",
                    "expected_profit": "€{:,.2f}",
                    "service_level_pct": "{:.1f}%",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download order recommendations",
            data=product_results.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_order_recommendations.csv",
            mime="text/csv",
        )
        st.subheader("Weekly inventory projection")
        st.dataframe(
            weekly_results.style.format(
                {
                    "forecast_demand": "{:,.1f}",
                    "opening_inventory": "{:,.1f}",
                    "expected_sales": "{:,.1f}",
                    "closing_inventory": "{:,.1f}",
                    "markdown_pct": "{:.0%}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

with model_tab:
    st.subheader("What the prototype does")
    st.markdown(
        """
1. **Predict:** a random-forest regressor learns weekly unit demand from synthetic
   historical launches, product category, selling price, markdown, and seasonality.
2. **Prescribe:** a mixed-integer linear program chooses carton-sized starting orders.
3. **Explain:** the dashboard reports expected sales, stock, leftovers, service, and
   profit under the selected inputs.

The workflow follows the course's predictive-to-prescriptive analytics and LP/MIP
modeling concepts. The paper's Appendix A provides the order-quantity and stock-cost
structure. This prototype adds carton decisions and allows unmet demand so that service
trade-offs can be demonstrated; those are educational extensions, not claims about the
paper's production implementation.
"""
    )
    st.latex(
        r"\max \sum_{i,t} p_i(1-m_{it})x_{it}"
        r"-\sum_i c_i Q_i"
        r"-\sum_{i,t}(h_i+\gamma_{it}g_i)I_{it}"
        r"-\sum_i(d_iw_i+S f_i y_i)"
    )
    st.latex(
        r"Q_i=P_i n_i,\quad I_{i,1}=Q_i,\quad"
        r"I_{i,t+1}=I_{it}-x_{it},\quad"
        r"0\le x_{it}\le\min(\hat d_{it},I_{it}),\quad"
        r"w_i=I_{i,T}-x_{i,T}"
    )
    st.markdown(
        """
**Decision variables:** carton count `nᵢ`, product activation `yᵢ`, weekly sales `xᵢₜ`,
opening inventory `Iᵢₜ`, and end-of-life leftover `wᵢ`.

**Inputs:** forecast demand `d̂ᵢₜ`, price `pᵢ`, markdown `mᵢₜ`, purchase cost `cᵢ`,
carton size `Pᵢ`, holding cost `hᵢ`, markdown handling cost `gᵢ`, disposal cost `dᵢ`,
fixed store handling cost `fᵢ`, and represented stores `S`. An optional budget caps
procurement spend.

**Constraints:** carton-sized order, product minimum/maximum, inventory balance,
sales cannot exceed forecast demand or available stock, final leftovers are disposed,
and optional procurement budget.
"""
    )
    st.markdown(
        "Research reference: Tschan, A. et al. (2025), *Data-Driven Inventory Control "
        "and Integrated Employee Involvement for Special Buys at ALDI SÜD Germany*, "
        "[INFORMS Journal on Applied Analytics](https://doi.org/10.1287/inte.2024.0182)."
    )
    st.caption(
        "AI use note: AI assistance was used to implement and document this prototype. "
        "The team should review, understand, and independently explain the assumptions "
        "and results before submission."
    )
