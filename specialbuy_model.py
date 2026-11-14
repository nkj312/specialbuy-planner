"""Synthetic demand forecasting and a compact multiperiod order-quantity MILP."""

from __future__ import annotations

from math import ceil, floor, isfinite, pi, sin, cos
from typing import Any

import numpy as np
import pandas as pd
import pulp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_FEATURES = ["product_id", "category"]
NUMERIC_FEATURES = [
    "selling_price",
    "unit_cost",
    "week_of_life",
    "markdown_pct",
    "month_sin",
    "month_cos",
]


class ModelInputError(ValueError):
    """Raised when supplied product or demand data cannot be modeled."""


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[
        ["product_id", "category", "selling_price", "unit_cost", "week", "markdown_pct", "month"]
    ].copy()
    features["product_id"] = features["product_id"].astype(str)
    features["category"] = features["category"].astype(str)
    features["week_of_life"] = pd.to_numeric(features.pop("week"), errors="coerce")
    month = pd.to_numeric(features.pop("month"), errors="coerce")
    features["month_sin"] = np.sin(2 * pi * month / 12)
    features["month_cos"] = np.cos(2 * pi * month / 12)
    return features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def _new_regressor() -> Pipeline:
    encoder = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", encoder),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=2,
                    max_features=0.9,
                    random_state=41,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _markdown_rates(schedule: Any, horizon: int) -> list[float]:
    try:
        rates = [
            float(part.strip().rstrip("%"))
            / (100 if "%" in part else 1)
            for part in str(schedule).split(",")
            if part.strip()
        ]
    except (TypeError, ValueError) as error:
        raise ModelInputError(f"Invalid markdown schedule: {schedule!r}.") from error
    if not rates or any(not isfinite(rate) or rate < 0 or rate >= 1 for rate in rates):
        raise ModelInputError("Markdown rates must be between 0% and 100%.")
    return (rates + [rates[-1]] * horizon)[:horizon]


def forecast_demand(
    history: pd.DataFrame,
    products: pd.DataFrame,
    horizon: int = 8,
    launch_month: int = 11,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate on the latest launches, refit, then forecast a new product cycle."""
    history_required = {
        "launch_id",
        "product_id",
        "category",
        "selling_price",
        "unit_cost",
        "week",
        "markdown_pct",
        "month",
        "weekly_sales",
    }
    product_required = {
        "product_id",
        "product_name",
        "category",
        "selling_price",
        "unit_cost",
        "markdown_schedule",
    }
    missing_history = history_required - set(history.columns)
    missing_products = product_required - set(products.columns)
    if missing_history:
        raise ModelInputError(
            f"Synthetic history is missing columns: {', '.join(sorted(missing_history))}."
        )
    if missing_products:
        raise ModelInputError(
            f"Product data is missing columns: {', '.join(sorted(missing_products))}."
        )
    if not 1 <= horizon <= 8:
        raise ModelInputError("The selling horizon must be from 1 to 8 weeks.")
    if not 1 <= launch_month <= 12:
        raise ModelInputError("Launch month must be from 1 to 12.")
    if products.empty or history.empty:
        raise ModelInputError("Product and synthetic history tables cannot be empty.")

    frame = history.copy().reset_index(drop=True)
    numeric = ["launch_id", "selling_price", "unit_cost", "week", "markdown_pct", "month", "weekly_sales"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any():
        raise ModelInputError("Synthetic history contains missing or non-numeric values.")
    if (frame[["selling_price", "unit_cost", "weekly_sales"]] < 0).any().any():
        raise ModelInputError("Prices, costs, and historical sales cannot be negative.")
    if frame["week"].lt(1).any() or frame["month"].lt(1).any() or frame["month"].gt(12).any():
        raise ModelInputError("Historical weeks must be positive and months must be 1 through 12.")
    if frame["launch_id"].nunique() < 4:
        raise ModelInputError("At least four historical launch cohorts are required.")
    if products["product_id"].astype(str).duplicated().any():
        raise ModelInputError("Product IDs must be unique.")

    launches = np.sort(frame["launch_id"].unique())
    split = max(1, int(len(launches) * 0.8))
    split = min(split, len(launches) - 1)
    training_ids = set(launches[:split])
    training_mask = frame["launch_id"].isin(training_ids)
    train = frame.loc[training_mask]
    test = frame.loc[~training_mask]
    if train.empty or test.empty:
        raise ModelInputError("A chronological launch holdout could not be created.")

    validation_model = _new_regressor()
    validation_model.fit(_features(train), train["weekly_sales"])
    test_prediction = np.maximum(0, validation_model.predict(_features(test)))
    actual = test["weekly_sales"].to_numpy(dtype=float)
    nonzero = actual > 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - test_prediction[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "MAE": float(mean_absolute_error(actual, test_prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, test_prediction))),
        "MAPE_pct": mape,
        "holdout_launches": int(frame.loc[~training_mask, "launch_id"].nunique()),
    }

    final_model = _new_regressor()
    final_model.fit(_features(frame), frame["weekly_sales"])
    product_rows = {str(row.product_id): row for row in products.itertuples(index=False)}
    future_rows: list[dict[str, Any]] = []
    for product_id, row in product_rows.items():
        rates = _markdown_rates(row.markdown_schedule, horizon)
        for week in range(1, horizon + 1):
            month = ((launch_month - 1 + (week - 1) // 4) % 12) + 1
            future_rows.append(
                {
                    "product_id": product_id,
                    "product_name": str(row.product_name),
                    "category": str(row.category),
                    "week": week,
                    "selling_price": float(row.selling_price),
                    "unit_cost": float(row.unit_cost),
                    "markdown_pct": rates[week - 1],
                    "month": month,
                }
            )
    forecast = pd.DataFrame(future_rows)
    forecast["forecast_demand"] = np.maximum(
        0, final_model.predict(_features(forecast))
    )
    return forecast, metrics


def optimize_orders(
    products: pd.DataFrame,
    forecast: pd.DataFrame,
    budget: float | None = None,
    stores: int = 1,
) -> dict[str, Any]:
    """Choose carton multiples to maximize predicted life-cycle profit."""
    required_products = {
        "product_id",
        "product_name",
        "selling_price",
        "unit_cost",
        "pack_size",
        "min_order_qty",
        "max_order_qty",
        "holding_cost",
        "markdown_cost",
        "disposal_cost",
        "fixed_handling_cost",
    }
    required_forecast = {"product_id", "week", "forecast_demand", "markdown_pct"}
    missing_products = required_products - set(products.columns)
    missing_forecast = required_forecast - set(forecast.columns)
    if missing_products:
        raise ModelInputError(
            f"Product data is missing columns: {', '.join(sorted(missing_products))}."
        )
    if missing_forecast:
        raise ModelInputError(
            f"Forecast is missing columns: {', '.join(sorted(missing_forecast))}."
        )
    if products.empty or forecast.empty:
        raise ModelInputError("Product and forecast tables cannot be empty.")
    if not isinstance(stores, int) or stores < 1:
        raise ModelInputError("The number of stores must be a positive whole number.")
    if budget is not None and (not isfinite(budget) or budget < 0):
        raise ModelInputError("Procurement budget must be finite and non-negative.")
    if forecast.duplicated(["product_id", "week"]).any():
        raise ModelInputError("Forecast rows must be unique by product and week.")

    product_rows = {str(row.product_id): row for row in products.itertuples(index=False)}
    forecast_frame = forecast.copy()
    forecast_frame["product_id"] = forecast_frame["product_id"].astype(str)
    if set(forecast_frame["product_id"]) != set(product_rows):
        raise ModelInputError("Every product must have forecast rows, and no unknown IDs are allowed.")
    for column in ["week", "forecast_demand", "markdown_pct"]:
        forecast_frame[column] = pd.to_numeric(forecast_frame[column], errors="coerce")
    if forecast_frame[["week", "forecast_demand", "markdown_pct"]].isna().any().any():
        raise ModelInputError("Forecast weeks, demand, and markdown rates must be numeric.")
    if (forecast_frame[["forecast_demand", "markdown_pct"]] < 0).any().any():
        raise ModelInputError("Forecast demand and markdown rates cannot be negative.")
    if (forecast_frame["markdown_pct"] >= 1).any():
        raise ModelInputError("Markdown rates must be below 100%.")
    weeks = sorted(forecast_frame["week"].unique())
    if weeks != list(range(1, len(weeks) + 1)):
        raise ModelInputError("Forecast weeks must start at 1 and be consecutive.")

    for product_id, row in product_rows.items():
        values = {
            "selling_price": row.selling_price,
            "unit_cost": row.unit_cost,
            "pack_size": row.pack_size,
            "min_order_qty": row.min_order_qty,
            "max_order_qty": row.max_order_qty,
            "holding_cost": row.holding_cost,
            "markdown_cost": row.markdown_cost,
            "disposal_cost": row.disposal_cost,
            "fixed_handling_cost": row.fixed_handling_cost,
        }
        try:
            numeric_values = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError) as error:
            raise ModelInputError(f"{product_id}: costs and order limits must be numeric.") from error
        if any(not isfinite(value) for value in numeric_values.values()):
            raise ModelInputError(f"{product_id}: costs and order limits must be finite.")
        if numeric_values["selling_price"] <= 0 or numeric_values["unit_cost"] < 0:
            raise ModelInputError(f"{product_id}: selling price must be positive and unit cost non-negative.")
        if numeric_values["pack_size"] < 1 or not numeric_values["pack_size"].is_integer():
            raise ModelInputError(f"{product_id}: carton size must be a positive whole number.")
        if not numeric_values["min_order_qty"].is_integer() or not numeric_values["max_order_qty"].is_integer():
            raise ModelInputError(f"{product_id}: minimum and maximum order quantities must be whole units.")
        if numeric_values["max_order_qty"] < 0 or numeric_values["min_order_qty"] < 0:
            raise ModelInputError(f"{product_id}: minimum and maximum orders cannot be negative.")
        if any(numeric_values[name] < 0 for name in ("holding_cost", "markdown_cost", "disposal_cost", "fixed_handling_cost")):
            raise ModelInputError(f"{product_id}: operating costs cannot be negative.")

    model = pulp.LpProblem("Synthetic_Special_Buy_Order_Planning", pulp.LpMaximize)
    cartons: dict[str, pulp.LpVariable] = {}
    active: dict[str, pulp.LpVariable] = {}
    inventory: dict[tuple[str, int], pulp.LpVariable] = {}
    sales: dict[tuple[str, int], pulp.LpVariable] = {}
    disposal: dict[str, pulp.LpVariable] = {}
    order_units: dict[str, Any] = {}

    for product_id, row in product_rows.items():
        pack_size = int(row.pack_size)
        min_cartons = ceil(int(row.min_order_qty) / pack_size)
        max_cartons = floor(int(row.max_order_qty) / pack_size)
        if max_cartons < min_cartons:
            raise ModelInputError(
                f"{product_id}: maximum order cannot satisfy the minimum using whole cartons."
            )
        cartons[product_id] = pulp.LpVariable(
            f"cartons_{product_id}", lowBound=0, upBound=max_cartons, cat=pulp.LpInteger
        )
        active[product_id] = pulp.LpVariable(f"active_{product_id}", cat=pulp.LpBinary)
        model += cartons[product_id] >= max(1, min_cartons) * active[product_id]
        model += cartons[product_id] <= max_cartons * active[product_id]
        order_units[product_id] = pack_size * cartons[product_id]

        product_forecast = forecast_frame[
            forecast_frame["product_id"] == product_id
        ].sort_values("week")
        if product_forecast["week"].astype(int).tolist() != weeks:
            raise ModelInputError(f"{product_id}: forecast must contain every modeled week.")
        for position, row_forecast in enumerate(product_forecast.itertuples(index=False)):
            week = int(row_forecast.week)
            inventory[product_id, week] = pulp.LpVariable(
                f"inventory_{product_id}_{week}", lowBound=0
            )
            sales[product_id, week] = pulp.LpVariable(
                f"sales_{product_id}_{week}", lowBound=0
            )
            model += sales[product_id, week] <= float(row_forecast.forecast_demand)
            model += sales[product_id, week] <= inventory[product_id, week]
            if position == 0:
                model += inventory[product_id, week] == order_units[product_id]
            else:
                previous_week = weeks[position - 1]
                model += (
                    inventory[product_id, week]
                    == inventory[product_id, previous_week] - sales[product_id, previous_week]
                )
        final_week = weeks[-1]
        disposal[product_id] = pulp.LpVariable(f"leftover_{product_id}", lowBound=0)
        model += (
            disposal[product_id]
            == inventory[product_id, final_week] - sales[product_id, final_week]
        )

    if budget is not None:
        model += (
            pulp.lpSum(
                float(product_rows[product_id].unit_cost) * order_units[product_id]
                for product_id in product_rows
            )
            <= budget
        )

    revenue_terms = []
    operating_cost_terms = []
    for product_id, row in product_rows.items():
        product_forecast = forecast_frame[
            forecast_frame["product_id"] == product_id
        ].sort_values("week")
        revenue_terms.extend(
            float(row.selling_price)
            * (1 - float(item.markdown_pct))
            * sales[product_id, int(item.week)]
            for item in product_forecast.itertuples(index=False)
        )
        operating_cost_terms.extend(
            float(row.holding_cost) * inventory[product_id, int(item.week)]
            + (
                float(row.markdown_cost) * inventory[product_id, int(item.week)]
                if float(item.markdown_pct) > 0
                else 0
            )
            for item in product_forecast.itertuples(index=False)
        )
        operating_cost_terms.extend(
            [
                float(row.unit_cost) * order_units[product_id],
                float(row.disposal_cost) * disposal[product_id],
                float(row.fixed_handling_cost) * stores * active[product_id],
            ]
        )
    model += pulp.lpSum(revenue_terms) - pulp.lpSum(operating_cost_terms)
    try:
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
    except pulp.PulpSolverError as error:
        raise RuntimeError(f"CBC could not run the order optimization: {error}") from error
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"The order optimization did not find an optimal solution: {status}.")

    product_results: list[dict[str, Any]] = []
    weekly_results: list[dict[str, Any]] = []
    for product_id, row in product_rows.items():
        order_qty = float(pulp.value(order_units[product_id]) or 0)
        product_forecast = forecast_frame[
            forecast_frame["product_id"] == product_id
        ].sort_values("week")
        values = {
            "forecast_demand": 0.0,
            "expected_sales": 0.0,
            "revenue": 0.0,
            "holding_cost": 0.0,
            "markdown_cost": 0.0,
        }
        for item in product_forecast.itertuples(index=False):
            week = int(item.week)
            opening = float(pulp.value(inventory[product_id, week]) or 0)
            sold = float(pulp.value(sales[product_id, week]) or 0)
            closing = max(0.0, opening - sold)
            demand_value = float(item.forecast_demand)
            revenue = float(row.selling_price) * (1 - float(item.markdown_pct)) * sold
            hold_cost = float(row.holding_cost) * opening
            markdown_cost = (
                float(row.markdown_cost) * opening if float(item.markdown_pct) > 0 else 0.0
            )
            weekly_results.append(
                {
                    "product_id": product_id,
                    "product_name": str(row.product_name),
                    "week": week,
                    "forecast_demand": demand_value,
                    "opening_inventory": opening,
                    "expected_sales": sold,
                    "closing_inventory": closing,
                    "markdown_pct": float(item.markdown_pct),
                }
            )
            values["forecast_demand"] += demand_value
            values["expected_sales"] += sold
            values["revenue"] += revenue
            values["holding_cost"] += hold_cost
            values["markdown_cost"] += markdown_cost

        leftover = float(pulp.value(disposal[product_id]) or 0)
        procurement = float(row.unit_cost) * order_qty
        disposal_cost = float(row.disposal_cost) * leftover
        fixed_cost = (
            float(row.fixed_handling_cost) * stores
            if float(pulp.value(active[product_id]) or 0) > 0.5
            else 0.0
        )
        profit = (
            values["revenue"]
            - procurement
            - values["holding_cost"]
            - values["markdown_cost"]
            - disposal_cost
            - fixed_cost
        )
        service = (
            100 * values["expected_sales"] / values["forecast_demand"]
            if values["forecast_demand"] > 0
            else 100.0
        )
        product_results.append(
            {
                "product_id": product_id,
                "product_name": str(row.product_name),
                "forecast_demand": values["forecast_demand"],
                "order_qty": order_qty,
                "cartons": int(round(order_qty / int(row.pack_size))),
                "expected_sales": values["expected_sales"],
                "unmet_demand": max(0.0, values["forecast_demand"] - values["expected_sales"]),
                "leftover_qty": leftover,
                "revenue": values["revenue"],
                "procurement_cost": procurement,
                "holding_cost": values["holding_cost"],
                "markdown_cost": values["markdown_cost"],
                "disposal_cost": disposal_cost,
                "fixed_handling_cost": fixed_cost,
                "expected_profit": profit,
                "service_level_pct": service,
            }
        )

    product_frame = pd.DataFrame(product_results)
    totals = {
        "forecast_demand": float(product_frame["forecast_demand"].sum()),
        "order_qty": float(product_frame["order_qty"].sum()),
        "expected_sales": float(product_frame["expected_sales"].sum()),
        "profit": float(product_frame["expected_profit"].sum()),
        "procurement_cost": float(product_frame["procurement_cost"].sum()),
        "leftover_qty": float(product_frame["leftover_qty"].sum()),
    }
    totals["service_level_pct"] = (
        100 * totals["expected_sales"] / totals["forecast_demand"]
        if totals["forecast_demand"] > 0
        else 100.0
    )
    return {
        "status": status,
        "objective_value": float(pulp.value(model.objective)),
        "product_results": product_frame,
        "weekly_results": pd.DataFrame(weekly_results),
        "totals": totals,
    }
