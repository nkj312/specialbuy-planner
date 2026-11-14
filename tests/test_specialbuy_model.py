"""Tests for the focused synthetic forecast-to-order prototype."""

import numpy as np
import pandas as pd
import pytest

from data.generate_synthetic import create_synthetic_data
from specialbuy_model import ModelInputError, forecast_demand, optimize_orders


def test_synthetic_history_is_repeatable_and_has_multiple_launches():
    products_a, history_a = create_synthetic_data(launches=6, seed=7)
    products_b, history_b = create_synthetic_data(launches=6, seed=7)

    pd.testing.assert_frame_equal(products_a, products_b)
    pd.testing.assert_frame_equal(history_a, history_b)
    assert history_a["launch_id"].nunique() == 6
    assert history_a["weekly_sales"].ge(0).all()


def test_forecast_uses_later_launch_holdout_and_returns_selected_horizon():
    products, history = create_synthetic_data(launches=8, seed=7)

    forecast, metrics = forecast_demand(history, products, horizon=4, launch_month=11)

    assert len(forecast) == len(products) * 4
    assert forecast["forecast_demand"].ge(0).all()
    assert metrics["holdout_launches"] == 2
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])


def test_milp_orders_carton_multiples_and_conserves_inventory():
    products, history = create_synthetic_data(launches=8, seed=7)
    products = products.iloc[:3].reset_index(drop=True)
    forecast, _ = forecast_demand(history, products, horizon=4, launch_month=3)

    result = optimize_orders(products, forecast)
    product_rows = {row.product_id: row for row in products.itertuples(index=False)}

    assert result["status"] == "Optimal"
    for row in result["product_results"].itertuples(index=False):
        assert row.order_qty % int(product_rows[row.product_id].pack_size) == pytest.approx(0)
        assert row.expected_sales <= row.forecast_demand + 1e-6
        assert row.leftover_qty >= -1e-6
    for product_id, rows in result["weekly_results"].groupby("product_id"):
        ordered = rows.sort_values("week")
        assert np.allclose(
            ordered["closing_inventory"].iloc[:-1],
            ordered["opening_inventory"].iloc[1:],
        )
        assert ordered["expected_sales"].le(ordered["forecast_demand"] + 1e-6).all()


def test_zero_budget_produces_zero_orders():
    products, history = create_synthetic_data(launches=8, seed=7)
    products = products.iloc[:2].reset_index(drop=True)
    forecast, _ = forecast_demand(history, products, horizon=3, launch_month=1)

    result = optimize_orders(products, forecast, budget=0)

    assert result["totals"]["order_qty"] == pytest.approx(0)
    assert result["totals"]["expected_sales"] == pytest.approx(0)


def test_invalid_horizon_is_reported():
    products, history = create_synthetic_data(launches=8, seed=7)

    with pytest.raises(ModelInputError, match="selling horizon"):
        forecast_demand(history, products, horizon=9)
