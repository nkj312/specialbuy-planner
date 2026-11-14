"""Create reproducible synthetic product and weekly-sales data for the demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

PRODUCTS = [
    ("SB101", "Portable blender", "Household", 34.99, 17.50, 4, 420, 42),
    ("SB102", "Solar garden lantern", "Garden", 24.99, 10.50, 6, 720, 55),
    ("SB103", "Bluetooth speaker", "Electronics", 39.99, 21.00, 4, 480, 48),
    ("SB104", "Seasonal baking set", "Seasonal", 18.99, 9.00, 8, 960, 60),
    ("SB105", "Fleece throw", "Household", 22.99, 10.00, 6, 720, 65),
    ("SB106", "Garden hose set", "Garden", 29.99, 13.00, 5, 600, 38),
    ("SB107", "Family board game", "Toys", 27.99, 12.50, 6, 720, 46),
    ("SB108", "LED work light", "Electronics", 19.99, 8.50, 10, 1000, 52),
]
WEEKLY_PROFILE = [1.65, 1.18, 0.92, 0.78, 0.70, 0.60, 0.50, 0.42]
MARKDOWN_SCHEDULE = "0,0,0,0.05,0.10,0.15,0.20,0.25"
CATEGORY_PEAK_MONTH = {
    "Household": 11,
    "Garden": 6,
    "Electronics": 11,
    "Seasonal": 12,
    "Toys": 12,
}


def create_synthetic_data(
    launches: int = 24, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a synthetic assortment and its repeated eight-week sales history."""
    if launches < 4:
        raise ValueError("At least four historical launches are needed.")
    rng = np.random.default_rng(seed)
    product_rows = []
    history_rows = []
    for (
        product_id,
        product_name,
        category,
        price,
        unit_cost,
        pack_size,
        max_order_qty,
        base_demand,
    ) in PRODUCTS:
        product_rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "selling_price": price,
                "unit_cost": unit_cost,
                "pack_size": pack_size,
                "min_order_qty": 0,
                "max_order_qty": max_order_qty,
                "holding_cost": round(price * 0.0018, 3),
                "markdown_cost": round(price * 0.010, 3),
                "disposal_cost": round(price * 0.035, 3),
                "fixed_handling_cost": 8.0,
                "markdown_schedule": MARKDOWN_SCHEDULE,
            }
        )
        for launch_id in range(1, launches + 1):
            launch_month = ((launch_id - 1) % 12) + 1
            launch_price = price * rng.normal(1.0, 0.025)
            launch_unit_cost = unit_cost * launch_price / price
            trend = 1.0 + 0.003 * (launch_id - 1)
            rates = [float(value) for value in MARKDOWN_SCHEDULE.split(",")]
            for week, lifecycle_factor in enumerate(WEEKLY_PROFILE, start=1):
                month = ((launch_month - 1 + (week - 1) // 4) % 12) + 1
                peak_month = CATEGORY_PEAK_MONTH[category]
                seasonal = 1 + 0.20 * np.cos(2 * np.pi * (month - peak_month) / 12)
                markdown_rate = rates[week - 1]
                price_effect = (price / launch_price) ** 0.45
                expected = (
                    base_demand
                    * lifecycle_factor
                    * seasonal
                    * (1 + 1.15 * markdown_rate)
                    * price_effect
                    * trend
                )
                history_rows.append(
                    {
                        "launch_id": launch_id,
                        "product_id": product_id,
                        "category": category,
                        "selling_price": round(launch_price, 2),
                        "unit_cost": round(launch_unit_cost, 2),
                        "week": week,
                        "markdown_pct": markdown_rate,
                        "month": month,
                        "weekly_sales": int(rng.poisson(max(0.1, expected))),
                    }
                )
    return pd.DataFrame(product_rows), pd.DataFrame(history_rows)


def main() -> None:
    products, history = create_synthetic_data()
    products.to_csv(ROOT / "synthetic_products.csv", index=False)
    history.to_csv(ROOT / "synthetic_history.csv", index=False)
    print(
        f"Wrote {len(products)} synthetic products and "
        f"{len(history)} synthetic weekly sales rows to {ROOT}."
    )


if __name__ == "__main__":
    main()
