import pandas as pd
import pytest

from src.forecasting.product_service import forecast_products


def create_sales_data():
    dates = pd.date_range(
        start="2024-01-01",
        periods=90,
        freq="D",
    )

    rows = []

    for date in dates:
        rows.append(
            {
                "date": date,
                "product_name": "Product A",
                "quantity": 10,
            }
        )

        rows.append(
            {
                "date": date,
                "product_name": "Product B",
                "quantity": 5,
            }
        )

    return pd.DataFrame(rows)


def create_inventory_data():
    return pd.DataFrame(
        [
            {
                "product_name": "Product A",
                "closing_stock": 20,
            },
            {
                "product_name": "Product B",
                "closing_stock": 100,
            },
        ]
    )


def test_forecast_products_returns_all_products():
    sales_df = create_sales_data()
    inventory_df = create_inventory_data()

    result, skipped = forecast_products(
        sales_df,
        inventory_df,
    )

    assert len(result) == 2
    assert skipped.empty

    assert set(result["product_name"]) == {
        "Product A",
        "Product B",
    }


def test_forecast_products_requires_sales_columns():
    sales_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2024-01-01",
                periods=90,
                freq="D",
            ),
            "product_name": ["Product A"] * 90,
        }
    )

    inventory_df = create_inventory_data()

    with pytest.raises(ValueError, match="Missing required sales columns"):
        forecast_products(
            sales_df,
            inventory_df,
        )


def test_low_stock_product_is_flagged_for_reorder():
    sales_df = create_sales_data()
    inventory_df = create_inventory_data()

    result, skipped = forecast_products(
        sales_df,
        inventory_df,
    )

    product_a = result[
        result["product_name"] == "Product A"
    ].iloc[0]

    assert product_a["status"] == "Reorder"
    assert product_a["recommended_order_quantity"] > 0


def test_missing_inventory_product_is_skipped():
    sales_df = create_sales_data()

    inventory_df = pd.DataFrame(
        [
            {
                "product_name": "Product A",
                "closing_stock": 20,
            }
        ]
    )

    result, skipped = forecast_products(
        sales_df,
        inventory_df,
    )

    assert len(result) == 1
    assert len(skipped) == 1

    assert result.iloc[0]["product_name"] == "Product A"

    assert skipped.iloc[0]["product_name"] == "Product B"
    assert (
        skipped.iloc[0]["reason"]
        == "No inventory record found."
    )