import pandas as pd

from src.analytics.inventory import (
    calculate_inventory_metrics,
    add_stock_status,
    inventory_summary,
)


def test_inventory_pipeline():

    sales_df = pd.read_csv(
        "data/sample/retail_sales.csv"
    )

    inventory_df = pd.read_csv(
        "data/sample/retail_inventory.csv"
    )

    sales_df["date"] = pd.to_datetime(
        sales_df["date"]
    )

    inventory = calculate_inventory_metrics(
        sales_df=sales_df,
        inventory_df=inventory_df,
    )

    inventory = add_stock_status(inventory)

    summary = inventory_summary(inventory)

    assert len(inventory) > 0

    assert summary["products"] > 0

    assert "stock_status" in inventory.columns

    assert "days_of_stock" in inventory.columns

    assert "closing_stock" in inventory.columns

    assert "units_sold" in inventory.columns