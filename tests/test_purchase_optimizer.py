import pandas as pd
import pytest

from src.optimization.purchase_optimizer import (
    optimize_purchase_plan,
)


def create_recommendations():
    return pd.DataFrame(
        [
            {
                "product_name": "Rice",
                "recommended_order_quantity": 20,
                "forecast_demand": 30,
                "current_stock": 5,
                "incoming_stock": 0,
                "status": "Reorder",
                "stock_risk": "High",
            },
            {
                "product_name": "Milk",
                "recommended_order_quantity": 20,
                "forecast_demand": 25,
                "current_stock": 2,
                "incoming_stock": 0,
                "status": "Reorder",
                "stock_risk": "Critical",
            },
            {
                "product_name": "Biscuits",
                "recommended_order_quantity": 20,
                "forecast_demand": 20,
                "current_stock": 10,
                "incoming_stock": 0,
                "status": "Reorder",
                "stock_risk": "Medium",
            },
        ]
    )


def create_supplier_prices():
    return pd.DataFrame(
        [
            {
                "product_name": "Rice",
                "supplier_name": "Supplier A",
                "unit_cost": 100,
                "minimum_order_quantity": 5,
                "pack_size": 1,
            },
            {
                "product_name": "Milk",
                "supplier_name": "Supplier B",
                "unit_cost": 50,
                "minimum_order_quantity": 4,
                "pack_size": 2,
            },
            {
                "product_name": "Biscuits",
                "supplier_name": "Supplier C",
                "unit_cost": 10,
                "minimum_order_quantity": 5,
                "pack_size": 5,
            },
        ]
    )


def test_plan_stays_within_budget():
    result = optimize_purchase_plan(
        create_recommendations(),
        create_supplier_prices(),
        budget=1500,
    )

    assert result["solver_status"] == "Optimal"

    assert (
        result["summary"]["planned_purchase_cost"]
        <= 1500
    )

    assert result["summary"]["budget_remaining"] >= 0


def test_pack_size_is_ignored():
    recommendations = pd.DataFrame(
    {
    "product_name": ["Biscuits"],
    "recommended_order_quantity": [7],
    }
    )


    supplier_prices = pd.DataFrame(
        {
            "product_name": ["Biscuits"],
            "supplier_name": ["Supplier A"],
            "unit_cost": [10.0],
            "minimum_order_quantity": [1],
            "pack_size": [5],
        }
    )

    result = optimize_purchase_plan(
        recommendations,
        supplier_prices,
        budget=70,
    )

    biscuits = result["plan"].iloc[0]

    assert biscuits["optimized_order_quantity"] == 7
    assert biscuits["planned_purchase_cost"] == 70




def test_supplier_moq_is_respected():
    result = optimize_purchase_plan(
        create_recommendations(),
        create_supplier_prices(),
        budget=1500,
    )

    plan = result["plan"]

    for _, row in plan.iterrows():

        quantity = row["optimized_order_quantity"]

        if quantity > 0:

            assert quantity >= row["minimum_order_quantity"]

def test_missing_supplier_quote_is_not_treated_as_free():
    recommendations = create_recommendations()

    prices = create_supplier_prices()

    prices = prices[
        prices["product_name"] != "Milk"
    ]

    result = optimize_purchase_plan(
        recommendations,
        prices,
        budget=1500,
    )

    plan = result["plan"]

    milk = plan[
        plan["product_name"] == "Milk"
    ].iloc[0]

    assert milk["supplier_quote_available"] == False
    assert milk["optimized_order_quantity"] == 0
    assert pd.isna(milk["planned_purchase_cost"])


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        optimize_purchase_plan(
            create_recommendations(),
            create_supplier_prices(),
            budget=-100,
        )


def test_zero_budget_produces_zero_purchases():
    result = optimize_purchase_plan(
        create_recommendations(),
        create_supplier_prices(),
        budget=0,
    )

    assert result["summary"]["planned_purchase_cost"] == 0
    assert result["summary"]["budget_remaining"] == 0
    assert result["summary"]["total_optimized_units"] == 0