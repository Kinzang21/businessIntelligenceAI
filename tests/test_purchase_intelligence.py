import pandas as pd
import pytest

from src.analytics.purchase_intelligence import (
    calculate_historical_unit_cost,
    build_purchase_recommendations,
)


@pytest.fixture
def sales_data():
    return pd.DataFrame(
        {
            "product_name": [
                "Coca Cola 500ml",
                "Coca Cola 500ml",
                "Milk 1L",
            ],
            "quantity": [10, 30, 5],
            "cost_price": [40, 50, 70],
        }
    )


@pytest.fixture
def recommendations():
    return pd.DataFrame(
        {
            "product_name": [
                "Coca Cola 500ml",
                "Milk 1L",
                "Unknown Product",
            ],
            "recommended_order_quantity": [24, 12, 5],
            "status": [
                "Reorder",
                "Out of Stock",
                "Reorder",
            ],
        }
    )


def test_calculates_quantity_weighted_historical_cost(sales_data):
    result = calculate_historical_unit_cost(sales_data)

    coca_cola = result[
        result["product_key"] == "coca cola 500ml"
    ].iloc[0]

    # (10 * 40 + 30 * 50) / 40 = 47.5
    assert coca_cola["estimated_unit_cost"] == pytest.approx(47.5)


def test_estimates_purchase_costs(
    sales_data,
    recommendations,
):
    result = build_purchase_recommendations(
        recommendations,
        sales_data,
        budget=2000,
    )

    products = result["recommendations"]

    coca_cola = products[
        products["product_name"] == "Coca Cola 500ml"
    ].iloc[0]

    assert coca_cola["estimated_unit_cost"] == pytest.approx(47.5)
    assert coca_cola["estimated_purchase_cost"] == pytest.approx(1140)


def test_unknown_cost_does_not_count_as_zero(
    sales_data,
    recommendations,
):
    result = build_purchase_recommendations(
        recommendations,
        sales_data,
        budget=10000,
    )

    summary = result["summary"]

    assert summary["products_without_cost_estimate"] == 1
    assert summary["total_estimated_purchase_cost"] is None
    assert summary["budget_status"] == "Incomplete Cost Data"


def test_prioritises_out_of_stock_products(
    sales_data,
    recommendations,
):
    result = build_purchase_recommendations(
        recommendations,
        sales_data,
    )

    products = result["recommendations"]

    assert products.iloc[0]["product_name"] == "Milk 1L"
    assert products.iloc[0]["priority"] == "Critical"


def test_detects_over_budget(
    sales_data,
    recommendations,
):
    result = build_purchase_recommendations(
        recommendations,
        sales_data,
        budget=100,
    )

    assert result["summary"]["budget_status"] == "Over Budget"


def test_rejects_negative_budget(
    sales_data,
    recommendations,
):
    with pytest.raises(ValueError):
        build_purchase_recommendations(
            recommendations,
            sales_data,
            budget=-100,
        )


def test_rejects_negative_order_quantity(sales_data):
    invalid_recommendations = pd.DataFrame(
        {
            "product_name": ["Milk 1L"],
            "recommended_order_quantity": [-5],
        }
    )

    with pytest.raises(ValueError):
        build_purchase_recommendations(
            invalid_recommendations,
            sales_data,
        )

