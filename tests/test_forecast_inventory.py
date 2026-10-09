import numpy as np
import pytest

from src.analytics.forecast_inventory import (
    create_forecast_inventory_recommendation,
)

from src.forecasting.explanation import build_reorder_explanation

def test_forecast_drives_reorder_recommendation():
    forecast = np.array(
        [10, 12, 11, 15, 14, 16, 12]
    )

    result = create_forecast_inventory_recommendation(
        current_stock=50,
        forecast=forecast,
        average_daily_demand=12,
        lead_time_days=5,
        safety_stock=10,
    )

    assert result["forecast_horizon_days"] == 7
    assert result["total_forecast_demand"] == 90
    assert result["forecast_daily_demand"] == pytest.approx(
        90 / 7
    )

    assert result["reorder_point"] == 70
    assert result["recommended_order_quantity"] == 50
    assert result["status"] == "Reorder"


def test_sufficient_stock():
    forecast = np.array(
        [5, 5, 5, 5, 5, 5, 5]
    )

    result = create_forecast_inventory_recommendation(
        current_stock=100,
        forecast=forecast,
        average_daily_demand=5,
        lead_time_days=5,
        safety_stock=10,
    )

    assert result["recommended_order_quantity"] == 0
    assert result["status"] == "Sufficient Stock"


def test_out_of_stock():
    forecast = np.array(
        [10, 10, 10, 10, 10, 10, 10]
    )

    result = create_forecast_inventory_recommendation(
        current_stock=0,
        forecast=forecast,
        average_daily_demand=10,
        lead_time_days=5,
        safety_stock=10,
    )

    assert result["status"] == "Out of Stock"
    assert result["recommended_order_quantity"] == 80


def test_invalid_forecast_is_rejected():
    forecast = np.array(
        [10, np.nan, 20]
    )

    with pytest.raises(ValueError):
        create_forecast_inventory_recommendation(
            current_stock=50,
            forecast=forecast,
            average_daily_demand=10,
            lead_time_days=5,
        )


def test_negative_forecast_is_rejected():
    forecast = np.array(
        [10, -5, 20]
    )

    with pytest.raises(ValueError):
        create_forecast_inventory_recommendation(
            current_stock=50,
            forecast=forecast,
            average_daily_demand=10,
            lead_time_days=5,
        )

def test_incoming_stock_reduces_recommended_order():
    forecast = np.array([10, 10, 10, 10, 10, 10, 10])

    result = create_forecast_inventory_recommendation(
        current_stock=20,
        forecast=forecast,
        average_daily_demand=10,
        lead_time_days=3,
        safety_stock=5,
        incoming_stock=15,
    )

    assert result["current_stock"] == 20
    assert result["incoming_stock"] == 15
    assert result["stock_position"] == 35
    assert result["recommended_order_quantity"] == 40


def test_supplier_minimum_order_quantity_is_respected():
    forecast = np.array([2, 2, 2, 2, 2, 2, 2])

    result = create_forecast_inventory_recommendation(
        current_stock=5,
        forecast=forecast,
        average_daily_demand=2,
        lead_time_days=2,
        safety_stock=0,
        minimum_order_quantity=12,
    )

    assert result["recommended_order_quantity"] == 12


def test_supplier_pack_size_is_respected():
    forecast = np.array([3, 3, 3, 3, 3, 3, 3])

    result = create_forecast_inventory_recommendation(
        current_stock=5,
        forecast=forecast,
        average_daily_demand=3,
        lead_time_days=2,
        safety_stock=0,
        pack_size=6,
    )

    assert result["recommended_order_quantity"] == 18


def test_incoming_stock_can_remove_reorder_requirement():
    forecast = np.array([5, 5, 5, 5, 5, 5, 5])

    result = create_forecast_inventory_recommendation(
        current_stock=20,
        forecast=forecast,
        average_daily_demand=5,
        lead_time_days=3,
        safety_stock=5,
        incoming_stock=20,
    )

    assert result["recommended_order_quantity"] == 0

def test_reorder_explanation_when_order_is_recommended():
    recommendation = {
        "current_stock": 20,
        "incoming_stock": 10,
        "stock_position": 30,
        "forecast_horizon_days": 7,
        "total_forecast_demand": 70,
        "safety_stock": 5,
        "reorder_point": 35,
        "minimum_order_quantity": 12,
        "pack_size": 6,
        "recommended_order_quantity": 48,
        "status": "Reorder",
    }

    result = build_reorder_explanation(
        product_name="Milk 1L",
        recommendation=recommendation,
    )

    assert result["product_name"] == "Milk 1L"
    assert result["inventory_status"] == "Reorder"
    assert result["recommended_order_quantity"] == 48

    assert "Forecast demand" in result["reorder_reason"]
    assert "48 units" in result["reorder_reason"]
    assert "48" in result["recommended_action"] or (
        "order" in result["recommended_action"].lower()
    )

    assert result["stock_position"] == 30
    assert result["reorder_point"] == 35


def test_reorder_explanation_when_no_order_is_needed():
    recommendation = {
        "current_stock": 100,
        "incoming_stock": 0,
        "stock_position": 100,
        "forecast_horizon_days": 7,
        "total_forecast_demand": 35,
        "safety_stock": 5,
        "reorder_point": 30,
        "minimum_order_quantity": 0,
        "pack_size": 1,
        "recommended_order_quantity": 0,
        "status": "Sufficient Stock",
    }

    result = build_reorder_explanation(
        product_name="Rice 5kg",
        recommendation=recommendation,
    )

    assert result["inventory_status"] == "Sufficient Stock"
    assert result["recommended_order_quantity"] == 0

    assert "No additional order is recommended" in (
        result["reorder_reason"]
    )

    assert "No additional order is recommended" in (
        result["recommended_action"]
    )


def test_reorder_explanation_when_product_is_out_of_stock():
    recommendation = {
        "current_stock": 0,
        "incoming_stock": 0,
        "stock_position": 0,
        "forecast_horizon_days": 7,
        "total_forecast_demand": 70,
        "safety_stock": 5,
        "reorder_point": 55,
        "minimum_order_quantity": 0,
        "pack_size": 1,
        "recommended_order_quantity": 75,
        "status": "Out of Stock",
    }

    result = build_reorder_explanation(
        product_name="Bread",
        recommendation=recommendation,
    )

    assert result["inventory_status"] == "Out of Stock"

    assert "No stock is currently available" in (
        result["stock_risk"]
    )

    assert "Prioritize replenishment" in (
        result["recommended_action"]
    )