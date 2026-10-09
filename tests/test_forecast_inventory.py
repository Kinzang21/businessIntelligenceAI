import numpy as np
import pytest

from src.analytics.forecast_inventory import (
    create_forecast_inventory_recommendation,
)


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