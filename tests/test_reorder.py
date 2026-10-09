from src.analytics.reorder import (
    calculate_reorder_point,
    calculate_recommended_order_quantity,
    classify_reorder_status,
    create_reorder_recommendation,
)


def test_reorder_point():
    result = calculate_reorder_point(
        average_daily_demand=10,
        lead_time_days=5,
        safety_stock=10,
    )

    assert result == 60


def test_recommended_order_quantity():
    result = calculate_recommended_order_quantity(
        current_stock=30,
        forecast_demand=80,
        safety_stock=10,
    )

    assert result == 60


def test_no_order_when_stock_is_sufficient():
    result = calculate_recommended_order_quantity(
        current_stock=100,
        forecast_demand=50,
        safety_stock=10,
    )

    assert result == 0


def test_reorder_status():
    assert classify_reorder_status(
        current_stock=20,
        reorder_point=30,
    ) == "Reorder"

    assert classify_reorder_status(
        current_stock=50,
        reorder_point=30,
    ) == "Sufficient Stock"

    assert classify_reorder_status(
        current_stock=0,
        reorder_point=30,
    ) == "Out of Stock"


def test_complete_reorder_recommendation():
    result = create_reorder_recommendation(
        current_stock=20,
        average_daily_demand=10,
        lead_time_days=5,
        forecast_demand=80,
        safety_stock=10,
    )

    assert result["reorder_point"] == 60
    assert result["recommended_order_quantity"] == 70
    assert result["status"] == "Reorder"