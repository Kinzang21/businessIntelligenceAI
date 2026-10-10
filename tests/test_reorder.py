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

from src.analytics.reorder import (
    calculate_recommended_order_quantity,
)


def test_reorder_accounts_for_incoming_stock():
    result = calculate_recommended_order_quantity(
        current_stock=20,
        forecast_demand=60,
        safety_stock=5,
        incoming_stock=15,
    )

    assert result == 30


def test_reorder_returns_zero_when_stock_is_sufficient():
    result = calculate_recommended_order_quantity(
        current_stock=40,
        forecast_demand=30,
        safety_stock=5,
        incoming_stock=10,
    )

    assert result == 0


def test_reorder_respects_minimum_order_quantity():
    result = calculate_recommended_order_quantity(
        current_stock=5,
        forecast_demand=10,
        safety_stock=0,
        incoming_stock=0,
        minimum_order_quantity=12,
    )

    assert result == 12


def test_reorder_respects_supplier_pack_size():
    result = calculate_recommended_order_quantity(
        current_stock=0,
        forecast_demand=13,
        safety_stock=0,
        incoming_stock=0,
        pack_size=6,
    )

    assert result == 18


def test_reorder_respects_minimum_and_pack_size():
    result = calculate_recommended_order_quantity(
        current_stock=0,
        forecast_demand=13,
        safety_stock=0,
        incoming_stock=0,
        minimum_order_quantity=14,
        pack_size=6,
    )

    assert result == 18