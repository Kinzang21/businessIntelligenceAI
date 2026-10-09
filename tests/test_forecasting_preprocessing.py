import pandas as pd

from src.forecasting.preprocessing import (
    prepare_daily_demand,
    get_forecasting_data_summary,
    check_forecasting_readiness,
)
from src.forecasting.diagnostics import (
    calculate_demand_diagnostics,
    classify_demand_pattern,
)

def test_prepare_daily_demand():

    df = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-03",
            ],
            "product_name": [
                "Coca Cola",
                "Coca Cola",
                "Coca Cola",
            ],
            "quantity": [
                5,
                3,
                10,
            ],
        }
    )

    result = prepare_daily_demand(
        df,
        "Coca Cola",
    )

    assert len(result) == 3

    # Jan 2 had no transaction,
    # therefore demand must be represented as zero.
    assert result.loc[1, "demand"] == 0

    # Jan 1 had two transactions: 5 + 3
    assert result.loc[0, "demand"] == 8


def test_forecasting_summary():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=30,
            ),
            "demand": [10] * 25 + [0] * 5,
        }
    )

    summary = get_forecasting_data_summary(
        daily
    )

    assert summary["days"] == 30
    assert summary["total_demand"] == 250
    assert summary["zero_demand_days"] == 5


def test_forecasting_readiness():

    summary = {
        "days": 60,
        "total_demand": 500,
        "average_daily_demand": 8.33,
        "zero_demand_days": 5,
        "zero_demand_ratio": 0.083,
    }

    result = check_forecasting_readiness(summary)

    assert result["ready"] is True

def test_demand_diagnostics():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=30,
            ),
            "demand": [10] * 25 + [0] * 5,
        }
    )

    diagnostics = calculate_demand_diagnostics(
        daily
    )

    assert diagnostics["days"] == 30
    assert diagnostics["total_demand"] == 250
    assert diagnostics["zero_demand_days"] == 5
    assert diagnostics["zero_demand_ratio"] == 5 / 30
    assert diagnostics["intermittent"] is False


def test_intermittent_demand():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=30,
            ),
            "demand": [
                10 if i % 3 == 0 else 0
                for i in range(30)
            ],
        }
    )

    diagnostics = calculate_demand_diagnostics(
        daily
    )

    assert diagnostics["intermittent"] is True

    pattern = classify_demand_pattern(
        diagnostics
    )

    assert pattern == "intermittent"

def test_detect_recurring_weekly_pattern():

    dates = pd.date_range(
        "2026-01-01",
        periods=84,
        freq="D",
    )

    weekly_pattern = [
        10,
        12,
        11,
        13,
        15,
        40,
        45,
    ]

    demand = (
        weekly_pattern
        * 12
    )

    daily = pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

    diagnostics = calculate_demand_diagnostics(
        daily
    )

    assert "weekly" in diagnostics[
        "seasonal_periods"
    ]

    assert (
        diagnostics["strongest_period"]
        == "weekly"
    )

    assert (
        diagnostics["strongest_seasonality"]
        >= 0.60
    )

    pattern = classify_demand_pattern(
        diagnostics
    )

    assert pattern == "seasonal"