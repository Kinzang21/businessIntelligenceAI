import numpy as np
import pandas as pd

from src.forecasting.diagnostics import (
    calculate_demand_diagnostics,
    classify_demand_pattern,
)
from src.forecasting.forecast import forecast_demand


def create_weekly_pattern_data():
    """
    Create synthetic retail demand with a repeating weekly pattern.

    The forecasting system is not told that the pattern is weekly.
    """

    rng = np.random.default_rng(42)

    dates = pd.date_range(
        "2025-01-01",
        periods=364,
        freq="D",
    )

    weekly_pattern = np.array(
        [20, 22, 21, 25, 30, 55, 60],
        dtype=float,
    )

    demand = np.tile(
        weekly_pattern,
        52,
    )

    noise = rng.normal(
        loc=0,
        scale=2,
        size=len(demand),
    )

    demand = np.maximum(
        demand + noise,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )


def create_trending_data():
    """
    Create synthetic demand with a strong upward trend
    and weekly variation.
    """

    rng = np.random.default_rng(42)

    dates = pd.date_range(
        "2025-01-01",
        periods=364,
        freq="D",
    )

    weekly_pattern = np.array(
        [10, 12, 11, 14, 16, 25, 28],
        dtype=float,
    )

    trend = np.linspace(
        0,
        30,
        len(dates),
    )

    seasonal = np.tile(
        weekly_pattern,
        52,
    )

    noise = rng.normal(
        loc=0,
        scale=1.5,
        size=len(dates),
    )

    demand = np.maximum(
        seasonal + trend + noise,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

def create_seasonal_no_trend_data():
    """
    Create demand with strong recurring seasonality but
    no underlying long-term trend.

    The system should recognize the recurring pattern
    without incorrectly calling the demand increasing
    or decreasing.
    """
    rng = np.random.default_rng(1234)

    dates = pd.date_range(
        "2025-01-01",
        periods=730,
        freq="D",
    )

    days = np.arange(len(dates))

    # Strong weekly purchasing pattern.
    weekly_pattern = np.array(
        [20, 22, 21, 25, 30, 45, 40],
        dtype=float,
    )

    weekly = np.tile(
        weekly_pattern,
        len(dates) // 7 + 1,
    )[:len(dates)]

    # Recurring annual cycle with NO long-term trend.
    annual = (
        1.0
        + 0.20
        * np.sin(
            2 * np.pi * days / 365
        )
    )

    demand = (
        weekly
        * annual
    )

    noise = rng.normal(
        loc=0,
        scale=1.5,
        size=len(dates),
    )

    demand = np.maximum(
        demand + noise,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

def create_one_year_trending_data():
    """
    Create one year of demand with a clear observed decline.

    The system should detect the direction, but should not
    claim that the decline is a persistent long-term trend
    because only one annual cycle is available.
    """
    dates = pd.date_range(
        "2025-01-01",
        periods=365,
        freq="D",
    )

    demand = np.linspace(
        30,
        20,
        len(dates),
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )


def create_short_trending_data():
    """
    Create a short historical series with a clear direction.
    The trend evidence should remain very limited.
    """
    dates = pd.date_range(
        "2025-01-01",
        periods=60,
        freq="D",
    )

    demand = np.linspace(
        30,
        20,
        len(dates),
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )


def test_one_year_trend_has_limited_long_term_evidence():
    data = create_one_year_trending_data()

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "decreasing"

    assert diagnostics["trend_evidence"] == "moderate"

    assert diagnostics["trend_confidence"] == "medium"

    assert "one annual cycle" in (
        diagnostics["trend_interpretation"]
    )


def test_short_history_has_very_limited_trend_evidence():
    data = create_short_trending_data()

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "decreasing"

    assert diagnostics["trend_evidence"] == "very_limited"

    assert diagnostics["trend_confidence"] == "very_low"


def test_multiple_years_provide_stronger_trend_evidence():
    dates = pd.date_range(
        "2022-01-01",
        periods=1095,
        freq="D",
    )

    demand = np.linspace(
        20,
        40,
        len(dates),
    )

    data = pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "increasing"

    assert diagnostics["trend_evidence"] == "strong"

    assert diagnostics["trend_confidence"] == "high"


def test_stable_demand_has_no_false_trend():
    dates = pd.date_range(
        "2025-01-01",
        periods=365,
        freq="D",
    )

    demand = np.full(
        len(dates),
        25.0,
    )

    data = pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "stable"

    assert diagnostics["trend_strength"] == 0.0

    assert diagnostics["trend_evidence"] == "moderate"

    assert diagnostics["trend_confidence"] == "medium"

def test_system_discovers_weekly_pattern():
    data = create_weekly_pattern_data()

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["strongest_period"] == "weekly"

    assert diagnostics["strongest_seasonality"] >= 0.80

    pattern = classify_demand_pattern(
        diagnostics
    )

    assert pattern == "seasonal"


def test_forecasting_engine_uses_detected_pattern():
    data = create_weekly_pattern_data()

    result = forecast_demand(
        data["demand"],
        horizon=7,
        min_train_size=60,
        step=7,
    )

    assert result["diagnostics"]["strongest_period"] == "weekly"

    assert result["best_model"] in (
        result["model_comparison"]["model"].tolist()
    )

    assert len(result["forecast"]) == 7


def test_system_detects_trending_demand():
    data = create_trending_data()

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "increasing"

    assert diagnostics["trend_strength"] > 0

def test_seasonality_without_trend_is_not_classified_as_trending():
    data = create_seasonal_no_trend_data()

    diagnostics = calculate_demand_diagnostics(
        data
    )

    assert diagnostics["trend"] == "stable"

def test_forecast_is_non_negative():
    data = create_weekly_pattern_data()

    result = forecast_demand(
        data["demand"],
        horizon=14,
        min_train_size=60,
        step=7,
    )

    assert all(
        value >= 0
        for value in result["forecast"]
    )

def create_recurring_event_pattern_data():
    """
    Synthetic demand with hidden recurring spikes.

    The forecasting system is not told what causes the spikes.
    The spikes simply repeat at approximately the same time
    each year.
    """
    rng = np.random.default_rng(123)

    dates = pd.date_range(
        "2022-01-01",
        periods=1095,
        freq="D",
    )

    demand = np.full(len(dates), 20.0)

    for year in [2022, 2023, 2024]:
        event_start = pd.Timestamp(f"{year}-06-01")

        mask = (
            (dates >= event_start)
            & (dates < event_start + pd.Timedelta(days=14))
        )

        demand[mask] += 50

    noise = rng.normal(
        loc=0,
        scale=2,
        size=len(demand),
    )

    demand = np.maximum(
        demand + noise,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )


def create_one_time_spike_data():
    """
    Synthetic demand containing a single unusual spike.

    The spike should NOT be interpreted as a recurring
    seasonal pattern because it happens only once.
    """
    rng = np.random.default_rng(456)

    dates = pd.date_range(
        "2022-01-01",
        periods=1095,
        freq="D",
    )

    demand = np.full(len(dates), 20.0)

    event_start = pd.Timestamp("2023-06-01")

    mask = (
        (dates >= event_start)
        & (dates < event_start + pd.Timedelta(days=14))
    )

    demand[mask] += 100

    noise = rng.normal(
        loc=0,
        scale=2,
        size=len(demand),
    )

    demand = np.maximum(
        demand + noise,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )


def test_system_detects_recurring_long_term_pattern():
    data = create_recurring_event_pattern_data()

    diagnostics = calculate_demand_diagnostics(data)

    assert diagnostics["strongest_period"] == "annual"
    assert diagnostics["strongest_seasonality"] >= 0.60


def test_system_does_not_treat_one_time_spike_as_annual_pattern():
    data = create_one_time_spike_data()

    diagnostics = calculate_demand_diagnostics(data)

    assert diagnostics["strongest_period"] != "annual"

def test_recurring_pattern_improves_forecast_accuracy():
    data = create_recurring_event_pattern_data()

    history = data["demand"].iloc[:-365].reset_index(drop=True)
    future = data["demand"].iloc[-365:].reset_index(drop=True)

    result = forecast_demand(
        history,
        horizon=365,
        min_train_size=365,
        step=30,
    )

    forecast = np.asarray(
        result["forecast"],
        dtype=float,
    )

    assert len(forecast) == 365

    seasonal_error = np.mean(
        np.abs(
            future.to_numpy()
            - forecast
        )
    )

    baseline_forecast = np.repeat(
        history.mean(),
        365,
    )

    baseline_error = np.mean(
        np.abs(
            future.to_numpy()
            - baseline_forecast
        )
    )

    assert seasonal_error < baseline_error

def create_realistic_retail_pattern_data():
    rng = np.random.default_rng(789)

    dates = pd.date_range(
        "2021-01-01",
        periods=1460,
        freq="D",
    )

    days = np.arange(len(dates))

    # Gradual long-term growth.
    trend = 0.015 * days

    # Weekly shopping pattern.
    weekly_pattern = np.array(
        [0.85, 0.90, 0.95, 1.00, 1.10, 1.35, 1.25]
    )
    weekly = np.tile(
        weekly_pattern,
        len(dates) // 7 + 1,
    )[:len(dates)]

    demand = (
        25
        + trend
    ) * weekly

    # Recurring annual demand spike.
    for year in range(2021, 2025):
        event_start = pd.Timestamp(
            f"{year}-06-01"
        )

        mask = (
            (dates >= event_start)
            & (
                dates
                < event_start
                + pd.Timedelta(days=14)
            )
        )

        demand[mask] *= 2.5

    # Random noise.
    noise = rng.normal(
        loc=0,
        scale=3,
        size=len(dates),
    )

    demand += noise

    # Occasional unusual spikes.
    unusual_days = rng.choice(
        len(dates),
        size=8,
        replace=False,
    )

    demand[unusual_days] *= rng.uniform(
        1.5,
        2.5,
        size=len(unusual_days),
    )

    # Some zero-demand days.
    zero_days = rng.choice(
        len(dates),
        size=20,
        replace=False,
    )

    demand[zero_days] = 0

    demand = np.maximum(
        demand,
        0,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    )

def test_forecasting_handles_realistic_retail_demand():
    data = create_realistic_retail_pattern_data()

    history = data["demand"].iloc[:-365].reset_index(drop=True)

    result = forecast_demand(
        history,
        horizon=365,
        min_train_size=365,
        step=30,
    )

    forecast = np.asarray(
        result["forecast"],
        dtype=float,
    )

    assert len(forecast) == 365

    assert np.all(
        np.isfinite(forecast)
    )

    assert np.all(
        forecast >= 0
    )

    assert result["best_model"] in (
        result["model_comparison"]["model"].tolist()
    )

    assert result["model_performance"]["wape"] >= 0

    assert result["reliability"]["score"] >= 0
    assert result["reliability"]["score"] <= 100