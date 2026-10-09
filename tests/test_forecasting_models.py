import pandas as pd
import pytest

from src.forecasting.models import (
    seasonal_naive_forecast,
    get_candidate_models,
)


def test_seasonal_naive_forecast():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70]
    )

    forecast = seasonal_naive_forecast(
        history,
        horizon=7,
        season_length=7,
    )

    assert len(forecast) == 7
    assert forecast.tolist() == [
        10, 20, 30, 40, 50, 60, 70
    ]


def test_seasonal_naive_repeats_pattern():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70]
    )

    forecast = seasonal_naive_forecast(
        history,
        horizon=14,
        season_length=7,
    )

    assert forecast.tolist() == [
        10, 20, 30, 40, 50, 60, 70,
        10, 20, 30, 40, 50, 60, 70,
    ]


def test_seasonal_naive_requires_enough_history():
    history = pd.Series([10, 20, 30])

    with pytest.raises(ValueError):
        seasonal_naive_forecast(
            history,
            horizon=7,
            season_length=7,
        )


def test_candidate_models():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70] * 4
    )

    models = get_candidate_models(
        history,
        horizon=7,
        seasonal_period=7,
    )

    assert "naive" in models
    assert "mean" in models
    assert "moving_average_7" in models
    assert "weighted_moving_average_7" in models
    assert "seasonal_naive" in models

    for forecast in models.values():
        assert len(forecast) == 7