import numpy as np
import pytest

from src.forecasting.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
    calculate_forecast_metrics,
)


def test_mae():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    result = mean_absolute_error(actual, predicted)

    assert result == pytest.approx(7 / 3)


def test_rmse():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    result = root_mean_squared_error(actual, predicted)

    expected = np.sqrt((4 + 4 + 9) / 3)

    assert result == pytest.approx(expected)


def test_wape():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    result = weighted_absolute_percentage_error(
        actual,
        predicted,
    )

    expected = 7 / 60 * 100

    assert result == pytest.approx(expected)


def test_all_forecast_metrics():
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 320])

    metrics = calculate_forecast_metrics(
        actual,
        predicted,
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wape" in metrics

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["wape"] > 0


def test_mismatched_lengths():
    actual = np.array([10, 20, 30])
    predicted = np.array([10, 20])

    with pytest.raises(ValueError):
        mean_absolute_error(actual, predicted)

    with pytest.raises(ValueError):
        root_mean_squared_error(actual, predicted)

    with pytest.raises(ValueError):
        weighted_absolute_percentage_error(actual, predicted)