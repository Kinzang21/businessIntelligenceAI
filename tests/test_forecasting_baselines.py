import pandas as pd
import numpy as np

from src.forecasting.baselines import (
    naive_forecast,
    mean_forecast,
    moving_average_forecast,
    weighted_moving_average_forecast,
)


def test_naive_forecast():

    history = pd.Series(
        [10, 12, 15, 20]
    )

    forecast = naive_forecast(
        history,
        horizon=3,
    )

    expected = np.array(
        [20, 20, 20]
    )

    np.testing.assert_array_equal(
        forecast,
        expected,
    )


def test_mean_forecast():

    history = pd.Series(
        [10, 20, 30, 40]
    )

    forecast = mean_forecast(
        history,
        horizon=2,
    )

    expected = np.array(
        [25, 25]
    )

    np.testing.assert_array_equal(
        forecast,
        expected,
    )


def test_moving_average():

    history = pd.Series(
        [10, 20, 30, 40, 50]
    )

    forecast = moving_average_forecast(
        history,
        horizon=2,
        window=3,
    )

    expected = np.array(
        [40, 40]
    )

    np.testing.assert_array_equal(
        forecast,
        expected,
    )


def test_weighted_moving_average():

    history = pd.Series(
        [10, 20, 30]
    )

    forecast = weighted_moving_average_forecast(
        history,
        horizon=2,
        window=3,
    )

    expected_value = (
        (10 * 1) +
        (20 * 2) +
        (30 * 3)
    ) / 6

    expected = np.array(
        [expected_value, expected_value]
    )

    np.testing.assert_allclose(
        forecast,
        expected,
    )