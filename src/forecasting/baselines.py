import pandas as pd
import numpy as np


def naive_forecast(
    history: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Forecast using the most recent observed demand.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    last_value = float(history.iloc[-1])

    return np.repeat(last_value, horizon)


def mean_forecast(
    history: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Forecast using the historical mean demand.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    mean_value = float(history.mean())

    return np.repeat(mean_value, horizon)


def moving_average_forecast(
    history: pd.Series,
    horizon: int,
    window: int = 7,
) -> np.ndarray:
    """
    Forecast using the mean of the most recent observations.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if window <= 0:
        raise ValueError("Window must be greater than zero.")

    window = min(window, len(history))

    recent_mean = float(
        history.iloc[-window:].mean()
    )

    return np.repeat(recent_mean, horizon)


def weighted_moving_average_forecast(
    history: pd.Series,
    horizon: int,
    window: int = 7,
) -> np.ndarray:
    """
    Forecast using a weighted moving average.

    More recent observations receive greater weight.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if window <= 0:
        raise ValueError("Window must be greater than zero.")

    window = min(window, len(history))

    recent = history.iloc[-window:].to_numpy(
        dtype=float
    )

    weights = np.arange(
        1,
        len(recent) + 1,
        dtype=float,
    )

    weighted_average = np.average(
        recent,
        weights=weights,
    )

    return np.repeat(
        weighted_average,
        horizon,
    )