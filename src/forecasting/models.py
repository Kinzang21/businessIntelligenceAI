import numpy as np
import pandas as pd

from .baselines import (
    naive_forecast,
    mean_forecast,
    moving_average_forecast,
    weighted_moving_average_forecast,
)


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    """
    Forecast future demand using the values from the previous season.

    Example:
        season_length=7 means Monday is predicted using the
        previous Monday, Tuesday using the previous Tuesday, etc.
    """
    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if season_length <= 0:
        raise ValueError("Season length must be greater than zero.")

    if len(history) < season_length:
        raise ValueError(
            f"At least {season_length} observations are required "
            "for seasonal naive forecasting."
        )

    recent_season = history.iloc[-season_length:].to_numpy(dtype=float)

    forecast = np.tile(
        recent_season,
        int(np.ceil(horizon / season_length)),
    )

    return forecast[:horizon]


def get_candidate_models(
    history: pd.Series,
    horizon: int,
    seasonal_period: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate forecasts from all currently available candidate models.
    """

    models = {
        "naive": naive_forecast(history, horizon),
        "mean": mean_forecast(history, horizon),
        "moving_average_7": moving_average_forecast(
            history,
            horizon,
            window=7,
        ),
        "weighted_moving_average_7": weighted_moving_average_forecast(
            history,
            horizon,
            window=7,
        ),
    }

    if seasonal_period is not None:
        if len(history) >= seasonal_period:
            models["seasonal_naive"] = seasonal_naive_forecast(
                history,
                horizon,
                seasonal_period,
            )

    return models