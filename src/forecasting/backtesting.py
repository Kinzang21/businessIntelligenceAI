import numpy as np
import pandas as pd

from .metrics import calculate_forecast_metrics


def rolling_backtest(
    history: pd.Series,
    forecast_function,
    horizon: int = 7,
    min_train_size: int = 30,
    step: int = 7,
    **forecast_kwargs,
) -> pd.DataFrame:
    """
    Perform rolling-origin time-series backtesting.

    Each backtest window:
    1. Uses only observations available at that point in time.
    2. Forecasts the next `horizon` observations.
    3. Compares the forecast with the actual observations.
    4. Moves forward by `step` observations.
    """

    history = pd.Series(history).astype(float).reset_index(drop=True)

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if min_train_size <= 0:
        raise ValueError(
            "Minimum training size must be greater than zero."
        )

    if step <= 0:
        raise ValueError("Step must be greater than zero.")

    if len(history) < min_train_size + horizon:
        raise ValueError(
            "History is too short for the requested backtest."
        )

    results = []

    train_end = min_train_size

    while train_end + horizon <= len(history):
        train = history.iloc[:train_end]

        actual = history.iloc[
            train_end:train_end + horizon
        ]

        predicted = forecast_function(
            train,
            horizon,
            **forecast_kwargs,
        )

        predicted = np.asarray(predicted, dtype=float)

        if len(predicted) != horizon:
            raise ValueError(
                "Forecast function returned an incorrect "
                "number of predictions."
            )

        metrics = calculate_forecast_metrics(
            actual.to_numpy(),
            predicted,
        )

        results.append(
            {
                "train_end": train_end,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape": metrics["wape"],
            }
        )

        train_end += step

    return pd.DataFrame(results)


def summarize_backtest(
    backtest_results: pd.DataFrame,
) -> dict:
    """Summarize performance and stability across backtest windows."""

    if backtest_results.empty:
        raise ValueError(
            "Backtest results cannot be empty."
        )

    wape = backtest_results["wape"]

    return {
        "windows": len(backtest_results),
        "mae": float(
            backtest_results["mae"].mean()
        ),
        "rmse": float(
            backtest_results["rmse"].mean()
        ),
        "wape": float(
            wape.mean()
        ),
        "wape_std": float(
            wape.std(ddof=0)
        ),
        "wape_min": float(
            wape.min()
        ),
        "wape_max": float(
            wape.max()
        ),
    }