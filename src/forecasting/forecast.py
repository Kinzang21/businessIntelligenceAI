import pandas as pd
from .reliability import calculate_reliability_score

from .baselines import (
    naive_forecast,
    mean_forecast,
    moving_average_forecast,
    weighted_moving_average_forecast,
)
from .diagnostics import calculate_demand_diagnostics
from .model_selection import evaluate_models, select_best_model
from .models import seasonal_naive_forecast
from src.forecasting import diagnostics

from src.forecasting import models

from src.forecasting import models


def build_candidate_models(
    seasonal_period: int | None = None,
) -> dict:
    """
    Build the forecasting model registry.

    Each value is a forecasting function that can be passed
    to the backtesting system.
    """

    models = {
        "naive": naive_forecast,
        "mean": mean_forecast,
        "moving_average_7": (
            lambda history, horizon:
            moving_average_forecast(
                history,
                horizon,
                window=7,
            )
        ),
        "weighted_moving_average_7": (
            lambda history, horizon:
            weighted_moving_average_forecast(
                history,
                horizon,
                window=7,
            )
        ),
    }

    if seasonal_period is not None:
        models["seasonal_naive"] = (
            lambda history, horizon:
            seasonal_naive_forecast(
                history,
                horizon,
                season_length=seasonal_period,
            )
        )

    return models


def forecast_demand(
    history: pd.Series,
    horizon: int = 7,
    min_train_size: int = 30,
    step: int = 7,
) -> dict:
    """
    Automatically diagnose, evaluate, select, and forecast.

    Returns the selected model, diagnostics, performance,
    and future forecast.
    """

    history = pd.Series(history).astype(float).reset_index(drop=True)

    if history.empty:
        raise ValueError("History cannot be empty.")

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    diagnostics_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2000-01-01",
                periods=len(history),
                freq="D",
            ),
            "demand": history,
        }
    )

    diagnostics = calculate_demand_diagnostics(
        diagnostics_df
    )

    seasonal_period = None

    if diagnostics["strongest_period"] is not None:
        detected_period = diagnostics["seasonal_periods"][
            diagnostics["strongest_period"]
        ]["period"]

    # A seasonal model must have enough observations
    # for the first rolling-backtest training window
    # plus one complete seasonal cycle.
        minimum_seasonal_history = min_train_size + detected_period

        if len(history) >= minimum_seasonal_history:
            seasonal_period = detected_period

    models = build_candidate_models(
        seasonal_period=seasonal_period
    )

    backtest_min_train_size = min_train_size

    if seasonal_period is not None:
        backtest_min_train_size = max(
            min_train_size,
            seasonal_period,
        )

    comparison = evaluate_models(
        history=history,
        models=models,
        horizon=horizon,
        min_train_size=backtest_min_train_size,
        step=step,
    )

    best_model = select_best_model(comparison)

    best_model_performance = comparison[
    comparison["model"] == best_model
    ].iloc[0]

    model_performance = {
        "wape": float(best_model_performance["wape"]),
        "mae": float(best_model_performance["mae"]),
        "rmse": float(best_model_performance["rmse"]),
        "windows": int(best_model_performance["windows"]),
        "wape_std": float(best_model_performance["wape_std"]),
        "wape_min": float(best_model_performance["wape_min"]),
        "wape_max": float(best_model_performance["wape_max"]),
    }

    reliability = calculate_reliability_score(
        diagnostics,
        model_performance,
    )

    forecast_function = models[best_model]

    forecast = forecast_function(
        history,
        horizon,
    )
    
    return {
    "forecast": forecast,
    "best_model": best_model,
    "diagnostics": diagnostics,
    "model_comparison": comparison,
    "model_performance": model_performance,
    "reliability": reliability,
}