import numpy as np

from src.analytics.reorder import (
    calculate_reorder_point,
    calculate_recommended_order_quantity,
    classify_reorder_status,
)


def create_forecast_inventory_recommendation(
    current_stock: float,
    forecast: np.ndarray,
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock: float = 0.0,
) -> dict:
    """
    Combine forecasted demand with current inventory
    to produce an explainable reorder recommendation.
    """

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    if forecast.ndim != 1:
        raise ValueError(
            "Forecast must be one-dimensional."
        )

    if len(forecast) == 0:
        raise ValueError(
            "Forecast cannot be empty."
        )

    if not np.all(np.isfinite(forecast)):
        raise ValueError(
            "Forecast contains invalid values."
        )

    if np.any(forecast < 0):
        raise ValueError(
            "Forecast cannot contain negative demand."
        )

    if current_stock < 0:
        raise ValueError(
            "Current stock cannot be negative."
        )

    if average_daily_demand < 0:
        raise ValueError(
            "Average daily demand cannot be negative."
        )

    if lead_time_days < 0:
        raise ValueError(
            "Lead time cannot be negative."
        )

    if safety_stock < 0:
        raise ValueError(
            "Safety stock cannot be negative."
        )

    forecast_horizon = len(forecast)

    total_forecast_demand = float(
        forecast.sum()
    )

    forecast_daily_demand = (
        total_forecast_demand
        / forecast_horizon
    )

    reorder_point = calculate_reorder_point(
        average_daily_demand=average_daily_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
    )

    recommended_order_quantity = (
        calculate_recommended_order_quantity(
            current_stock=current_stock,
            forecast_demand=total_forecast_demand,
            safety_stock=safety_stock,
        )
    )

    status = classify_reorder_status(
        current_stock=current_stock,
        reorder_point=reorder_point,
    )

    return {
        "current_stock": float(current_stock),
        "forecast_horizon_days": forecast_horizon,
        "total_forecast_demand": total_forecast_demand,
        "forecast_daily_demand": forecast_daily_demand,
        "average_daily_demand": float(
            average_daily_demand
        ),
        "lead_time_days": int(
            lead_time_days
        ),
        "safety_stock": float(
            safety_stock
        ),
        "reorder_point": float(
            reorder_point
        ),
        "recommended_order_quantity": int(
            recommended_order_quantity
        ),
        "status": status,
    }