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
    incoming_stock: float = 0.0,
    minimum_order_quantity: int = 0,
    pack_size: int = 1,
) -> dict:
    """
    Combine forecast demand with inventory and supplier constraints
    to produce an explainable replenishment recommendation.
    """

    forecast = np.asarray(forecast, dtype=float)

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

    if not np.isfinite(current_stock):
        raise ValueError(
            "Current stock must be finite."
        )

    if not np.isfinite(average_daily_demand):
        raise ValueError(
            "Average daily demand must be finite."
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

    if incoming_stock < 0 or not np.isfinite(incoming_stock):
        raise ValueError(
            "Incoming stock must be finite and non-negative."
        )

    if not isinstance(minimum_order_quantity, int):
        raise ValueError(
            "Minimum order quantity must be an integer."
        )

    if not isinstance(pack_size, int):
        raise ValueError(
            "Pack size must be an integer."
        )

    if minimum_order_quantity < 0:
        raise ValueError(
            "Minimum order quantity cannot be negative."
        )

    if pack_size < 1:
        raise ValueError(
            "Pack size must be at least 1."
        )

    forecast_horizon = len(forecast)

    total_forecast_demand = float(
        forecast.sum()
    )

    forecast_daily_demand = (
        total_forecast_demand / forecast_horizon
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
            incoming_stock=incoming_stock,
            minimum_order_quantity=minimum_order_quantity,
            pack_size=pack_size,
        )
    )

    # Use stock position when determining whether replenishment
    # is needed. Incoming stock is included to avoid duplicate orders.
    stock_position = current_stock + incoming_stock

    status = classify_reorder_status(
        current_stock=stock_position,
        reorder_point=reorder_point,
    )

    if current_stock <= 0:
        status = (
            "Out of Stock"
            if incoming_stock <= 0
            else "Incoming Stock"
        )

    return {
        "current_stock": float(current_stock),
        "incoming_stock": float(incoming_stock),
        "stock_position": float(stock_position),
        "forecast_horizon_days": int(forecast_horizon),
        "total_forecast_demand": total_forecast_demand,
        "forecast_daily_demand": forecast_daily_demand,
        "average_daily_demand": float(average_daily_demand),
        "lead_time_days": int(lead_time_days),
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "minimum_order_quantity": int(minimum_order_quantity),
        "pack_size": int(pack_size),
        "recommended_order_quantity": int(
            recommended_order_quantity
        ),
        "status": status,
    }