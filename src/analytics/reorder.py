import math


def calculate_reorder_point(
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock: float = 0.0,
) -> float:
    if average_daily_demand < 0:
        raise ValueError("Average daily demand cannot be negative.")

    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")

    if safety_stock < 0:
        raise ValueError("Safety stock cannot be negative.")

    return (
        average_daily_demand * lead_time_days
        + safety_stock
    )


def calculate_recommended_order_quantity(
    current_stock: float,
    forecast_demand: float,
    safety_stock: float = 0.0,
) -> int:
    if current_stock < 0:
        raise ValueError("Current stock cannot be negative.")

    if forecast_demand < 0:
        raise ValueError("Forecast demand cannot be negative.")

    if safety_stock < 0:
        raise ValueError("Safety stock cannot be negative.")

    required_stock = (
        forecast_demand
        + safety_stock
    )

    order_quantity = max(
        required_stock - current_stock,
        0,
    )

    return math.ceil(order_quantity)


def classify_reorder_status(
    current_stock: float,
    reorder_point: float,
) -> str:
    if current_stock < 0:
        raise ValueError("Current stock cannot be negative.")

    if reorder_point < 0:
        raise ValueError("Reorder point cannot be negative.")

    if current_stock <= 0:
        return "Out of Stock"

    if current_stock <= reorder_point:
        return "Reorder"

    return "Sufficient Stock"


def create_reorder_recommendation(
    current_stock: float,
    average_daily_demand: float,
    lead_time_days: int,
    forecast_demand: float,
    safety_stock: float = 0.0,
) -> dict:
    reorder_point = calculate_reorder_point(
        average_daily_demand=average_daily_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
    )

    order_quantity = calculate_recommended_order_quantity(
        current_stock=current_stock,
        forecast_demand=forecast_demand,
        safety_stock=safety_stock,
    )

    status = classify_reorder_status(
        current_stock=current_stock,
        reorder_point=reorder_point,
    )

    return {
        "current_stock": float(current_stock),
        "reorder_point": float(reorder_point),
        "recommended_order_quantity": int(order_quantity),
        "status": status,
    }