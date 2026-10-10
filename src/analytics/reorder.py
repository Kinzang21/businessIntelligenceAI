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
    incoming_stock: float = 0.0,
    minimum_order_quantity: int = 0,
    pack_size: int = 1,
) -> int:
    """
    Calculate an order quantity using forecast demand,
    available inventory, incoming stock, and supplier constraints.

    Parameters
    ----------
    current_stock:
        Units currently available in inventory.

    forecast_demand:
        Expected demand over the planning horizon.

    safety_stock:
        Additional stock reserved to reduce stockout risk.

    incoming_stock:
        Units already ordered and expected to arrive.

    minimum_order_quantity:
        Supplier's minimum order quantity.

    pack_size:
        Number of units in one supplier pack.
        The recommended order must be a multiple of this value.

    Returns
    -------
    int
        Recommended quantity to order.
    """

    # Validate numeric quantities.
    if current_stock < 0:
        raise ValueError(
            "Current stock cannot be negative."
        )

    if forecast_demand < 0:
        raise ValueError(
            "Forecast demand cannot be negative."
        )

    if safety_stock < 0:
        raise ValueError(
            "Safety stock cannot be negative."
        )

    if incoming_stock < 0:
        raise ValueError(
            "Incoming stock cannot be negative."
        )

    if minimum_order_quantity < 0:
        raise ValueError(
            "Minimum order quantity cannot be negative."
        )

    if pack_size < 1:
        raise ValueError(
            "Pack size must be at least 1."
        )

    if not isinstance(minimum_order_quantity, int):
        raise ValueError(
            "Minimum order quantity must be an integer."
        )

    if not isinstance(pack_size, int):
        raise ValueError(
            "Pack size must be an integer."
        )

    # Calculate the stock required for the planning horizon.
    required_stock = (
        forecast_demand + safety_stock
    )

    # Account for stock already available or expected to arrive.
    available_stock = (
        current_stock + incoming_stock
    )

    # Calculate the remaining quantity needed.
    net_requirement = max(
        required_stock - available_stock,
        0,
    )

    # No order is necessary if existing and incoming stock
    # can cover the requirement.
    if net_requirement == 0:
        return 0

    # Round up to whole units.
    order_quantity = math.ceil(
        net_requirement
    )

    # Respect the supplier's minimum order quantity.
    order_quantity = max(
        order_quantity,
        minimum_order_quantity,
    )

    # Round up to the next valid supplier pack multiple.
    order_quantity = (
        math.ceil(order_quantity / pack_size)
        * pack_size
    )

    return int(order_quantity)

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