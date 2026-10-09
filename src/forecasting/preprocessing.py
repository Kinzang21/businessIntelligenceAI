import pandas as pd


def prepare_daily_demand(
    df: pd.DataFrame,
    product_name: str,
) -> pd.DataFrame:
    """
    Convert transaction-level sales data into a continuous
    daily demand time series for one product.
    """

    required_columns = {"date", "product_name", "quantity"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required forecasting columns: "
            + ", ".join(sorted(missing))
        )

    data = df.copy()

    # Convert date and quantity
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["quantity"] = pd.to_numeric(
        data["quantity"],
        errors="coerce",
    )

    # Remove invalid rows
    data = data.dropna(
        subset=["date", "quantity", "product_name"]
    )

    # Select product
    data = data[
        data["product_name"].astype(str) == str(product_name)
    ]

    if data.empty:
        raise ValueError(
            f"No sales data found for product: {product_name}"
        )

    # Aggregate transactions occurring on the same day
    daily = (
        data.groupby("date", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "demand"})
    )

    # Create continuous daily date range
    full_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    daily["demand"] = daily["demand"].astype(float)

    return daily


def get_forecasting_data_summary(
    daily_demand: pd.DataFrame,
) -> dict:
    """
    Generate basic diagnostics used to determine
    whether forecasting is appropriate.
    """

    if daily_demand.empty:
        return {
            "days": 0,
            "total_demand": 0,
            "average_daily_demand": 0,
            "zero_demand_days": 0,
            "zero_demand_ratio": 0,
        }

    total_demand = daily_demand["demand"].sum()
    days = len(daily_demand)
    zero_days = int(
        (daily_demand["demand"] == 0).sum()
    )

    return {
        "days": days,
        "total_demand": float(total_demand),
        "average_daily_demand": float(
            daily_demand["demand"].mean()
        ),
        "zero_demand_days": zero_days,
        "zero_demand_ratio": float(zero_days / days),
    }


def check_forecasting_readiness(
    summary: dict,
    minimum_days: int = 30,
) -> dict:
    """
    Determine whether a product has enough historical
    data for forecasting.
    """

    days = summary["days"]

    if days < minimum_days:
        return {
            "ready": False,
            "reason": (
                f"Only {days} days of history available. "
                f"At least {minimum_days} days are recommended."
            ),
        }

    if summary["total_demand"] <= 0:
        return {
            "ready": False,
            "reason": "No recorded demand for this product.",
        }

    return {
        "ready": True,
        "reason": "Sufficient historical data available.",
    }