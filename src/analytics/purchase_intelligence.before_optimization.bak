
import numpy as np
import pandas as pd


def _normalise_product_names(series: pd.Series) -> pd.Series:
    """Standardise product names for safer matching."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
    )


def calculate_historical_unit_cost(
    sales_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate each product's unit cost from historical sales records.

    Uses a quantity-weighted average of cost_price.

    This is a historical cost estimate, not a live supplier quotation.
    """

    required_columns = {
        "product_name",
        "quantity",
        "cost_price",
    }

    missing = required_columns - set(sales_df.columns)

    if missing:
        raise ValueError(
            f"Sales data is missing required columns: {sorted(missing)}"
        )

    sales = sales_df.copy()

    sales["quantity"] = pd.to_numeric(
        sales["quantity"], errors="coerce"
    )

    sales["cost_price"] = pd.to_numeric(
        sales["cost_price"], errors="coerce"
    )

    sales = sales.dropna(
        subset=["product_name", "quantity", "cost_price"]
    )

    # Invalid records should not influence the cost estimate.
    sales = sales[
        (sales["quantity"] > 0)
        & (sales["cost_price"] >= 0)
    ].copy()

    if sales.empty:
        return pd.DataFrame(
            columns=[
                "product_key",
                "estimated_unit_cost",
                "cost_records",
            ]
        )

    sales["product_key"] = _normalise_product_names(
        sales["product_name"]
    )

    sales["weighted_cost"] = (
        sales["quantity"] * sales["cost_price"]
    )

    costs = (
        sales.groupby("product_key", as_index=False)
        .agg(
            total_weighted_cost=("weighted_cost", "sum"),
            total_units=("quantity", "sum"),
            cost_records=("cost_price", "count"),
        )
    )

    costs["estimated_unit_cost"] = (
        costs["total_weighted_cost"] / costs["total_units"]
    )

    return costs[
        [
            "product_key",
            "estimated_unit_cost",
            "cost_records",
        ]
    ]


def build_purchase_recommendations(
    recommendations_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    budget: float | None = None,
) -> dict:
    """
    Enrich existing forecast/reorder recommendations with purchasing costs.

    Parameters
    ----------
    recommendations_df:
        Output from the existing product-level forecasting service.

    sales_df:
        Historical sales records containing quantity and cost_price.

    budget:
        Optional purchasing budget in the same currency as cost_price.

    Returns
    -------
    dict containing:
        recommendations: product-level purchase recommendations
        summary: overall purchasing and budget metrics

    This function estimates costs only. It does not change the existing
    recommended order quantities to force them within a budget.
    """

    if budget is not None:
        try:
            budget = float(budget)
        except (TypeError, ValueError) as exc:
            raise ValueError("Budget must be a valid non-negative number.") from exc

        if not np.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be a valid non-negative number.")

    required_columns = {
        "product_name",
        "recommended_order_quantity",
    }

    missing = required_columns - set(recommendations_df.columns)

    if missing:
        raise ValueError(
            "Recommendations data is missing required columns: "
            f"{sorted(missing)}"
        )

    recommendations = recommendations_df.copy()

    if recommendations["product_name"].isna().any():
        raise ValueError("Product names cannot be missing.")

    if recommendations["product_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Product names cannot be empty.")

    recommendations["product_key"] = _normalise_product_names(
        recommendations["product_name"]
    )

    if recommendations["product_key"].duplicated().any():
        raise ValueError(
            "Recommendations contain duplicate product names. "
            "Resolve duplicates before calculating purchasing costs."
        )

    recommendations["recommended_order_quantity"] = pd.to_numeric(
        recommendations["recommended_order_quantity"],
        errors="coerce",
    )

    quantities = recommendations["recommended_order_quantity"]

    if (
        quantities.isna().any()
        or not np.isfinite(quantities).all()
        or (quantities < 0).any()
    ):
        raise ValueError(
            "Recommended order quantities must be finite, "
            "non-negative numbers."
        )

    historical_costs = calculate_historical_unit_cost(sales_df)

    recommendations = recommendations.merge(
        historical_costs,
        on="product_key",
        how="left",
        validate="one_to_one",
    )

    # Calculate the estimated cost only when a historical unit cost exists.
    recommendations["estimated_purchase_cost"] = (
        recommendations["recommended_order_quantity"]
        * recommendations["estimated_unit_cost"]
    )

    recommendations["cost_estimate_available"] = (
        recommendations["estimated_unit_cost"].notna()
    )

    # Transparent prioritisation based on existing recommendations.
    status = (
        recommendations["status"].astype("string").str.casefold()
        if "status" in recommendations.columns
        else pd.Series(
            "unknown",
            index=recommendations.index,
            dtype="string",
        )
    )

    recommendations["priority"] = np.select(
        [
            recommendations["recommended_order_quantity"].eq(0),
            status.str.contains("out of stock", na=False),
            status.str.contains("reorder|low stock", na=False),
            status.str.contains("incoming stock", na=False),
        ],
        [
            "No Order Needed",
            "Critical",
            "High",
            "Medium",
        ],
        default="Normal",
    )

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Normal": 3,
        "No Order Needed": 4,
    }

    recommendations["_priority_order"] = (
        recommendations["priority"].map(priority_order)
    )

    recommendations = recommendations.sort_values(
        ["_priority_order", "product_name"],
        kind="stable",
    ).drop(columns="_priority_order").reset_index(drop=True)

    order_required = recommendations[
        recommendations["recommended_order_quantity"] > 0
    ]

    known_costs = order_required["estimated_purchase_cost"].dropna()

    total_estimated_cost = (
        float(known_costs.sum())
        if len(known_costs) == len(order_required)
        else None
    )

    products_to_order = int(
        (recommendations["recommended_order_quantity"] > 0).sum()
    )

    products_without_cost = int(
        (
            (recommendations["recommended_order_quantity"] > 0)
            & (~recommendations["cost_estimate_available"])
        ).sum()
    )

    if budget is None:
        budget_status = "Budget Not Provided"
        budget_remaining = None
        budget_utilisation_pct = None

    elif total_estimated_cost is not None:
        # Every product requiring an order has a known cost.
        budget_remaining = budget - total_estimated_cost

        budget_status = (
            "Within Budget"
            if total_estimated_cost <= budget
            else "Over Budget"
        )

        budget_utilisation_pct = (
            total_estimated_cost / budget * 100
            if budget > 0
            else (0.0 if total_estimated_cost == 0 else None)
        )

    else:
        # Some product costs are missing. Check whether the known costs
        # alone already exceed the budget.
        known_order_cost = float(known_costs.sum())

        if known_order_cost > budget:
            budget_status = "Over Budget"
            budget_remaining = budget - known_order_cost
            budget_utilisation_pct = (
                known_order_cost / budget * 100
                if budget > 0
                else None
            )
        else:
            budget_status = "Incomplete Cost Data"
            budget_remaining = None
            budget_utilisation_pct = None

    summary = {
        "products_analysed": int(len(recommendations)),
        "products_to_order": products_to_order,
        "products_without_cost_estimate": products_without_cost,
        "total_units_to_order": int(
            order_required["recommended_order_quantity"].sum()
        ),
        "total_estimated_purchase_cost": total_estimated_cost,
        "budget": budget,
        "budget_status": budget_status,
        "budget_remaining": budget_remaining,
        "budget_utilisation_pct": budget_utilisation_pct,
    }

    # Keep rows with unknown costs visible. Do not silently treat unknown
    # costs as zero when calculating the overall purchasing budget.
    return {
        "recommendations": recommendations.drop(
            columns="product_key"
        ),
        "summary": summary,
    }
