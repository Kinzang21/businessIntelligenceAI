from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pulp


def optimize_purchase_plan(
    recommendations_df: pd.DataFrame,
    supplier_prices_df: pd.DataFrame,
    budget: float,
    time_limit_seconds: int = 30,
) -> dict:
    """
    Optimize purchasing quantities within a user-defined budget.

    The optimizer:
    - Uses actual supplier-quoted unit costs.
    - Respects the forecast-based recommended quantities.
    - Accounts for current and incoming stock through the existing
    - forecast/reorder recommendations.
    - Respects minimum order quantities.
    - optimizes individual units without pack-size restrictions.
    - Maximizes priority-weighted purchasing coverage.
    - Never intentionally exceeds the allocated budget.

    Required recommendation columns:
        product_name
        recommended_order_quantity

    Optional recommendation columns:
        current_stock
        forecast_demand
        incoming_stock
        stock_risk
        reliability_score
        status

    Required supplier-price columns:
        product_name
        supplier_name
        unit_cost

    Optional supplier-price columns:
        minimum_order_quantity

    Returns:
        {
            "plan": DataFrame,
            "summary": dict,
            "solver_status": str,
        }
    """

    # ---------------------------------------------------------
    # 1. VALIDATE INPUTS
    # ---------------------------------------------------------

    try:
        budget = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Budget must be a valid non-negative number."
        ) from exc

    if not math.isfinite(budget) or budget < 0:
        raise ValueError(
            "Budget must be a valid non-negative number."
        )

    if time_limit_seconds < 1:
        raise ValueError(
            "Time limit must be at least one second."
        )

    recommendation_required = {
        "product_name",
        "recommended_order_quantity",
    }

    missing = recommendation_required - set(
        recommendations_df.columns
    )

    if missing:
        raise ValueError(
            "Recommendations are missing required columns: "
            f"{sorted(missing)}"
        )

    supplier_required = {
        "product_name",
        "supplier_name",
        "unit_cost",
    }

    missing = supplier_required - set(
        supplier_prices_df.columns
    )

    if missing:
        raise ValueError(
            "Supplier prices are missing required columns: "
            f"{sorted(missing)}"
        )

    recommendations = recommendations_df.copy()
    prices = supplier_prices_df.copy()

    if recommendations.empty:
        raise ValueError(
            "There are no product recommendations to optimize."
        )

    if recommendations["product_name"].isna().any():
        raise ValueError(
            "Recommendation product names cannot be missing."
        )

    if prices["product_name"].isna().any():
        raise ValueError(
            "Supplier-price product names cannot be missing."
        )

    def normalize(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().str.casefold()

    recommendations["_key"] = normalize(
        recommendations["product_name"]
    )

    prices["_key"] = normalize(prices["product_name"])

    if recommendations["_key"].eq("").any():
        raise ValueError(
            "Recommendation product names cannot be empty."
        )

    if prices["_key"].eq("").any():
        raise ValueError(
            "Supplier-price product names cannot be empty."
        )

    if recommendations["_key"].duplicated().any():
        raise ValueError(
            "Recommendations contain duplicate product names."
        )

    # This initial implementation accepts one supplier quotation
    # per product. Supplier comparison can be added separately.
    if prices["_key"].duplicated().any():
        raise ValueError(
            "Provide exactly one selected supplier quotation "
            "per product in this version."
        )

    recommendations["recommended_order_quantity"] = pd.to_numeric(
        recommendations["recommended_order_quantity"],
        errors="coerce",
    )

    required_quantities = recommendations[
        "recommended_order_quantity"
    ]

    if (
        required_quantities.isna().any()
        or not np.isfinite(required_quantities).all()
        or (required_quantities < 0).any()
    ):
        raise ValueError(
            "Recommended quantities must be finite, "
            "non-negative numbers."
        )

    prices["unit_cost"] = pd.to_numeric(
        prices["unit_cost"],
        errors="coerce",
    )

    if (
        prices["unit_cost"].isna().any()
        or not np.isfinite(prices["unit_cost"]).all()
        or (prices["unit_cost"] <= 0).any()
    ):
        raise ValueError(
            "Supplier unit costs must be finite positive numbers."
        )

    # ---------------------------------------------------------
    # 2. PREPARE SUPPLIER AND RECOMMENDATION DATA
    # ---------------------------------------------------------

    for column, default in [
        ("minimum_order_quantity", 1),
    ]:
        if column not in prices.columns:
            prices[column] = default

        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce",
        )

        if (
            prices[column].isna().any()
            or not np.isfinite(prices[column]).all()
            or (prices[column] < 1).any()
            or (prices[column] % 1 != 0).any()
        ):
            raise ValueError(
                f"{column} must contain positive integers."
            )

        prices[column] = prices[column].astype(int)

    # Remove supplier quotation fields from recommendations
    # so the selected supplier quotation remains authoritative.
    recommendations = recommendations.drop(
        columns=[
            "supplier_name",
            "unit_cost",
            "minimum_order_quantity",
        ],
        errors="ignore",
    )

    data = recommendations.merge(
        prices[
            [
                "_key",
                "supplier_name",
                "unit_cost",
                "minimum_order_quantity",
            ]
        ],
        on="_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    data["supplier_quote_available"] = (
        data["_merge"] == "both"
    )

    data = data.drop(columns="_merge")

    # ---------------------------------------------------------
    # 3. DERIVE PURCHASE LIMITS AND PRIORITY WEIGHTS
    # ---------------------------------------------------------

    def numeric_column(
        frame: pd.DataFrame,
        column: str,
        default: float = 0.0,
    ) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(
                default,
                index=frame.index,
                dtype=float,
            )

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        return values.fillna(default).astype(float)

    data["current_stock_numeric"] = numeric_column(
        data, "current_stock"
    )

    data["incoming_stock_numeric"] = numeric_column(
        data, "incoming_stock"
    )

    data["forecast_demand_numeric"] = numeric_column(
        data, "forecast_demand"
    )

    data["forecast_demand_available"] = (
        data["forecast_demand_numeric"] > 0
    )

    # The existing forecast engine supplies the maximum recommended
    # order quantity. The optimizer may reduce it but will not increase
    # it in this initial version.
    data["purchase_limit"] = np.floor(
        data["recommended_order_quantity"]
    ).astype(int)

    # Priority weights are explicit business rules, not learned
    # probabilities or claims of guaranteed future demand.
    weights = []

    for _, row in data.iterrows():

        status = str(row.get("status", "")).casefold()
        risk = str(row.get("stock_risk", "")).casefold()

        weight = 1.0

        if "critical" in risk or "out of stock" in status:
            weight = 5.0
        elif "high" in risk or "reorder" in status:
            weight = 3.0
        elif "medium" in risk or "low stock" in status:
            weight = 2.0

        weights.append(weight)

    data["priority_weight"] = weights

        # ---------------------------------------------------------
    # 4. BUILD THE INTEGER OPTIMIZATION MODEL
    # ---------------------------------------------------------

    model = pulp.LpProblem(
        "Budget_Constrained_Purchase_Optimization",
        pulp.LpMaximize,
    )

    quantity_variables = {}
    order_variables = {}

    eligible_indices = data.index[
        data["supplier_quote_available"]
        & (data["purchase_limit"] > 0)
        & (
            data["purchase_limit"]
            >= data["minimum_order_quantity"]
        )
    ].tolist()

    # Missing quotations are excluded.
    # Optimize individual units; pack_size is not used.
    for index in eligible_indices:

        row = data.loc[index]

        limit = int(row["purchase_limit"])
        minimum_order = int(row["minimum_order_quantity"])

        quantity = pulp.LpVariable(
            f"quantity_{index}",
            lowBound=0,
            upBound=limit,
            cat=pulp.LpInteger,
        )

        ordered = pulp.LpVariable(
            f"ordered_{index}",
            cat=pulp.LpBinary,
        )

        quantity_variables[index] = quantity
        order_variables[index] = ordered

        # If an order is placed, at least the MOQ is required.
        model += (
            quantity >= minimum_order * ordered,
            f"MOQ_{index}",
        )

        # Prevent ordering more than the recommendation,
        # and force quantity to zero when no order is placed.
        model += (
            quantity <= limit * ordered,
            f"OrderLimit_{index}",
        )

    # Hard purchasing-budget constraint using actual unit prices.
    total_cost = pulp.lpSum(
        quantity_variables[index]
        * float(data.loc[index, "unit_cost"])
        for index in quantity_variables
    )

    model += total_cost <= budget, "PurchasingBudget"

    # Maximize priority-weighted individual units purchased.
    model += pulp.lpSum(
        quantity_variables[index]
        * float(data.loc[index, "priority_weight"])
        for index in quantity_variables
    )    

    # ---------------------------------------------------------
    # 5. SOLVE
    # ---------------------------------------------------------

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=time_limit_seconds,
    )

    model.solve(solver)

    solver_status = pulp.LpStatus[model.status]

    if solver_status not in {"Optimal", "Not Solved"}:
        raise RuntimeError(
            "The purchasing optimization did not produce "
            f"a usable solution. Solver status: {solver_status}"
        )

    # Do not use an unproven time-limited solution as though it
    # were guaranteed optimal.
    if solver_status != "Optimal":
        raise RuntimeError(
            "The solver did not prove optimality within the time "
            "limit. Increase time_limit_seconds and try again."
        )

    # ---------------------------------------------------------
    # 6. BUILD THE PRODUCT-WISE PLAN
    # ---------------------------------------------------------

    optimized_quantities = []
    reasons = []

    for index, row in data.iterrows():

        original_quantity = int(row["purchase_limit"])

        if not row["supplier_quote_available"]:

            optimized_quantity = 0
            reason = (
                "No valid supplier quotation; excluded "
                "from optimization."
            )

        elif original_quantity == 0:

            optimized_quantity = 0
            reason = (
                "No purchase was recommended by the "
                "forecasting engine."
            )

        elif (
            original_quantity
            < int(row["minimum_order_quantity"])
        ):

            optimized_quantity = 0
            reason = (
                "Recommended quantity is below the "
                "supplier's minimum order quantity."
            )

        elif index not in quantity_variables:

            optimized_quantity = 0
            reason = (
                "No feasible order selected within "
                "the supplier and budget constraints."
            )

        else:

            quantity_value = pulp.value(
                quantity_variables[index]
            )

            optimized_quantity = int(
                round(quantity_value or 0)
            )

            if optimized_quantity == original_quantity:

                reason = (
                    "Full recommended quantity fits the "
                    "optimized plan."
                )

            elif optimized_quantity > 0:

                reason = (
                    "Quantity reduced to fit the budget "
                    "and supplier constraints."
                )

            else:

                reason = (
                    "No feasible order selected within "
                    "the allocated budget."
                )

        optimized_quantities.append(optimized_quantity)
        reasons.append(reason)

    data["optimized_order_quantity"] = optimized_quantities

    data["planned_purchase_cost"] = (
        data["optimized_order_quantity"]
        * data["unit_cost"].fillna(0)
    )

    # Missing quotes must not be interpreted as zero-cost purchases.
    data.loc[
        ~data["supplier_quote_available"],
        "planned_purchase_cost",
    ] = np.nan

    data["optimization_explanation"] = reasons

    data["unfulfilled_recommended_quantity"] = (
        data["purchase_limit"]
        - data["optimized_order_quantity"]
    )

    # ---------------------------------------------------------
    # 7. CALCULATE THE SUMMARY
    # ---------------------------------------------------------

    ordered_rows = data[
        data["optimized_order_quantity"] > 0
    ]

    total_planned_cost = float(
        data["planned_purchase_cost"].fillna(0).sum()
    )

    budget_remaining = budget - total_planned_cost

    if budget_remaining < 0 and abs(budget_remaining) < 1e-7:
        budget_remaining = 0.0

    missing_quote_count = int(
        (
            (data["purchase_limit"] > 0)
            & (~data["supplier_quote_available"])
        ).sum()
    )

    infeasible_count = int(
        (
            (data["purchase_limit"] > 0)
            & data["supplier_quote_available"]
            & (data["optimized_order_quantity"] == 0)
        ).sum()
    )

    total_recommended_units = int(
        data["purchase_limit"].sum()
    )

    total_optimized_units = int(
        data["optimized_order_quantity"].sum()
    )

    summary = {
        "allocated_budget": budget,
        "planned_purchase_cost": total_planned_cost,
        "budget_remaining": budget_remaining,
        "budget_utilisation_pct": (
            total_planned_cost / budget * 100
            if budget > 0
            else None
        ),
        "products_analysed": int(len(data)),
        "products_to_order": int(len(ordered_rows)),
        "products_missing_supplier_quotes": missing_quote_count,
        "products_not_ordered": infeasible_count,
        "total_recommended_units": total_recommended_units,
        "total_optimized_units": total_optimized_units,
        "unfulfilled_recommended_units": int(
            data["unfulfilled_recommended_quantity"].sum()
        ),
    }

    output_columns = [
        column
        for column in [
            "product_name",
            "supplier_name",
            "unit_cost",
            "minimum_order_quantity",
            "recommended_order_quantity",
            "optimized_order_quantity",
            "planned_purchase_cost",
            "current_stock",
            "incoming_stock",
            "forecast_demand",
            "status",
            "stock_risk",
            "reliability_score",
            "priority_weight",
            "supplier_quote_available",
            "unfulfilled_recommended_quantity",
            "optimization_explanation",
        ]
        if column in data.columns
    ]

    plan = data[output_columns].copy()

    return {
        "plan": plan,
        "summary": summary,
        "solver_status": solver_status,
    }