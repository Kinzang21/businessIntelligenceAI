import pandas as pd

from src.forecasting.explanation import (
    build_forecast_explanation,
    build_reorder_explanation, 
)
from src.analytics.forecast_inventory import (
    create_forecast_inventory_recommendation,
)
from src.forecasting.forecast import forecast_demand
from src.forecasting.preprocessing import (
    check_forecasting_readiness,
    get_forecasting_data_summary,
    prepare_daily_demand,
)


def forecast_products(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    horizon: int = 14,
    lead_time_days: int = 5,
    safety_stock: float = 0.0,
    min_history_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate product-level forecasts and inventory recommendations.

    Optional inventory columns:
        product_id
        supplier_id
        supplier_name
        lead_time_days
        safety_stock
        incoming_stock
        minimum_order_quantity
        pack_size

    Products are matched by product_id when both datasets have it.
    Otherwise, matching falls back to product_name.

    Returns:
        successful_results:
            Forecasts and inventory recommendations.

        skipped_products:
            Products that could not be processed, with reasons.
    """

    required_sales_columns = {
        "date",
        "product_name",
        "quantity",
    }

    required_inventory_columns = {
        "product_name",
        "closing_stock",
    }

    missing_sales = (
        required_sales_columns - set(sales_df.columns)
    )

    if missing_sales:
        raise ValueError(
            "Missing required sales columns: "
            + ", ".join(sorted(missing_sales))
        )

    missing_inventory = (
        required_inventory_columns - set(inventory_df.columns)
    )

    if missing_inventory:
        raise ValueError(
            "Missing required inventory columns: "
            + ", ".join(sorted(missing_inventory))
        )

    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    if lead_time_days < 0:
        raise ValueError(
            "Lead time cannot be negative."
        )

    if safety_stock < 0:
        raise ValueError(
            "Safety stock cannot be negative."
        )

    sales = sales_df.copy()
    inventory = inventory_df.copy()

    # Normalize product names for matching.
    sales["product_name"] = (
        sales["product_name"]
        .astype("string")
        .str.strip()
    )

    inventory["product_name"] = (
        inventory["product_name"]
        .astype("string")
        .str.strip()
    )

    # Use product IDs as the preferred matching key when
    # both datasets provide them.
    use_product_id = (
        "product_id" in sales.columns
        and "product_id" in inventory.columns
    )

    if use_product_id:
        sales["product_id"] = (
            sales["product_id"]
            .astype("string")
            .str.strip()
        )

        inventory["product_id"] = (
            inventory["product_id"]
            .astype("string")
            .str.strip()
        )

    inventory["closing_stock"] = pd.to_numeric(
        inventory["closing_stock"],
        errors="coerce",
    )

    optional_numeric_columns = [
        "lead_time_days",
        "safety_stock",
        "incoming_stock",
        "minimum_order_quantity",
        "pack_size",
    ]

    for column in optional_numeric_columns:
        if column in inventory.columns:
            inventory[column] = pd.to_numeric(
                inventory[column],
                errors="coerce",
            )

    # Reject unusable inventory rows rather than treating
    # missing stock values as zero.
    inventory = inventory.dropna(
        subset=["product_name", "closing_stock"]
    )

    # Duplicate product records must not be selected arbitrarily.
    matching_key = (
        "product_id"
        if use_product_id
        else "product_name"
    )

    if inventory[matching_key].duplicated().any():
        raise ValueError(
            f"Inventory contains duplicate {matching_key} values. "
            "Resolve duplicate inventory records before forecasting."
        )

    products = sorted(
        sales["product_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    successful_results = []
    skipped_products = []

    for product_name in products:
        try:
            product_sales = sales[
                sales["product_name"] == product_name
            ]

            daily_demand = prepare_daily_demand(
                product_sales,
                product_name,
            )

            summary = get_forecasting_data_summary(
                daily_demand
            )

            readiness = check_forecasting_readiness(
                summary,
                minimum_days=min_history_days,
            )

            if not readiness["ready"]:
                skipped_products.append(
                    {
                        "product_name": product_name,
                        "reason": readiness["reason"],
                    }
                )
                continue

            forecast_result = forecast_demand(
                daily_demand["demand"],
                horizon=horizon,
            )

            # Match by ID when both datasets have IDs.
            if use_product_id:
                product_ids = (
                    product_sales["product_id"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                )

                if len(product_ids) != 1:
                    skipped_products.append(
                        {
                            "product_name": product_name,
                            "reason": (
                                "Sales data does not contain exactly "
                                "one product ID for this product name."
                            ),
                        }
                    )
                    continue

                product_inventory = inventory[
                    inventory["product_id"].astype(str).str.strip()
                    == product_ids[0]
                ]
            else:
                product_inventory = inventory[
                    inventory["product_name"].astype(str)
                    == product_name
                ]

            if product_inventory.empty:
                skipped_products.append(
                    {
                        "product_name": product_name,
                        "reason": "No inventory record found.",
                    }
                )
                continue

            inventory_row = product_inventory.iloc[0]

            current_stock = float(
                inventory_row["closing_stock"]
            )

            def get_optional_number(
                column: str,
                default: float,
            ) -> float:
                if column not in inventory_row.index:
                    return default

                value = inventory_row[column]

                if pd.isna(value):
                    return default

                return float(value)

            product_lead_time = int(
                get_optional_number(
                    "lead_time_days",
                    lead_time_days,
                )
            )

            product_safety_stock = get_optional_number(
                "safety_stock",
                safety_stock,
            )

            incoming_stock = get_optional_number(
                "incoming_stock",
                0,
            )

            minimum_order_quantity = int(
                get_optional_number(
                    "minimum_order_quantity",
                    0,
                )
            )

            pack_size = int(
                get_optional_number(
                    "pack_size",
                    1,
                )
            )

            recommendation = (
                create_forecast_inventory_recommendation(
                    current_stock=current_stock,
                    forecast=forecast_result["forecast"],
                    average_daily_demand=summary[
                        "average_daily_demand"
                    ],
                    lead_time_days=product_lead_time,
                    safety_stock=product_safety_stock,
                    incoming_stock=incoming_stock,
                    minimum_order_quantity=minimum_order_quantity,
                    pack_size=pack_size,
                )
            )

            explanation = build_forecast_explanation(
                product_name=product_name,
                forecast_demand=recommendation[
                    "total_forecast_demand"
                ],
                horizon=horizon,
                best_model=forecast_result["best_model"],
                model_performance=forecast_result[
                    "model_performance"
                ],
                diagnostics=forecast_result["diagnostics"],
                reliability=forecast_result["reliability"],
            )

            reorder_explanation = build_reorder_explanation(
                product_name=product_name,
                recommendation=recommendation,
            )

            result_row = {
                "product_name": product_name,
                "forecast_horizon_days": horizon,
                "forecast_demand": round(
                    recommendation["total_forecast_demand"],
                    2,
                ),
                "reliability_rating": explanation[
                    "reliability_rating"
                ],
                "accuracy_statement": explanation[
                    "accuracy_statement"
                ],
                "stability_statement": explanation[
                    "stability_statement"
                ],
                "trend_statement": explanation[
                    "trend_statement"
                ],
                "trend": explanation["trend"],
                "trend_evidence": explanation[
                    "trend_evidence"
                ],
                "trend_confidence": explanation[
                    "trend_confidence"
                ],
                "pattern_statement": explanation[
                    "pattern_statement"
                ],
                "caution_statement": explanation[
                    "caution_statement"
                ],
                "current_stock": round(
                    recommendation["current_stock"],
                    2,
                ),
                "incoming_stock": round(
                    recommendation["incoming_stock"],
                    2,
                ),
                "stock_position": round(
                    recommendation["stock_position"],
                    2,
                ),
                "lead_time_days": product_lead_time,
                "safety_stock": product_safety_stock,
                "reorder_point": round(
                    recommendation["reorder_point"],
                    2,
                ),
                "minimum_order_quantity": minimum_order_quantity,
                "pack_size": pack_size,
                "recommended_order_quantity": (
                    recommendation[
                        "recommended_order_quantity"
                    ]
                ),
                "status": recommendation["status"],
                "stock_risk": reorder_explanation["stock_risk"],
                "reorder_reason": reorder_explanation["reorder_reason"],
                "recommended_action": reorder_explanation[
                    "recommended_action"
                ],
                "best_model": forecast_result["best_model"],
                "wape": round(
                    forecast_result["model_performance"]["wape"],
                    2,
                ),
                "reliability_score": round(
                    forecast_result["reliability"]["score"],
                    2,
                ),
            }

            if "product_id" in inventory_row.index:
                result_row["product_id"] = inventory_row[
                    "product_id"
                ]

            if "supplier_id" in inventory_row.index:
                result_row["supplier_id"] = inventory_row[
                    "supplier_id"
                ]

            if "supplier_name" in inventory_row.index:
                result_row["supplier_name"] = inventory_row[
                    "supplier_name"
                ]

            successful_results.append(result_row)

        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            skipped_products.append(
                {
                    "product_name": product_name,
                    "reason": str(exc),
                }
            )

    return (
        pd.DataFrame(successful_results),
        pd.DataFrame(skipped_products),
    )