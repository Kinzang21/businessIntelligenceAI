import pandas as pd

from src.forecasting.explanation import (
    build_forecast_explanation,
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
    Generate product-level demand forecasts and inventory recommendations.

    Returns:
        successful_results:
            Products for which forecasting and inventory recommendation
            were successfully completed.

        skipped_products:
            Products that could not be forecasted, together with the reason.
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

    missing_sales = required_sales_columns - set(sales_df.columns)
    if missing_sales:
        raise ValueError(
            "Missing required sales columns: "
            + ", ".join(sorted(missing_sales))
        )

    missing_inventory = required_inventory_columns - set(inventory_df.columns)
    if missing_inventory:
        raise ValueError(
            "Missing required inventory columns: "
            + ", ".join(sorted(missing_inventory))
        )

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")

    if safety_stock < 0:
        raise ValueError("Safety stock cannot be negative.")

    products = sorted(
        sales_df["product_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    successful_results = []
    skipped_products = []

    for product_name in products:
        try:
            daily_demand = prepare_daily_demand(
                sales_df,
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

            product_inventory = inventory_df[
                inventory_df["product_name"].astype(str)
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

            current_stock = float(
                product_inventory["closing_stock"].iloc[-1]
            )

            recommendation = create_forecast_inventory_recommendation(
                current_stock=current_stock,
                forecast=forecast_result["forecast"],
                average_daily_demand=summary[
                    "average_daily_demand"
                ],
                lead_time_days=lead_time_days,
                safety_stock=safety_stock,
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

            successful_results.append(
                {
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
                    "trend": explanation[
                        "trend"
                    ],
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
                        current_stock,
                        2,
                    ),
                    "reorder_point": round(
                        recommendation["reorder_point"],
                        2,
                    ),
                    "recommended_order_quantity": (
                        recommendation[
                            "recommended_order_quantity"
                        ]
                    ),
                    "status": recommendation["status"],
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
            )

        except (ValueError, KeyError, TypeError) as exc:
            skipped_products.append(
                {
                    "product_name": product_name,
                    "reason": str(exc),
                }
            )

    successful_results_df = pd.DataFrame(
        successful_results
    )

    skipped_products_df = pd.DataFrame(
        skipped_products
    )

    return successful_results_df, skipped_products_df