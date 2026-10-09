def build_forecast_explanation(
    product_name: str,
    forecast_demand: float,
    horizon: int,
    best_model: str,
    model_performance: dict,
    diagnostics: dict,
    reliability: dict,
) -> dict:
    """
    Build deterministic, business-readable forecast explanation facts.

    This function does not use an LLM. Every explanation is derived
    from calculated forecasting and diagnostic results.
    """

    wape = float(model_performance.get("wape", float("inf")))
    wape_std = float(model_performance.get("wape_std", 0.0))
    wape_min = float(model_performance.get("wape_min", 0.0))
    wape_max = float(model_performance.get("wape_max", 0.0))
    windows = int(model_performance.get("windows", 0))

    reliability_score = float(reliability.get("score", 0.0))
    reliability_rating = str(
        reliability.get("rating", "unknown")
    )

    trend = diagnostics.get("trend", "unknown")
    classification = diagnostics.get(
        "classification",
        "unknown",
    )

    strongest_period = diagnostics.get(
        "strongest_period"
    )

    strongest_seasonality = float(
        diagnostics.get(
            "strongest_seasonality",
            0.0,
        )
    )

    if wape == float("inf"):
        accuracy_statement = (
            "Historical forecast error could not be "
            "calculated reliably."
        )
    elif wape <= 10:
        accuracy_statement = (
            "Historical forecast error is low."
        )
    elif wape <= 20:
        accuracy_statement = (
            "Historical forecast error is relatively low."
        )
    elif wape <= 30:
        accuracy_statement = (
            "Historical forecast error is moderate."
        )
    elif wape <= 50:
        accuracy_statement = (
            "Historical forecast error is high."
        )
    else:
        accuracy_statement = (
            "Historical forecast error is very high."
        )

    if wape_std <= 5:
        stability_statement = (
            "Forecast performance has been highly "
            "stable across historical periods."
        )
    elif wape_std <= 10:
        stability_statement = (
            "Forecast performance has been reasonably "
            "stable across historical periods."
        )
    elif wape_std <= 15:
        stability_statement = (
            "Forecast performance varies noticeably "
            "across historical periods."
        )
    else:
        stability_statement = (
            "Forecast performance varies substantially "
            "across historical periods."
        )

    trend_evidence = diagnostics.get("trend_evidence", "unknown")
    trend_confidence = diagnostics.get("trend_confidence", "unknown")
    trend_interpretation = diagnostics.get("trend_interpretation", "")

    if trend == "increasing":
        if trend_evidence == "strong":
            trend_statement = (
                "Historical demand shows a strong increasing trend "
                "based on multiple years of observations."
            )
        elif trend_evidence == "moderate":
            trend_statement = (
                "Demand appears to be increasing over the observed period, "
                "but the available history provides only moderate evidence "
                "for a persistent long-term trend."
            )
        elif trend_evidence in {"limited", "very_limited"}:
            trend_statement = (
                "Demand appears to be increasing, but there is limited "
                "historical evidence to establish a reliable long-term trend."
            )
        else:
            trend_statement = "A clear long-term increasing trend was not established."

    elif trend == "decreasing":
        if trend_evidence == "strong":
            trend_statement = (
                "Historical demand shows a strong decreasing trend "
                "based on multiple years of observations."
            )
        elif trend_evidence == "moderate":
            trend_statement = (
                "Demand appears to be decreasing over the observed period, "
                "but the available history provides only moderate evidence "
                "for a persistent long-term trend."
            )
        elif trend_evidence in {"limited", "very_limited"}:
            trend_statement = (
                "Demand appears to be decreasing, but there is limited "
                "historical evidence to establish a reliable long-term trend."
            )
        else:
            trend_statement = "A clear long-term decreasing trend was not established."

    elif trend == "stable":
        trend_statement = "Historical demand is relatively stable."

    else:
        trend_statement = "A clear demand trend was not established."

    if strongest_period is not None:
        pattern_statement = (
            f"A recurring {strongest_period} demand "
            "pattern was detected."
        )
    else:
        pattern_statement = (
            "No strong recurring seasonal pattern was detected."
        )

    if reliability_rating == "high":
        caution_statement = (
            "The forecast has relatively strong supporting evidence."
        )
    elif reliability_rating == "medium":
        caution_statement = (
            "The forecast should be treated as an estimate "
            "rather than a precise demand prediction."
        )
    elif reliability_rating == "low":
        caution_statement = (
            "Use caution when making inventory decisions "
            "based on this forecast."
        )
    else:
        caution_statement = (
            "The forecast has limited supporting evidence "
            "and should not be treated as precise."
        )

    return {
        "product_name": product_name,
        "forecast_horizon_days": int(horizon),
        "forecast_demand": round(float(forecast_demand), 2),
        "average_daily_forecast": round(
            float(forecast_demand) / horizon,
            2,
        ),
        "best_model": best_model,
        "wape": round(wape, 2),
        "wape_std": round(wape_std, 2),
        "wape_min": round(wape_min, 2),
        "wape_max": round(wape_max, 2),
        "backtest_windows": windows,
        "reliability_score": reliability_score,
        "reliability_rating": reliability_rating,
        "demand_classification": classification,
        "trend": trend,
        "trend_evidence": trend_evidence,
        "trend_confidence": trend_confidence,
        "trend_interpretation": trend_interpretation,
        "strongest_pattern": strongest_period,
        "seasonality_strength": round(
            strongest_seasonality,
            3,
        ),
        "accuracy_statement": accuracy_statement,
        "stability_statement": stability_statement,
        "trend_statement": trend_statement,
        "pattern_statement": pattern_statement,
        "caution_statement": caution_statement,
    }

def build_reorder_explanation(
    product_name: str,
    recommendation: dict,
) -> dict:
    """
    Generate deterministic, business-friendly explanations
    for inventory replenishment recommendations.

    All explanations are derived from calculated inventory
    values and the recommended order quantity.
    """

    current_stock = float(
        recommendation["current_stock"]
    )

    incoming_stock = float(
        recommendation["incoming_stock"]
    )

    stock_position = float(
        recommendation["stock_position"]
    )

    forecast_demand = float(
        recommendation["total_forecast_demand"]
    )

    safety_stock = float(
        recommendation["safety_stock"]
    )

    reorder_point = float(
        recommendation["reorder_point"]
    )

    recommended_quantity = int(
        recommendation["recommended_order_quantity"]
    )

    minimum_order_quantity = int(
        recommendation["minimum_order_quantity"]
    )

    pack_size = int(
        recommendation["pack_size"]
    )

    status = str(
        recommendation["status"]
    )

    # Explain the current inventory situation.
    if current_stock <= 0 and incoming_stock <= 0:
        stock_risk = (
            "No stock is currently available, and no incoming "
            "stock has been recorded."
        )

    elif current_stock <= 0 and incoming_stock > 0:
        stock_risk = (
            "Current stock is zero, but incoming stock has "
            "been recorded. Confirm its expected arrival date."
        )

    elif stock_position <= reorder_point:
        stock_risk = (
            "The available stock position is at or below "
            "the calculated reorder point."
        )

    else:
        stock_risk = (
            "The available stock position is above the "
            "calculated reorder point."
        )

    # Explain the replenishment decision.
    if recommended_quantity > 0:
        reorder_reason = (
            f"Forecast demand over the next "
            f"{recommendation['forecast_horizon_days']} days is "
            f"{forecast_demand:.2f} units. After accounting for "
            f"current stock, incoming stock, and safety stock, "
            f"the recommended order quantity is "
            f"{recommended_quantity} units."
        )

        if (
            minimum_order_quantity > 0
            and recommended_quantity >= minimum_order_quantity
        ):
            reorder_reason += (
                f" The supplier minimum order quantity is "
                f"{minimum_order_quantity} units."
            )

        if pack_size > 1:
            reorder_reason += (
                f" The recommended quantity follows a pack "
                f"size of {pack_size} units."
            )

        if current_stock <= 0 and incoming_stock <= 0:
            recommended_action = (
                "Prioritize replenishment and confirm supplier "
                "availability before placing an order."
            )

        elif current_stock <= 0 and incoming_stock > 0:
            recommended_action = (
                "Check when the incoming stock will arrive. "
                "Consider additional replenishment if it will "
                "not arrive in time to meet expected demand."
            )

        else:
            recommended_action = (
                "Review the recommendation and confirm supplier "
                "availability before placing the order."
            )

    else:
        reorder_reason = (
            "No additional order is recommended because current "
            "stock and incoming stock are sufficient to cover "
            "the forecast demand and safety stock under the "
            "current calculation."
        )

        recommended_action = (
            "No additional order is recommended by the current "
            "forecast and inventory calculation. Reassess if "
            "demand or delivery conditions change."
        )

    return {
        "product_name": product_name,
        "inventory_status": status,
        "stock_risk": stock_risk,
        "reorder_reason": reorder_reason,
        "recommended_action": recommended_action,
        "recommended_order_quantity": recommended_quantity,
        "forecast_demand": round(forecast_demand, 2),
        "stock_position": round(stock_position, 2),
        "reorder_point": round(reorder_point, 2),
        "safety_stock": round(safety_stock, 2),
    }
