def calculate_reliability_score(
    diagnostics: dict,
    model_performance: dict,
) -> dict:
    """
    Calculate a transparent forecast reliability score.

    This is not a statistical confidence interval.
    It summarizes the amount and quality of evidence
    supporting a forecast.
    """

    score = 100.0
    reasons = []
    warnings = []

    days = diagnostics.get("days", 0)
    cv = diagnostics.get("coefficient_of_variation")
    seasonality = diagnostics.get(
        "strongest_seasonality",
        0.0,
    )

    wape = model_performance.get("wape", float("inf"))
    wape_std = model_performance.get(
        "wape_std",
        None,
    )
    windows = model_performance.get("windows", 0)

    # ---------------------------------------------------------
    # 1. Historical data
    # ---------------------------------------------------------

    if days < 30:
        score -= 40
        warnings.append(
            "Very limited historical data."
        )
    elif days < 90:
        score -= 20
        warnings.append(
            "Limited historical data."
        )
    elif days < 180:
        score -= 10
        reasons.append(
            "Moderate amount of historical data."
        )
    else:
        reasons.append(
            "Strong amount of historical data."
        )

    # ---------------------------------------------------------
    # 2. Backtesting evidence
    # ---------------------------------------------------------

    if windows < 3:
        score -= 25
        warnings.append(
            "Very few backtesting windows."
        )
    elif windows < 6:
        score -= 10
        warnings.append(
            "Limited backtesting evidence."
        )
    else:
        reasons.append(
            "Forecast evaluated across multiple historical windows."
        )

    # ---------------------------------------------------------
    # 3. Forecast error
    # ---------------------------------------------------------

    if wape == float("inf"):
        score -= 40
        warnings.append(
            "WAPE could not be calculated reliably."
        )
    elif wape > 50:
        score -= 35
        warnings.append(
            "Very high historical forecast error."
        )
    elif wape > 30:
        score -= 25
        warnings.append(
            "High historical forecast error."
        )
    elif wape > 20:
        score -= 15
        warnings.append(
            "Moderate historical forecast error."
        )
    elif wape <= 10:
        reasons.append(
            "Low historical forecast error."
        )
    else:
        reasons.append(
            "Acceptable historical forecast error."
        )

    # ---------------------------------------------------------
    # 4. Backtest stability
    # ---------------------------------------------------------

    if wape_std is not None:

        if wape_std > 15:
            score -= 15
            warnings.append(
                "Forecast error varies substantially "
                "across historical periods."
            )

        elif wape_std > 10:
            score -= 8
            warnings.append(
                "Forecast error shows noticeable "
                "variation across historical periods."
            )

        elif wape_std > 5:
            score -= 3
            reasons.append(
                "Forecast performance is reasonably "
                "stable across historical periods."
            )

        else:
            reasons.append(
                "Forecast performance is highly stable "
                "across historical periods."
            )

    # ---------------------------------------------------------
    # 5. Demand variability
    # ---------------------------------------------------------

    if cv is not None:
        if cv > 1.5:
            score -= 15
            warnings.append(
                "Demand is highly variable."
            )
        elif cv > 0.75:
            score -= 8
            warnings.append(
                "Demand has substantial variability."
            )
        elif cv < 0.30:
            reasons.append(
                "Demand is relatively stable."
            )

    # ---------------------------------------------------------
    # 6. Intermittent demand
    # ---------------------------------------------------------

    if diagnostics.get("intermittent", False):
        score -= 10
        warnings.append(
            "Demand is intermittent with many zero-demand days."
        )

    # ---------------------------------------------------------
    # 7. Recurring pattern
    # ---------------------------------------------------------

    if seasonality >= 0.75:
        score += 5
        reasons.append(
            "Strong recurring demand pattern detected."
        )
    elif seasonality >= 0.60:
        reasons.append(
            "Recurring demand pattern detected."
        )

    # ---------------------------------------------------------
    # Final score
    # ---------------------------------------------------------

    score = max(0.0, min(100.0, score))

    if score >= 80:
        rating = "high"
    elif score >= 60:
        rating = "medium"
    elif score >= 40:
        rating = "low"
    else:
        rating = "very_low"

    return {
        "score": round(score, 1),
        "rating": rating,
        "reasons": reasons,
        "warnings": warnings,
    }