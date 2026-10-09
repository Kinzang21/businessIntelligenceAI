import numpy as np
import pandas as pd


def calculate_demand_diagnostics(
    daily_demand: pd.DataFrame,
) -> dict:
    """
    Analyze a product's daily demand series.

    The diagnostics are deliberately data-driven.
    They do not assume any particular country's calendar
    or business events.
    """

    if daily_demand.empty:
        return {
            "days": 0,
            "total_demand": 0.0,
            "average_demand": 0.0,
            "std_demand": 0.0,
            "coefficient_of_variation": None,
            "zero_demand_days": 0,
            "zero_demand_ratio": 0.0,
            "trend": "unknown",
            "trend_strength": 0.0,
            "trend_evidence": "none",
            "trend_confidence": "very_low",
            "trend_interpretation": (
                "No historical demand data is available."
            ),
            "intermittent": False,
            "seasonal_periods": {},
            "strongest_period": None,
            "strongest_seasonality": 0.0,
        }

    data = daily_demand.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["demand"] = pd.to_numeric(
        data["demand"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["date", "demand"]
    )

    data = data.sort_values("date")

    demand = data["demand"].astype(float)

    days = len(data)
    total_demand = demand.sum()
    average_demand = demand.mean()
    std_demand = demand.std(ddof=0)

    zero_demand_days = int(
        (demand == 0).sum()
    )

    zero_demand_ratio = (
        zero_demand_days / days
        if days > 0
        else 0.0
    )

    # -------------------------
    # Coefficient of variation
    # -------------------------

    if average_demand > 0:
        coefficient_of_variation = (
            std_demand / average_demand
        )
    else:
        coefficient_of_variation = None

    # -------------------------
    # Intermittent demand
    # -------------------------

    intermittent = (
        zero_demand_ratio >= 0.30
    )

    # -------------------------
    # Automatic seasonality
    # -------------------------

    seasonal_periods = {}

    candidate_periods = {
        "weekly": 7,
        "monthly": 30,
        "quarterly": 90,
        "annual": 365,
    }

    for name, period in candidate_periods.items():

        if days >= period * 2:

            original = demand.iloc[:-period].to_numpy(
                dtype=float
            )

            lagged = demand.iloc[period:].to_numpy(
                dtype=float
            )

            if (
                len(original) == 0
                or len(lagged) == 0
                or np.std(original) == 0
                or np.std(lagged) == 0
            ):
                correlation = np.nan

            else:
                correlation = np.corrcoef(
                    original,
                    lagged,
                )[0, 1]

            if pd.notna(correlation):

                seasonal_strength = max(
                    0.0,
                    float(correlation),
                )

                seasonal_periods[name] = {
                    "period": period,
                    "autocorrelation": float(
                        correlation
                    ),
                    "strength": seasonal_strength,
                }

    if seasonal_periods:

        strongest_period = max(
            seasonal_periods,
            key=lambda name:
                seasonal_periods[name]["strength"],
        )

        strongest_seasonality = seasonal_periods[
            strongest_period
        ]["strength"]

    else:

        strongest_period = None
        strongest_seasonality = 0.0

    # -------------------------
    # Trend
    # -------------------------

    if days >= 7 and average_demand > 0:

        x = np.linspace(
            0.0,
            1.0,
            days,
        )

        design_columns = [
            np.ones(days),
            x,
        ]

        seasonal_components = [
            info
            for info in seasonal_periods.values()
            if info["strength"] >= 0.30
        ]

        for seasonal in seasonal_components:

            period = seasonal["period"]

            angle = (
                2
                * np.pi
                * np.arange(days)
                / period
            )

            design_columns.append(
                np.sin(angle)
            )

            design_columns.append(
                np.cos(angle)
            )

        design_matrix = np.column_stack(
            design_columns
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            demand.to_numpy(dtype=float),
            rcond=None,
        )

        fitted_change = float(
            coefficients[1]
        )

        relative_change = (
            fitted_change / average_demand
        )

        if relative_change > 0.10:
            trend = "increasing"
        elif relative_change < -0.10:
            trend = "decreasing"
        else:
            trend = "stable"

        trend_strength = min(
            abs(relative_change),
            1.0,
        )
        if trend_strength < 1e-12:
            trend_strength = 0.0

    else:

        trend = "insufficient_data"
        trend_strength = 0.0
        relative_change = 0.0

    # -------------------------
    # Trend evidence
    # -------------------------
    #
    # The direction of a fitted trend is not the same
    # thing as evidence that the trend will persist.
    #
    # Longer histories provide stronger evidence.
    # Multiple annual cycles are especially important
    # when evaluating long-term demand behavior.
    # -------------------------

    if trend == "insufficient_data":

        trend_evidence = "none"
        trend_confidence = "very_low"

        trend_interpretation = (
            "There is insufficient historical data "
            "to establish a meaningful demand trend."
        )

    elif days < 90:

        trend_evidence = "very_limited"
        trend_confidence = "very_low"

        trend_interpretation = (
            f"Demand appears {trend}, but the historical "
            "period is too short to establish whether "
            "this direction is persistent."
        )

    elif days < 180:

        trend_evidence = "limited"
        trend_confidence = "low"

        trend_interpretation = (
            f"Demand appears {trend}, but the available "
            "history provides limited evidence that this "
            "is a persistent long-term trend."
        )

    elif days < 365:

        trend_evidence = "moderate"
        trend_confidence = "medium"

        trend_interpretation = (
            f"Demand appears {trend}. The historical "
            "period provides moderate evidence for this "
            "direction, but recurring long-term patterns "
            "cannot yet be assessed reliably."
        )

    else:

        # One year is enough to describe the observed
        # direction, but not enough to confirm recurring
        # annual behavior.
        if days < 730:

            trend_evidence = "moderate"
            trend_confidence = "medium"

            if trend == "stable":

                trend_interpretation = (
                    "Demand appears relatively stable "
                    "over the observed period. However, "
                    "recurring annual behavior cannot "
                    "yet be confirmed."
                )

            else:

                trend_interpretation = (
                    f"Demand appears {trend} over the "
                    "observed period. However, only one "
                    "annual cycle is available, so it is "
                    "not yet possible to establish whether "
                    "this is a persistent long-term trend "
                    "or part of a recurring cycle."
                )

        else:

            trend_evidence = "strong"
            trend_confidence = "high"

            trend_interpretation = (
                f"Demand shows a {trend} trend across "
                "multiple historical cycles, providing "
                "stronger evidence that the observed "
                "direction is persistent."
            )

    return {
        "days": days,
        "total_demand": float(total_demand),
        "average_demand": float(
            average_demand
        ),
        "std_demand": float(
            std_demand
        ),
        "coefficient_of_variation": (
            float(coefficient_of_variation)
            if coefficient_of_variation is not None
            else None
        ),
        "zero_demand_days": zero_demand_days,
        "zero_demand_ratio": float(
            zero_demand_ratio
        ),
        "trend": trend,
        "trend_strength": float(
            trend_strength
        ),
        "trend_evidence": trend_evidence,
        "trend_confidence": trend_confidence,
        "trend_interpretation": trend_interpretation,
        "intermittent": intermittent,
        "seasonal_periods": seasonal_periods,
        "strongest_period": strongest_period,
        "strongest_seasonality": float(
            strongest_seasonality
        ),
    }


def classify_demand_pattern(
    diagnostics: dict,
) -> str:
    """
    Classify the overall demand structure.
    """

    if diagnostics["days"] < 30:
        return "insufficient_history"

    if diagnostics["total_demand"] <= 0:
        return "no_demand"

    if diagnostics["intermittent"]:
        return "intermittent"

    if diagnostics["strongest_period"] is not None:
        if diagnostics["strongest_seasonality"] >= 0.60:
            return "seasonal"

    cv = diagnostics[
        "coefficient_of_variation"
    ]

    if cv is None:
        return "unknown"

    if cv < 0.30:
        return "stable"

    if cv < 0.75:
        return "moderately_variable"

    return "highly_variable"