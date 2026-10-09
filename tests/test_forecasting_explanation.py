from src.forecasting.explanation import (
    build_forecast_explanation,
)


def test_forecast_explanation_contains_core_facts():
    diagnostics = {
        "classification": "moderately_variable",
        "trend": "stable",
        "strongest_period": "weekly",
        "strongest_seasonality": 0.72,
    }

    performance = {
        "wape": 21.5,
        "wape_std": 6.0,
        "wape_min": 15.0,
        "wape_max": 35.0,
        "windows": 10,
    }

    reliability = {
        "score": 82.0,
        "rating": "high",
    }

    result = build_forecast_explanation(
        product_name="Coca Cola 500ml",
        forecast_demand=200,
        horizon=14,
        best_model="mean",
        model_performance=performance,
        diagnostics=diagnostics,
        reliability=reliability,
    )

    assert result["product_name"] == "Coca Cola 500ml"
    assert result["forecast_horizon_days"] == 14
    assert result["forecast_demand"] == 200.0
    assert result["average_daily_forecast"] == 14.29
    assert result["best_model"] == "mean"
    assert result["wape"] == 21.5
    assert result["wape_std"] == 6.0
    assert result["reliability_score"] == 82.0
    assert result["reliability_rating"] == "high"


def test_explanation_identifies_high_error():
    diagnostics = {
        "classification": "highly_variable",
        "trend": "stable",
        "strongest_period": None,
        "strongest_seasonality": 0.1,
    }

    performance = {
        "wape": 47.5,
        "wape_std": 16.0,
        "wape_min": 22.0,
        "wape_max": 83.6,
        "windows": 20,
    }

    reliability = {
        "score": 60.0,
        "rating": "medium",
    }

    result = build_forecast_explanation(
        product_name="Basmati Rice 5kg",
        forecast_demand=57,
        horizon=14,
        best_model="mean",
        model_performance=performance,
        diagnostics=diagnostics,
        reliability=reliability,
    )

    assert result["reliability_rating"] == "medium"
    assert "high" in result["accuracy_statement"]
    assert "substantially" in result["stability_statement"]
    assert "caution" not in result["caution_statement"].lower()


def test_explanation_handles_recurring_pattern():
    diagnostics = {
        "classification": "seasonal",
        "trend": "increasing",
        "strongest_period": "weekly",
        "strongest_seasonality": 0.80,
    }

    performance = {
        "wape": 12.0,
        "wape_std": 4.0,
        "wape_min": 8.0,
        "wape_max": 20.0,
        "windows": 15,
    }

    reliability = {
        "score": 90.0,
        "rating": "high",
    }

    result = build_forecast_explanation(
        product_name="Bread",
        forecast_demand=160,
        horizon=14,
        best_model="moving_average_7",
        model_performance=performance,
        reliability=reliability,
        diagnostics=diagnostics,
    )

    assert result["strongest_pattern"] == "weekly"
    assert result["seasonality_strength"] == 0.8
    assert "increasing trend" in result["trend_statement"]
    assert "weekly" in result["pattern_statement"]
from src.forecasting.explanation import build_forecast_explanation


def create_base_inputs():
    model_performance = {
        "wape": 15.0,
        "wape_std": 4.0,
        "wape_min": 10.0,
        "wape_max": 20.0,
        "windows": 10,
    }

    diagnostics = {
        "trend": "decreasing",
        "trend_evidence": "moderate",
        "trend_confidence": "medium",
        "trend_interpretation": (
            "Demand appears decreasing over the observed period."
        ),
        "classification": "seasonal",
        "strongest_period": 7,
        "strongest_seasonality": 0.65,
    }

    reliability = {
        "score": 75.0,
        "rating": "medium",
    }

    return model_performance, diagnostics, reliability


def test_explanation_uses_evidence_aware_moderate_trend():
    model_performance, diagnostics, reliability = create_base_inputs()

    result = build_forecast_explanation(
        product_name="Milk 1L",
        forecast_demand=100,
        horizon=7,
        best_model="seasonal_naive",
        model_performance=model_performance,
        diagnostics=diagnostics,
        reliability=reliability,
    )

    assert result["trend_evidence"] == "moderate"
    assert result["trend_confidence"] == "medium"
    assert "moderate evidence" in result["trend_statement"]


def test_explanation_uses_strong_trend_evidence():
    model_performance, diagnostics, reliability = create_base_inputs()

    diagnostics["trend_evidence"] = "strong"
    diagnostics["trend_confidence"] = "high"

    result = build_forecast_explanation(
        product_name="Milk 1L",
        forecast_demand=100,
        horizon=7,
        best_model="seasonal_naive",
        model_performance=model_performance,
        diagnostics=diagnostics,
        reliability=reliability,
    )

    assert result["trend_evidence"] == "strong"
    assert result["trend_confidence"] == "high"
    assert "strong decreasing trend" in result["trend_statement"]


def test_explanation_uses_limited_trend_evidence():
    model_performance, diagnostics, reliability = create_base_inputs()

    diagnostics["trend_evidence"] = "very_limited"
    diagnostics["trend_confidence"] = "very_low"

    result = build_forecast_explanation(
        product_name="Milk 1L",
        forecast_demand=100,
        horizon=7,
        best_model="naive",
        model_performance=model_performance,
        diagnostics=diagnostics,
        reliability=reliability,
    )

    assert result["trend_evidence"] == "very_limited"
    assert result["trend_confidence"] == "very_low"
    assert "limited historical evidence" in result["trend_statement"]