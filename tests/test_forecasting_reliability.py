from src.forecasting.reliability import (
    calculate_reliability_score,
)


def test_high_reliability():
    diagnostics = {
        "days": 365,
        "zero_demand_ratio": 0.05,
        "coefficient_of_variation": 0.25,
        "intermittent": False,
        "strongest_seasonality": 0.80,
    }

    performance = {
        "wape": 8.0,
        "windows": 20,
    }

    result = calculate_reliability_score(
        diagnostics,
        performance,
    )

    assert result["score"] >= 80
    assert result["rating"] == "high"


def test_low_reliability():
    diagnostics = {
        "days": 40,
        "zero_demand_ratio": 0.40,
        "coefficient_of_variation": 1.8,
        "intermittent": True,
        "strongest_seasonality": 0.10,
    }

    performance = {
        "wape": 55.0,
        "windows": 2,
    }

    result = calculate_reliability_score(
        diagnostics,
        performance,
    )

    assert result["score"] < 60
    assert result["rating"] in (
        "low",
        "very_low",
    )


def test_score_bounds():
    diagnostics = {
        "days": 365,
        "zero_demand_ratio": 0.0,
        "coefficient_of_variation": 0.1,
        "intermittent": False,
        "strongest_seasonality": 1.0,
    }

    performance = {
        "wape": 5.0,
        "windows": 50,
    }

    result = calculate_reliability_score(
        diagnostics,
        performance,
    )

    assert 0 <= result["score"] <= 100


def test_reliability_contains_explanations():
    diagnostics = {
        "days": 120,
        "zero_demand_ratio": 0.10,
        "coefficient_of_variation": 0.5,
        "intermittent": False,
        "strongest_seasonality": 0.65,
    }

    performance = {
        "wape": 18.0,
        "windows": 10,
    }

    result = calculate_reliability_score(
        diagnostics,
        performance,
    )

    assert "reasons" in result
    assert "warnings" in result
    assert isinstance(result["reasons"], list)
    assert isinstance(result["warnings"], list)