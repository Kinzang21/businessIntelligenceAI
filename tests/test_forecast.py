import pandas as pd

from src.forecasting.forecast import (
    build_candidate_models,
    forecast_demand,
)


def test_build_candidate_models():
    models = build_candidate_models()

    assert "naive" in models
    assert "mean" in models
    assert "moving_average_7" in models
    assert "weighted_moving_average_7" in models


def test_build_candidate_models_with_seasonality():
    models = build_candidate_models(
        seasonal_period=7
    )

    assert "seasonal_naive" in models


def test_forecast_demand():
    history = pd.Series(
        [10, 12, 11, 13, 15, 40, 45] * 12
    )

    result = forecast_demand(
        history,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    assert "forecast" in result
    assert "best_model" in result
    assert "diagnostics" in result
    assert "model_comparison" in result

    assert len(result["forecast"]) == 7

    assert result["best_model"] in (
        result["model_comparison"]["model"].tolist()
    )


def test_forecast_contains_no_negative_demand():
    history = pd.Series(
        [10, 12, 11, 13, 15, 40, 45] * 12
    )

    result = forecast_demand(
        history,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    assert all(value >= 0 for value in result["forecast"])


def test_forecast_model_comparison_is_ranked():
    history = pd.Series(
        [10, 12, 11, 13, 15, 40, 45] * 12
    )

    result = forecast_demand(
        history,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    comparison = result["model_comparison"]

    assert comparison.iloc[0]["rank"] == 1
    assert comparison["wape"].is_monotonic_increasing

def test_forecast_includes_reliability():
    history = pd.Series(
        [10, 12, 11, 13, 15, 40, 45] * 12
    )

    result = forecast_demand(
        history,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    reliability = result["reliability"]

    assert "score" in reliability
    assert "rating" in reliability
    assert "reasons" in reliability
    assert "warnings" in reliability

    assert 0 <= reliability["score"] <= 100