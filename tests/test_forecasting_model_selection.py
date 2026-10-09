import pandas as pd
import pytest

from src.forecasting.baselines import (
    naive_forecast,
    mean_forecast,
)

from src.forecasting.model_selection import (
    evaluate_models,
    select_best_model,
)


def test_evaluate_models():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70] * 10
    )

    models = {
        "naive": naive_forecast,
        "mean": mean_forecast,
    }

    comparison = evaluate_models(
        history=history,
        models=models,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    assert len(comparison) == 2

    assert "model" in comparison.columns
    assert "mae" in comparison.columns
    assert "rmse" in comparison.columns
    assert "wape" in comparison.columns
    assert "rank" in comparison.columns

    assert comparison.iloc[0]["rank"] == 1
    assert comparison.iloc[1]["rank"] == 2


def test_select_best_model():
    comparison = pd.DataFrame(
        {
            "model": [
                "naive",
                "moving_average",
                "seasonal_naive",
            ],
            "wape": [
                30.0,
                15.0,
                20.0,
            ],
            "mae": [
                10.0,
                5.0,
                7.0,
            ],
            "rmse": [
                12.0,
                6.0,
                8.0,
            ],
        }
    )

    best_model = select_best_model(comparison)

    assert best_model == "moving_average"


def test_select_best_model_uses_mae_as_tiebreaker():
    comparison = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
            ],
            "wape": [
                10.0,
                10.0,
            ],
            "mae": [
                8.0,
                5.0,
            ],
            "rmse": [
                10.0,
                7.0,
            ],
        }
    )

    best_model = select_best_model(comparison)

    assert best_model == "model_b"


def test_empty_models():
    history = pd.Series(
        [10, 20, 30, 40, 50]
    )

    with pytest.raises(ValueError):
        evaluate_models(
            history=history,
            models={},
        )