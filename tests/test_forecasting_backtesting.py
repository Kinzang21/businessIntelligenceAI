import pandas as pd
import pytest

from src.forecasting.backtesting import (
    rolling_backtest,
    summarize_backtest,
)
from src.forecasting.baselines import naive_forecast


def test_rolling_backtest():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70] * 10
    )

    results = rolling_backtest(
        history,
        naive_forecast,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    assert not results.empty
    assert len(results) > 1

    assert "mae" in results.columns
    assert "rmse" in results.columns
    assert "wape" in results.columns


def test_backtest_summary():
    history = pd.Series(
        [10, 20, 30, 40, 50, 60, 70] * 10
    )

    results = rolling_backtest(
        history,
        naive_forecast,
        horizon=7,
        min_train_size=21,
        step=7,
    )

    summary = summarize_backtest(results)

    assert summary["windows"] == len(results)
    assert summary["mae"] >= 0
    assert summary["rmse"] >= 0
    assert summary["wape"] >= 0


def test_backtest_requires_enough_history():
    history = pd.Series([10, 20, 30])

    with pytest.raises(ValueError):
        rolling_backtest(
            history,
            naive_forecast,
            horizon=7,
            min_train_size=30,
        )


def test_backtest_does_not_use_future_data():
    history = pd.Series(
        [10] * 30 + [100] * 7
    )

    results = rolling_backtest(
        history,
        naive_forecast,
        horizon=7,
        min_train_size=30,
        step=7,
    )

    # The first forecast should only see the first 30 values.
    # Therefore naive forecast should predict 10,
    # not the future value of 100.
    assert results.iloc[0]["mae"] == pytest.approx(90)