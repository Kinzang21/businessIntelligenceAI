import pandas as pd

from .backtesting import rolling_backtest, summarize_backtest


def evaluate_models(
    history: pd.Series,
    models: dict,
    horizon: int = 7,
    min_train_size: int = 30,
    step: int = 7,
) -> pd.DataFrame:
    """
    Evaluate multiple forecasting models using rolling backtesting.

    Parameters
    ----------
    history:
        Historical demand series.

    models:
        Dictionary where:
            key = model name
            value = forecasting function

    horizon:
        Number of future observations predicted per window.

    min_train_size:
        Minimum amount of historical data used before
        the first forecast.

    step:
        Number of observations to move forward between windows.

    Returns
    -------
    pd.DataFrame
        Model performance ranked by WAPE.
    """

    if not models:
        raise ValueError("At least one forecasting model is required.")

    results = []

    for model_name, forecast_function in models.items():

        backtest_results = rolling_backtest(
            history=history,
            forecast_function=forecast_function,
            horizon=horizon,
            min_train_size=min_train_size,
            step=step,
        )

        summary = summarize_backtest(backtest_results)

        results.append(
            {
                "model": model_name,
                "windows": summary["windows"],
                "mae": summary["mae"],
                "rmse": summary["rmse"],
                "wape": summary["wape"],
                "wape_std": summary["wape_std"],
                "wape_min": summary["wape_min"],
                "wape_max": summary["wape_max"],
            }
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by=["wape", "mae", "rmse"],
        ascending=True,
    ).reset_index(drop=True)

    comparison["rank"] = range(1, len(comparison) + 1)

    return comparison


def select_best_model(
    comparison: pd.DataFrame,
) -> str:
    """
    Select the best-performing model.

    WAPE is the primary ranking metric,
    followed by MAE and RMSE.
    """

    if comparison.empty:
        raise ValueError("Model comparison cannot be empty.")

    required_columns = {
        "model",
        "wape",
        "mae",
        "rmse",
    }

    missing = required_columns - set(comparison.columns)

    if missing:
        raise ValueError(
            "Missing model comparison columns: "
            + ", ".join(sorted(missing))
        )

    ranked = comparison.sort_values(
        by=["wape", "mae", "rmse"],
        ascending=True,
    )

    return str(ranked.iloc[0]["model"])