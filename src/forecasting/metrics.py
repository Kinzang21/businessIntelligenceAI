import numpy as np


def mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have the same length.")

    if len(actual) == 0:
        raise ValueError("Actual and predicted values cannot be empty.")

    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have the same length.")

    if len(actual) == 0:
        raise ValueError("Actual and predicted values cannot be empty.")

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def weighted_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate WAPE.

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    Returns a percentage.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have the same length.")

    if len(actual) == 0:
        raise ValueError("Actual and predicted values cannot be empty.")

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return 0.0 if np.sum(np.abs(predicted)) == 0 else float("inf")

    return float(
        np.sum(np.abs(actual - predicted)) / denominator * 100
    )


def calculate_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate all core forecasting metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "wape": weighted_absolute_percentage_error(actual, predicted),
    }