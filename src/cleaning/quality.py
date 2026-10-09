import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "product_name",
    "quantity",
    "unit_price",
}


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """
    Return required columns that are missing.
    """

    return sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )


def generate_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a business data-quality report.
    """

    report = {}

    report["total_rows"] = len(df)

    report["total_columns"] = len(df.columns)

    report["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    report["missing_values"] = int(
        df.isna().sum().sum()
    )

    report["missing_by_column"] = (
        df.isna()
        .sum()
        .loc[lambda x: x > 0]
        .to_dict()
    )

    # Invalid dates
    if "date" in df.columns:

        converted_dates = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        report["invalid_dates"] = int(
            converted_dates.isna().sum()
        )

    else:

        report["invalid_dates"] = 0

    # Invalid quantities
    if "quantity" in df.columns:

        converted_quantity = pd.to_numeric(
            df["quantity"],
            errors="coerce",
        )

        report["invalid_quantities"] = int(
            converted_quantity.isna().sum()
        )

    else:

        report["invalid_quantities"] = 0

    # Invalid prices
    if "unit_price" in df.columns:

        converted_prices = pd.to_numeric(
            df["unit_price"],
            errors="coerce",
        )

        report["invalid_prices"] = int(
            converted_prices.isna().sum()
        )

    else:

        report["invalid_prices"] = 0

    return report