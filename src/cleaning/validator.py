import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "product_name",
    "quantity",
    "unit_price",
}


def validate_columns(df: pd.DataFrame) -> dict:
    """
    Check whether the uploaded dataset contains
    the minimum required columns.
    """

    columns = {column.lower().strip() for column in df.columns}

    missing_columns = REQUIRED_COLUMNS - columns

    return {
        "valid": len(missing_columns) == 0,
        "missing_columns": sorted(missing_columns),
        "available_columns": sorted(columns),
    }


def validate_data(df: pd.DataFrame) -> dict:
    """
    Perform basic data-quality checks.
    """

    results = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }

    return results