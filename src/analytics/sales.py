import pandas as pd


def prepare_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare standardized sales data for analysis.
    """

    df = df.copy()

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # NUMERIC FIELDS
    # -----------------------------------------------------

    numeric_columns = [
        "quantity",
        "unit_price",
        "cost_price",
        "discount",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # -----------------------------------------------------
    # OPTIONAL COLUMNS
    # -----------------------------------------------------

    if "cost_price" not in df.columns:

        df["cost_price"] = 0.0

    if "discount" not in df.columns:

        df["discount"] = 0.0

    # -----------------------------------------------------
    # FILL SAFE VALUES
    # -----------------------------------------------------

    df["discount"] = df["discount"].fillna(0)

    # -----------------------------------------------------
    # CALCULATE SALES
    # -----------------------------------------------------

    df["gross_sales"] = (
        df["quantity"] *
        df["unit_price"]
    )

    # -----------------------------------------------------
    # DISCOUNT
    # -----------------------------------------------------

    df["discount_amount"] = (
        df["gross_sales"] *
        df["discount"] /
        100
    )

    # -----------------------------------------------------
    # REVENUE
    # -----------------------------------------------------

    df["revenue"] = (
        df["gross_sales"] -
        df["discount_amount"]
    )

    # -----------------------------------------------------
    # COST
    # -----------------------------------------------------

    df["cost"] = (
        df["quantity"] *
        df["cost_price"]
    )

    # -----------------------------------------------------
    # PROFIT
    # -----------------------------------------------------

    df["profit"] = (
        df["revenue"] -
        df["cost"]
    )

    # -----------------------------------------------------
    # PROFIT MARGIN
    # -----------------------------------------------------

    df["profit_margin"] = (
        df["profit"] /
        df["revenue"] *
        100
    ).fillna(0)

    return df


def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate core business KPIs.
    """

    revenue = df["revenue"].sum()
    cost = df["cost"].sum()
    profit = df["profit"].sum()

    transactions = (
        df["transaction_id"].nunique()
        if "transaction_id" in df.columns
        else len(df)
    )

    average_order_value = (
        revenue / transactions
        if transactions > 0
        else 0
    )

    profit_margin = (
        profit / revenue * 100
        if revenue > 0
        else 0
    )

    return {
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "transactions": transactions,
        "average_order_value": average_order_value,
        "profit_margin": profit_margin,
    }


def top_products(
    df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:

    return (
        df.groupby("product_name")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
        .head(n)
        .reset_index()
    )


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:

    return (
        df.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
        .reset_index()
    )