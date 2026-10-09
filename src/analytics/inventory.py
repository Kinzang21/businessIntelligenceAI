import pandas as pd


def calculate_inventory_metrics(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate inventory metrics by combining sales history
    with the current inventory dataset.
    """

    sales_required = {
        "product_id",
        "product_name",
        "category",
        "quantity",
        "date",
    }

    inventory_required = {
        "product_id",
        "product_name",
        "category",
        "closing_stock",
    }

    missing_sales = sales_required - set(sales_df.columns)

    if missing_sales:
        raise ValueError(
            "Missing sales columns: "
            + ", ".join(sorted(missing_sales))
        )

    missing_inventory = (
        inventory_required - set(inventory_df.columns)
    )

    if missing_inventory:
        raise ValueError(
            "Missing inventory columns: "
            + ", ".join(sorted(missing_inventory))
        )

    sales = sales_df.copy()
    inventory = inventory_df.copy()

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="coerce",
    )

    sales["quantity"] = pd.to_numeric(
        sales["quantity"],
        errors="coerce",
    )

    inventory["closing_stock"] = pd.to_numeric(
        inventory["closing_stock"],
        errors="coerce",
    )

    sales = sales.dropna(
        subset=[
            "product_id",
            "product_name",
            "date",
            "quantity",
        ]
    )

    inventory = inventory.dropna(
        subset=[
            "product_id",
            "product_name",
            "closing_stock",
        ]
    )

    # Calculate sales metrics from the sales dataset.
    sales_metrics = (
        sales.groupby(
            [
                "product_id",
                "product_name",
                "category",
            ],
            dropna=False,
        )
        .agg(
            units_sold=("quantity", "sum"),
        )
        .reset_index()
    )

    # Calculate the number of days represented by sales history.
    if sales.empty:
        number_of_days = 1
    else:
        number_of_days = max(
            (
                sales["date"].max()
                - sales["date"].min()
            ).days
            + 1,
            1,
        )

    sales_metrics["average_daily_sales"] = (
        sales_metrics["units_sold"]
        / number_of_days
    )

    # Keep the latest inventory record for each product.
    inventory_current = (
        inventory.sort_values("product_id")
        .drop_duplicates(
            subset=["product_id"],
            keep="last",
        )
    )

    inventory_current = inventory_current[
        [
            "product_id",
            "product_name",
            "category",
            "closing_stock",
        ]
    ]

    # Combine sales history with current inventory.
    product_inventory = sales_metrics.merge(
        inventory_current,
        on=[
            "product_id",
            "product_name",
            "category",
        ],
        how="left",
    )

    # Estimate how many days current stock can cover.
    product_inventory["days_of_stock"] = (
        product_inventory["closing_stock"]
        / product_inventory["average_daily_sales"]
    )

    product_inventory["days_of_stock"] = (
        product_inventory["days_of_stock"]
        .replace(
            [float("inf"), -float("inf")],
            None,
        )
    )

    return product_inventory


def classify_stock(
    days_of_stock,
    low_stock_days: int = 7,
):
    """
    Classify inventory based on estimated stock coverage.
    """

    if days_of_stock is None:
        return "No Sales Data"

    if pd.isna(days_of_stock):
        return "No Sales Data"

    if days_of_stock <= 0:
        return "Out of Stock"

    if days_of_stock <= low_stock_days:
        return "Low Stock"

    if days_of_stock <= 30:
        return "Healthy"

    return "Overstocked"


def add_stock_status(
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a human-readable stock status.
    """

    inventory_df = inventory_df.copy()

    inventory_df["stock_status"] = (
        inventory_df["days_of_stock"]
        .apply(classify_stock)
    )

    return inventory_df


def inventory_summary(
    inventory_df: pd.DataFrame,
) -> dict:
    """
    Generate high-level inventory KPIs.
    """

    return {
        "products": len(inventory_df),

        "low_stock": int(
            (
                inventory_df["stock_status"]
                == "Low Stock"
            ).sum()
        ),

        "out_of_stock": int(
            (
                inventory_df["stock_status"]
                == "Out of Stock"
            ).sum()
        ),

        "overstocked": int(
            (
                inventory_df["stock_status"]
                == "Overstocked"
            ).sum()
        ),
    }