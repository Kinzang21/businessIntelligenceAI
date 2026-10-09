from src.ingestion.loader import load_file
from src.analytics.sales import (
    prepare_sales_data,
    calculate_kpis,
    top_products,
)


def test_sales_pipeline():

    df = load_file(
        "data/sample/retail_sales.csv"
    )

    df = prepare_sales_data(df)

    kpis = calculate_kpis(df)

    assert kpis["revenue"] > 0
    assert kpis["profit"] > 0
    assert kpis["transactions"] > 0

    products = top_products(df)

    assert len(products) > 0