from pathlib import Path

import numpy as np
from numpy.ma import product
import pandas as pd


SEED = 42
rng = np.random.default_rng(SEED)

START_DATE = "2026-01-01"
END_DATE = "2026-12-31"

PRODUCTS = [
    {
        "product_id": "P001",
        "product_name": "Coca Cola 500ml",
        "category": "Beverage",
        "unit_price": 60,
        "cost_price": 45,
        "base_demand": 12,
        "weekly_strength": 0.25,
        "growth": 0.10,
    },
    {
        "product_id": "P002",
        "product_name": "Wai Wai Noodles",
        "category": "Grocery",
        "unit_price": 35,
        "cost_price": 25,
        "base_demand": 18,
        "weekly_strength": 0.12,
        "growth": 0.05,
    },
    {
        "product_id": "P003",
        "product_name": "Milk 1L",
        "category": "Dairy",
        "unit_price": 85,
        "cost_price": 70,
        "base_demand": 9,
        "weekly_strength": 0.20,
        "growth": 0.03,
    },
    {
        "product_id": "P004",
        "product_name": "Basmati Rice 5kg",
        "category": "Grocery",
        "unit_price": 420,
        "cost_price": 360,
        "base_demand": 4,
        "weekly_strength": 0.10,
        "growth": 0.02,
    },
    {
        "product_id": "P005",
        "product_name": "Potato Chips 100g",
        "category": "Snacks",
        "unit_price": 50,
        "cost_price": 35,
        "base_demand": 7,
        "weekly_strength": 0.30,
        "growth": 0.08,
    },
    {
        "product_id": "P006",
        "product_name": "Bread",
        "category": "Bakery",
        "unit_price": 55,
        "cost_price": 40,
        "base_demand": 10,
        "weekly_strength": 0.35,
        "growth": 0.04,
    },
    {
        "product_id": "P007",
        "product_name": "Pepsi 500ml",
        "category": "Beverage",
        "unit_price": 60,
        "cost_price": 45,
        "base_demand": 9,
        "weekly_strength": 0.28,
        "growth": 0.09,
    },
]


def generate_product_sales(product, dates):
    rows = []

    for day_index, date in enumerate(dates):
        # Gradual growth across the year.
        growth_factor = 1 + product["growth"] * (
            day_index / (len(dates) - 1)
        )

        # Weekly purchasing pattern.
        weekday = date.weekday()

        weekly_multiplier = 1.0

        if weekday in (4, 5):
            weekly_multiplier += product["weekly_strength"]

        if weekday == 6:
            weekly_multiplier += product["weekly_strength"] * 0.5

        # Mild recurring yearly pattern.
        # This is intentionally not labelled as a Bhutanese event.
        annual_multiplier = 1.0 + 0.12 * np.sin(
            2 * np.pi * day_index / 365
        )

        expected = (
            product["base_demand"]
            * growth_factor
            * weekly_multiplier
            * annual_multiplier
        )

        # Occasional random demand spikes.
        spike = 1.0

        if rng.random() < 0.025:
            spike = rng.uniform(1.5, 2.8)

        expected *= spike

        # Some products occasionally have zero-demand days.
        quantity = None

        if product["product_name"] == "Basmati Rice 5kg":
            if rng.random() < 0.08:
                quantity = 0

        elif product["product_name"] == "Potato Chips 100g":
            if rng.random() < 0.04:
                quantity = 0

        if quantity is None:
            quantity = max(0, int(rng.poisson(expected)))
        # Small discount probability.
        discount = 0

        if rng.random() < 0.08:
            discount = int(product["unit_price"] * rng.uniform(0.03, 0.10))

        if quantity > 0:
            rows.append(
                {
                    "date": date,
                    "transaction_id": f"T{len(rows) + 1:05d}",
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "quantity": quantity,
                    "unit_price": product["unit_price"],
                    "cost_price": product["cost_price"],
                    "discount": discount,
                }
            )

    return rows


def main():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []

    for product in PRODUCTS:
        rows.extend(generate_product_sales(product, dates))

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "product_id"]).reset_index(drop=True)

    output_path = Path("data/sample/retail_sales.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Products: {df['product_name'].nunique()}")
    print()
    print("Rows per product:")
    print(df.groupby("product_name").size())
    print()
    print("Total quantity by product:")
    print(df.groupby("product_name")["quantity"].sum())


if __name__ == "__main__":
    main()