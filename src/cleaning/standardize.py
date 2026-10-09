import re

import pandas as pd


# ---------------------------------------------------------
# CANONICAL COLUMN ALIASES
# ---------------------------------------------------------

COLUMN_ALIASES = {
    "date": [
        "date",
        "sale date",
        "sales date",
        "transaction date",
        "invoice date",
        "order date",
        "sold date",
    ],

    "transaction_id": [
        "transaction id",
        "transaction",
        "transaction number",
        "invoice",
        "invoice id",
        "invoice number",
        "bill number",
        "receipt number",
        "order id",
        "order number",
    ],

    "product_id": [
        "product id",
        "product code",
        "item code",
        "sku",
        "sku code",
        "item id",
    ],

    "product_name": [
        "product",
        "product name",
        "product description",
        "item",
        "item name",
        "item description",
        "description",
        "goods",
    ],

    "category": [
        "category",
        "product category",
        "item category",
        "type",
        "product type",
    ],

    "quantity": [
        "quantity",
        "qty",
        "quantity sold",
        "qty sold",
        "units sold",
        "units",
        "sales quantity",
        "sold quantity",
    ],

    "unit_price": [
        "unit price",
        "selling price",
        "sale price",
        "selling rate",
        "price",
        "rate",
        "unit selling price",
    ],

    "cost_price": [
        "cost price",
        "purchase price",
        "buying price",
        "cost",
        "unit cost",
        "purchase cost",
    ],

    "discount": [
        "discount",
        "discount %",
        "discount percent",
        "discount percentage",
    ],
}


# ---------------------------------------------------------
# NORMALIZE COLUMN NAMES
# ---------------------------------------------------------

def normalize_column_name(name: str) -> str:
    """
    Convert a column name into a standardized comparison format.
    """

    name = str(name).strip().lower()

    name = re.sub(r"[%()\-_/]", " ", name)

    name = re.sub(r"\s+", " ", name)

    return name.strip()


# ---------------------------------------------------------
# BUILD ALIAS LOOKUP
# ---------------------------------------------------------

def build_alias_lookup() -> dict:
    """
    Create a lookup table from possible column names
    to canonical column names.
    """

    lookup = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():

        for alias in aliases:

            lookup[normalize_column_name(alias)] = canonical_name

    return lookup


# ---------------------------------------------------------
# DETECT COLUMN MAPPING
# ---------------------------------------------------------

def detect_column_mapping(df: pd.DataFrame) -> dict:
    """
    Detect which uploaded columns correspond to our
    canonical business data columns.
    """

    lookup = build_alias_lookup()

    mapping = {}
    unmapped = []

    for column in df.columns:

        normalized = normalize_column_name(column)

        if normalized in lookup:

            canonical = lookup[normalized]

            # Avoid mapping two columns to the same field.
            if canonical not in mapping.values():

                mapping[column] = canonical

        else:

            unmapped.append(column)

    return {
        "mapping": mapping,
        "unmapped_columns": unmapped,
    }


# ---------------------------------------------------------
# STANDARDIZE DATASET
# ---------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Rename recognized business columns to our canonical schema.
    """

    df = df.copy()

    detection = detect_column_mapping(df)

    mapping = detection["mapping"]

    df = df.rename(columns=mapping)

    return df, detection