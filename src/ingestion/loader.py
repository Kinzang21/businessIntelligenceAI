from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Excel file from a Windows file path.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    if extension == ".csv":
        return pd.read_csv(file_path)

    return pd.read_excel(file_path)


def load_uploaded_file(
    uploaded_file: BinaryIO,
    file_name: str
) -> pd.DataFrame:
    """
    Load a CSV or Excel file uploaded through Streamlit.
    """

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)