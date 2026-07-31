"""
modules/data.py
================
Handles: CSV ingestion, validation, cleaning, and loading into SQLite.

Design decisions:
- We use SQLAlchemy's engine (not raw sqlite3) so the rest of the app
  (analytics, forecasting, text-to-SQL) can all share one consistent
  connection interface, and so it's trivial to swap SQLite for Postgres
  later if this were a real client deployment.
- Validation returns a structured report (dict) rather than just printing,
  so the Streamlit UI can render ingestion stats as KPI cards.
- Cleaning is conservative: we never silently drop revenue-bearing rows
  without reporting it. A "black box" cleaning step is a red flag in a
  consulting-style analytics tool — clients need to trust the numbers.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "sales.db"
TABLE_NAME = "sales"

REQUIRED_COLUMNS = [
    "Transaction_ID", "Date", "Region", "Country", "Drug_Name",
    "Therapy_Area", "Hospital", "Doctor_Segment", "Sales_Representative",
    "Units_Sold", "Revenue", "Discount", "Manufacturing_Cost", "Profit",
    "Marketing_Spend", "Inventory", "Customer_Type", "Quarter", "Month"
]


def get_engine():
    """Single shared SQLAlchemy engine, used across all modules."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run structural + quality checks on an uploaded dataframe.
    Returns a report dict — never raises, so the UI can show
    warnings instead of crashing on messy client data.
    """
    report = {
        "row_count": len(df),
        "missing_columns": [c for c in REQUIRED_COLUMNS if c not in df.columns],
        "missing_values": {},
        "duplicate_rows": 0,
        "negative_revenue_rows": 0,
        "is_valid": True,
    }

    if report["missing_columns"]:
        report["is_valid"] = False
        return report  # can't check further without the right columns

    # Missing values per column
    missing = df[REQUIRED_COLUMNS].isnull().sum()
    report["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}

    # Duplicate transaction IDs
    report["duplicate_rows"] = int(df["Transaction_ID"].duplicated().sum())

    # Sanity check: revenue shouldn't be negative
    report["negative_revenue_rows"] = int((df["Revenue"] < 0).sum())

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe based on validation findings.
    Steps are explicit and ordered so behavior is auditable.
    """
    df = df.copy()

    # 1. Drop exact duplicate transactions
    df = df.drop_duplicates(subset="Transaction_ID", keep="first")

    # 2. Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # 3. Numeric columns: coerce, fill sensible defaults
    numeric_cols = ["Units_Sold", "Revenue", "Discount", "Manufacturing_Cost",
                     "Profit", "Marketing_Spend", "Inventory"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where core financial fields are unrecoverable
    df = df.dropna(subset=["Revenue", "Units_Sold"])

    # Fill non-critical numeric gaps with 0 (e.g. missing marketing spend)
    df[["Discount", "Marketing_Spend"]] = df[["Discount", "Marketing_Spend"]].fillna(0)

    # 4. Recompute Profit if missing or inconsistent, so downstream KPIs are trustworthy
    df["Profit"] = df["Revenue"] - df["Manufacturing_Cost"].fillna(0) - df["Discount"]

    # 5. Drop rows with negative revenue (bad data, not a real business case here)
    df = df[df["Revenue"] >= 0]

    # 6. Standardize text fields
    text_cols = ["Region", "Country", "Drug_Name", "Therapy_Area",
                 "Hospital", "Doctor_Segment", "Sales_Representative", "Customer_Type"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # 7. Derive Quarter/Month from Date (overrides any inconsistent source values)
    df["Month"] = df["Date"].dt.strftime("%B")
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)

    return df.reset_index(drop=True)


def load_to_sqlite(df: pd.DataFrame, replace: bool = True) -> int:
    """Write the cleaned dataframe into SQLite. Returns rows written."""
    engine = get_engine()
    mode = "replace" if replace else "append"
    df.to_sql(TABLE_NAME, engine, if_exists=mode, index=False)
    return len(df)


def load_from_sqlite() -> pd.DataFrame:
    """Read the full sales table back out for analytics/dashboard use."""
    engine = get_engine()
    try:
        return pd.read_sql_table(TABLE_NAME, engine)
    except ValueError:
        # Table doesn't exist yet (no data uploaded)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def ingest_pipeline(df: pd.DataFrame) -> dict:
    """
    Full ingestion flow used by the Streamlit upload page:
    validate -> clean -> load -> return stats for display.
    """
    validation_report = validate_data(df)

    if validation_report["missing_columns"]:
        return {"success": False, "validation": validation_report}

    cleaned_df = clean_data(df)
    rows_written = load_to_sqlite(cleaned_df)

    return {
        "success": True,
        "validation": validation_report,
        "rows_ingested": rows_written,
        "rows_dropped": validation_report["row_count"] - rows_written,
    }
