import pandas as pd
import numpy as np
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def clean_currency_column(series):
    """Converts strings like '30k', '5000', 'N/A', 'na' into numeric values."""
    def parse_value(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val in ["na", "n/a", "none", "", "nan"]:
            return np.nan
        val = val.replace(",", "").replace("₦", "").replace("$", "")
        match = re.match(r"([\d.]+)\s*(k|m)?", val)
        if not match:
            return np.nan
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix == "k":
            number *= 1_000
        elif suffix == "m":
            number *= 1_000_000
        return number
    return series.apply(parse_value)


def clean_text_column(series):
    """Standardizes text columns: strips whitespace, fixes casing, collapses typos."""
    def parse_text(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        val = re.sub(r"\s+", " ", val)
        return val
    return series.apply(parse_text)


def clean_boolean_column(series):
    """Converts yes/no/true/false variants into 1/0."""
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0,
        "na": np.nan, "n/a": np.nan
    }
    def parse_bool(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        return mapping.get(val, np.nan)
    return series.apply(parse_bool)


def load_and_clean_data():
    df = pd.read_csv(config.RAW_DATA_PATH)

    # Standardize column names (in case of trailing spaces, special chars)
    df.columns = [c.strip().replace(" ", "_").replace("?", "") for c in df.columns]

    # Parse dates
    for date_col in ["Date", "Follow_Up_Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # Clean currency/numeric columns
    for col in ["Service_Cost", "Parts_Cost", "Labour_Cost"]:
        if col in df.columns:
            df[col] = clean_currency_column(df[col])

    # Clean numeric columns that should be plain numbers
    for col in ["Rating", "Service_Time_Hours", "Wait_Time_Mins"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical text columns
    text_cols = [
        "Service_Type", "Vehicle_Type", "Status", "Workshop_Location",
        "Referral_Source", "Spare_Parts_Availability", "Parts_Source",
        "Service_Priority", "Customer_Type", "Mechanic_Skill_Level",
        "Payment_Mode", "Service_Package"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = clean_text_column(df[col])

    # Clean boolean-like columns
    bool_cols = [
        "Return_Visit", "Fuel_TopUp", "Promo_Code_Used", "Has_Insurance_Cover",
        "Request_For_Pickup", "Home_Service", "Towing_Required", "Whatsapp_Followup"
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = clean_boolean_column(df[col])

    # Drop rows with no date or no customer id — unusable for time series/segmentation
    df = df.dropna(subset=["Date", "Customer_ID"])

    # Fill remaining numeric NaNs with median (simple, defensible default)
    numeric_cols = ["Service_Cost", "Parts_Cost", "Labour_Cost", "Rating",
                     "Service_Time_Hours", "Wait_Time_Mins"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {config.PROCESSED_DATA_PATH}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    load_and_clean_data()