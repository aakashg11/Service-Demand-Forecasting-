import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def build_demand_timeseries(df):
    """Aggregates service counts per day per service type — the forecasting target."""
    daily = (
        df.groupby([pd.Grouper(key="Date", freq="D"), "Service_Type"])
        .size()
        .reset_index(name="Service_Count")
    )

    # Fill in missing days with zero counts (important for time series continuity)
    all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    all_types = daily["Service_Type"].dropna().unique()

    full_index = pd.MultiIndex.from_product([all_dates, all_types], names=["Date", "Service_Type"])
    daily_full = daily.set_index(["Date", "Service_Type"]).reindex(full_index, fill_value=0).reset_index()

    # Date-based features useful for LightGBM
    daily_full["day_of_week"] = daily_full["Date"].dt.dayofweek
    daily_full["day_of_month"] = daily_full["Date"].dt.day
    daily_full["month"] = daily_full["Date"].dt.month
    daily_full["is_weekend"] = daily_full["day_of_week"].isin([5, 6]).astype(int)

    # Lag features — yesterday's count and 7-day rolling average per service type
    daily_full = daily_full.sort_values(["Service_Type", "Date"])
    daily_full["lag_1"] = daily_full.groupby("Service_Type")["Service_Count"].shift(1)
    daily_full["rolling_7"] = (
        daily_full.groupby("Service_Type")["Service_Count"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )

    daily_full = daily_full.dropna(subset=["lag_1", "rolling_7"])

    daily_full.to_csv(config.DEMAND_DATA_PATH, index=False)
    print(f"Demand time series saved to {config.DEMAND_DATA_PATH}")
    print(f"Shape: {daily_full.shape}")
    return daily_full


def build_customer_features(df):
    """Aggregates one row per customer — used for churn prediction and LTV segmentation."""
    agg = df.groupby("Customer_ID").agg(
        total_visits=("Date", "count"),
        avg_service_cost=("Service_Cost", "mean"),
        total_spend=("Service_Cost", "sum"),
        avg_rating=("Rating", "mean"),
        avg_wait_time=("Wait_Time_Mins", "mean"),
        avg_parts_cost=("Parts_Cost", "mean"),
        avg_labour_cost=("Labour_Cost", "mean"),
        last_visit_date=("Date", "max"),
        first_visit_date=("Date", "min"),
        return_visit_rate=("Return_Visit", "mean"),
        has_insurance=("Has_Insurance_Cover", "max"),
        used_promo=("Promo_Code_Used", "max"),
    ).reset_index()

    # Customer tenure in days
    agg["tenure_days"] = (agg["last_visit_date"] - agg["first_visit_date"]).dt.days

    # Recency — days since last visit (relative to max date in dataset)
    max_date = df["Date"].max()
    agg["days_since_last_visit"] = (max_date - agg["last_visit_date"]).dt.days

    # Churn label definition: customer is "churned" if no visit in the most recent
    # quarter of the data's date range AND their historical return_visit_rate is low.
    # This is a proxy label since we don't have a true future outcome to observe.
    churn_threshold_days = (max_date - df["Date"].min()).days * 0.25
    agg["churned"] = (agg["days_since_last_visit"] > churn_threshold_days).astype(int)

    agg.to_csv(config.CUSTOMER_FEATURES_PATH, index=False)
    print(f"Customer features saved to {config.CUSTOMER_FEATURES_PATH}")
    print(f"Shape: {agg.shape}")
    print(f"Churn rate: {agg['churned'].mean():.2%}")
    return agg


if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_DATA_PATH, parse_dates=["Date"])
    build_demand_timeseries(df)
    build_customer_features(df)