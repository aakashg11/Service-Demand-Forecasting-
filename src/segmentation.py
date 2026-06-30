import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def segment_customers():
    df = pd.read_csv(config.CUSTOMER_FEATURES_PATH)

    # Simple LTV proxy score combining spend and visit frequency
    df["ltv_score"] = (
        df["total_spend"].rank(pct=True) * 0.6 +
        df["total_visits"].rank(pct=True) * 0.4
    )

    df["ltv_tier"] = pd.qcut(
        df["ltv_score"], q=[0, 0.5, 0.8, 1.0],
        labels=["Low value", "Medium value", "High value"]
    )

    top_20_pct_revenue = (
        df.sort_values("ltv_score", ascending=False)
        .head(int(len(df) * 0.2))["total_spend"].sum()
        / df["total_spend"].sum()
    )

    print(f"Top 20% of customers contribute {top_20_pct_revenue:.1%} of total revenue")
    print(df["ltv_tier"].value_counts())

    output_path = os.path.join(os.path.dirname(config.CUSTOMER_FEATURES_PATH), "customer_segments.csv")
    df.to_csv(output_path, index=False)
    print(f"Segmented data saved to {output_path}")
    return df


if __name__ == "__main__":
    segment_customers()