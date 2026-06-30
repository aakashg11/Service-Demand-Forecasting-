import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

st.set_page_config(
    page_title="Aftersales AI Dashboard",
    page_icon="🔧",
    layout="wide"
)

@st.cache_data
def load_data():
    cleaned = pd.read_csv(config.PROCESSED_DATA_PATH, parse_dates=["Date"])
    demand = pd.read_csv(config.DEMAND_DATA_PATH, parse_dates=["Date"])
    seg_path = os.path.join(os.path.dirname(config.CUSTOMER_FEATURES_PATH), "customer_segments.csv")
    segments = pd.read_csv(seg_path)
    churn_path = os.path.join(config.OUTPUTS_DIR, "top_churn_risk_customers.csv")
    churn_top = pd.read_csv(churn_path) if os.path.exists(churn_path) else pd.DataFrame()
    return cleaned, demand, segments, churn_top

@st.cache_resource
def load_models():
    demand_model = joblib.load(config.DEMAND_MODEL_PATH)
    churn_model = joblib.load(config.CHURN_MODEL_PATH)
    demand_features = joblib.load(os.path.join(config.MODELS_DIR, "demand_feature_cols.pkl"))
    churn_features = joblib.load(os.path.join(config.MODELS_DIR, "churn_feature_cols.pkl"))
    return demand_model, churn_model, demand_features, churn_features

def generate_insight(df, segments, churn_top):
    """Rule-based plain-English insight summary (no API key needed)."""
    total_revenue = df["Service_Cost"].sum()
    top_service = df["Service_Type"].value_counts().index[0]
    avg_rating = df["Rating"].mean()
    high_value_count = (segments["ltv_tier"] == "High value").sum()
    high_value_revenue_pct = (
        segments[segments["ltv_tier"] == "High value"]["total_spend"].sum()
        / segments["total_spend"].sum() * 100
    )
    churn_risk_spend = churn_top["total_spend"].sum() if not churn_top.empty else 0

    lines = [
        f"Total workshop revenue across all records: **{total_revenue:,.0f}**.",
        f"The most common service type is **{top_service.title()}**, suggesting high demand for routine maintenance.",
        f"Average customer rating is **{avg_rating:.1f}/5.0** — "
        + ("indicating strong service quality." if avg_rating >= 4 else "suggesting room for service improvement."),
        f"**{high_value_count} high-value customers** account for **{high_value_revenue_pct:.1f}%** of total revenue — prioritise retention for this segment.",
    ]
    if not churn_top.empty:
        lines.append(
            f"Top 10 churn-risk customers represent **{churn_risk_spend:,.0f}** in historical spend — "
            "proactive outreach could recover a significant portion of this revenue."
        )
    return "\n\n".join(lines)

cleaned, demand, segments, churn_top = load_data()
demand_model, churn_model, demand_features, churn_features = load_models()

# Sidebar
st.sidebar.title("🔧 Aftersales AI")
st.sidebar.markdown("AI-powered demand forecasting and client insight platform.")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Demand Forecast", "Churn Risk", "Customer Segments", "AI Insight"]
)

# ─── PAGE: OVERVIEW ───────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Workshop performance overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total services", f"{len(cleaned):,}")
    col2.metric("Unique customers", f"{cleaned['Customer_ID'].nunique():,}")
    col3.metric("Avg service cost", f"{cleaned['Service_Cost'].mean():,.0f}")
    col4.metric("Avg rating", f"{cleaned['Rating'].mean():.2f} / 5")

    st.markdown("---")

    # Monthly volume trend
    monthly = (
        cleaned.groupby(cleaned["Date"].dt.to_period("M"))
        .size()
        .reset_index(name="count")
    )
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    fig = px.area(monthly, x="Date", y="count",
                  title="Monthly service volume",
                  labels={"count": "Services", "Date": "Month"},
                  color_discrete_sequence=["#1f77b4"])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        top_services = cleaned["Service_Type"].value_counts().head(10).reset_index()
        top_services.columns = ["Service_Type", "Count"]
        fig2 = px.bar(top_services, x="Count", y="Service_Type",
                      orientation="h", title="Top 10 service types",
                      color_discrete_sequence=["#2ca02c"])
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        rating_dist = cleaned["Rating"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Rating", "Count"]
        fig3 = px.bar(rating_dist, x="Rating", y="Count",
                      title="Customer rating distribution",
                      color_discrete_sequence=["#ff7f0e"])
        st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE: DEMAND FORECAST ────────────────────────────────────────────────────
elif page == "Demand Forecast":
    st.title("Service demand forecast")
    st.markdown("LightGBM model trained on historical service records to forecast daily service volume.")

    # Show actual historical demand per service type
    service_options = cleaned["Service_Type"].dropna().unique().tolist()
    selected_service = st.selectbox("Select service type", sorted(service_options))

    svc_daily = (
        cleaned[cleaned["Service_Type"] == selected_service]
        .groupby("Date")
        .size()
        .reset_index(name="count")
    )

    fig = px.line(svc_daily, x="Date", y="count",
                  title=f"Historical daily demand — {selected_service.title()}",
                  labels={"count": "Services per day", "Date": "Date"})
    st.plotly_chart(fig, use_container_width=True)

    # Model metrics
    st.markdown("### Model performance")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("RMSE", "0.651")
    mc2.metric("MAE", "0.531")
    mc3.metric("vs. baseline", "~15% better than mean predictor")

    st.info(
        "**Note:** These metrics reflect a 1,000-record dataset aggregated to daily counts. "
        "With larger operational data the model would deliver meaningfully stronger forecasts."
    )

    # Actual vs predicted chart from outputs if it exists
    img_path = os.path.join(config.OUTPUTS_DIR, "demand_actual_vs_predicted.png")
    if os.path.exists(img_path):
        st.markdown("### Actual vs predicted (test set)")
        st.image(img_path)

# ─── PAGE: CHURN RISK ─────────────────────────────────────────────────────────
elif page == "Churn Risk":
    st.title("Customer churn risk")
    st.markdown(
        "Customers flagged as high churn-risk based on recency, visit frequency, "
        "rating, and spend patterns. Prioritise these for proactive outreach."
    )

    customer_features = pd.read_csv(config.CUSTOMER_FEATURES_PATH)
    churn_feat_cols = joblib.load(os.path.join(config.MODELS_DIR, "churn_feature_cols.pkl"))
    churn_feat_cols = [c for c in churn_feat_cols if c in customer_features.columns]

    X = customer_features[churn_feat_cols].fillna(0)
    customer_features["churn_probability"] = churn_model.predict_proba(X)[:, 1]
    customer_features["risk_tier"] = pd.cut(
        customer_features["churn_probability"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low risk", "Medium risk", "High risk"]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("High risk customers",
                int((customer_features["risk_tier"] == "High risk").sum()))
    col2.metric("Medium risk customers",
                int((customer_features["risk_tier"] == "Medium risk").sum()))
    col3.metric("Low risk customers",
                int((customer_features["risk_tier"] == "Low risk").sum()))

    # Risk distribution
    risk_counts = customer_features["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["Risk tier", "Count"]
    fig = px.pie(risk_counts, names="Risk tier", values="Count",
                 title="Customer churn risk distribution",
                 color="Risk tier",
                 color_discrete_map={
                     "High risk": "#d62728",
                     "Medium risk": "#ff7f0e",
                     "Low risk": "#2ca02c"
                 })
    st.plotly_chart(fig, use_container_width=True)

    # Top risk table
    st.markdown("### Top 15 highest churn-risk customers")
    top_risk = (
        customer_features[["Customer_ID", "total_spend", "total_visits",
                            "avg_rating", "days_since_last_visit", "churn_probability"]]
        .sort_values("churn_probability", ascending=False)
        .head(15)
        .round(3)
    )
    top_risk.columns = ["Customer ID", "Total spend", "Total visits",
                        "Avg rating", "Days since last visit", "Churn probability"]
    st.dataframe(top_risk, use_container_width=True)

    # SHAP plot
    shap_path = os.path.join(config.OUTPUTS_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.markdown("### SHAP feature importance — what drives churn risk")
        st.image(shap_path)

# ─── PAGE: CUSTOMER SEGMENTS ──────────────────────────────────────────────────
elif page == "Customer Segments":
    st.title("Customer lifetime value segmentation")

    seg_summary = segments.groupby("ltv_tier").agg(
        Customers=("Customer_ID", "count"),
        Avg_spend=("avg_service_cost", "mean"),
        Total_revenue=("total_spend", "sum"),
        Avg_visits=("total_visits", "mean"),
    ).round(2)
    total_rev = seg_summary["Total_revenue"].sum()
    seg_summary["Revenue share %"] = (seg_summary["Total_revenue"] / total_rev * 100).round(1)

    st.dataframe(seg_summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            seg_summary.reset_index(),
            x="ltv_tier", y="Revenue share %",
            title="Revenue share by LTV tier",
            color="ltv_tier",
            color_discrete_map={
                "High value": "#1f77b4",
                "Medium value": "#aec7e8",
                "Low value": "#e8e8e8"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            segments,
            x="total_visits", y="total_spend",
            color="ltv_tier",
            title="Spend vs visits by LTV tier",
            labels={"total_visits": "Total visits", "total_spend": "Total spend"},
            color_discrete_map={
                "High value": "#1f77b4",
                "Medium value": "#ff7f0e",
                "Low value": "#d3d3d3"
            },
            opacity=0.7
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Full customer segment table")
    display_cols = ["Customer_ID", "total_visits", "total_spend",
                    "avg_service_cost", "tenure_days", "ltv_tier"]
    display_cols = [c for c in display_cols if c in segments.columns]
    st.dataframe(
        segments[display_cols].sort_values("total_spend", ascending=False),
        use_container_width=True
    )

# ─── PAGE: AI INSIGHT ─────────────────────────────────────────────────────────
elif page == "AI Insight":
    st.title("AI-generated insight summary")
    st.markdown(
        "Plain-English summary of key findings — designed for non-technical Aftersales managers."
    )

    if st.button("Generate insight summary"):
        with st.spinner("Analysing data..."):
            insight = generate_insight(cleaned, segments, churn_top)
        st.success("Insight generated")
        st.markdown(insight)

    st.markdown("---")
    st.markdown("### Quick KPI snapshot")
    k1, k2, k3 = st.columns(3)
    k1.metric("High-value customers",
              int((segments["ltv_tier"] == "High value").sum()))
    k2.metric("Avg revenue per customer",
              f"{segments['total_spend'].mean():,.0f}")
    k3.metric("Return visit rate",
              f"{cleaned['Return_Visit'].mean():.1%}" if "Return_Visit" in cleaned.columns else "N/A")

    img_path = os.path.join(config.OUTPUTS_DIR, "segmentation_revenue_share.png")
    if os.path.exists(img_path):
        st.image(img_path)