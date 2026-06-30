import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import os
import sys
from sklearn.metrics import (
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
    mean_squared_error, mean_absolute_error
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def evaluate_demand_model():
    print("\n=== DEMAND MODEL EVALUATION ===")
    df = pd.read_csv(config.DEMAND_DATA_PATH, parse_dates=["Date"])
    df = pd.get_dummies(df, columns=["Service_Type"], prefix="svc")

    feature_cols = joblib.load(
        os.path.join(config.MODELS_DIR, "demand_feature_cols.pkl")
    )
    # Keep only cols that exist in this df (in case of dummies mismatch)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["Service_Count"]

    split_date = df["Date"].quantile(0.8, interpolation="nearest")
    test_mask = df["Date"] > split_date
    X_test = X[test_mask]
    y_test = y[test_mask]

    model = joblib.load(config.DEMAND_MODEL_PATH)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # no negative service counts

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))

    print(f"  RMSE          : {rmse:.3f}")
    print(f"  MAE           : {mae:.3f}")
    print(f"  Baseline MAE  : {baseline_mae:.3f}  (predicting mean every day)")
    print(f"  Improvement   : {((baseline_mae - mae) / baseline_mae * 100):.1f}% over baseline")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values[:60], label="Actual", alpha=0.8)
    ax.plot(preds[:60], label="Predicted", alpha=0.8, linestyle="--")
    ax.set_title("Demand forecast — actual vs predicted (first 60 test points)")
    ax.set_ylabel("Service count")
    ax.set_xlabel("Test sample index")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUTS_DIR, "demand_actual_vs_predicted.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {out_path}")

    return {"rmse": rmse, "mae": mae, "baseline_mae": baseline_mae}


def evaluate_churn_model():
    print("\n=== CHURN MODEL EVALUATION ===")
    df = pd.read_csv(config.CUSTOMER_FEATURES_PATH)

    feature_cols = joblib.load(
        os.path.join(config.MODELS_DIR, "churn_feature_cols.pkl")
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["churned"]

    model = joblib.load(config.CHURN_MODEL_PATH)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    auc = roc_auc_score(y, probs)
    print(f"  ROC-AUC       : {auc:.3f}")
    print(f"  NOTE: AUC near 0.5 reflects proxy label limitation on a 1k-row")
    print(f"        simulated dataset. See README for methodology notes.")
    print(f"\n  Classification report (full dataset):")
    print(classification_report(y, preds))

    # Churn risk table — top 10 highest-risk customers
    df["churn_probability"] = probs
    top_risk = (
        df[["Customer_ID", "total_spend", "total_visits",
            "avg_rating", "churn_probability"]]
        .sort_values("churn_probability", ascending=False)
        .head(10)
    )
    print("\n  Top 10 highest churn-risk customers:")
    print(top_risk.to_string(index=False))

    top_risk_path = os.path.join(config.OUTPUTS_DIR, "top_churn_risk_customers.csv")
    top_risk.to_csv(top_risk_path, index=False)
    print(f"\n  Saved to {top_risk_path}")

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, probs, ax=ax, name="LightGBM Churn")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("Churn model — ROC curve")
    ax.legend()
    plt.tight_layout()
    roc_path = os.path.join(config.OUTPUTS_DIR, "churn_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved to {roc_path}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, preds, ax=ax)
    ax.set_title("Churn model — confusion matrix")
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUTS_DIR, "churn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {cm_path}")

    return {"auc": auc, "churn_probs": df[["Customer_ID", "churn_probability"]]}


def evaluate_segmentation():
    print("\n=== SEGMENTATION SUMMARY ===")
    seg_path = os.path.join(
        os.path.dirname(config.CUSTOMER_FEATURES_PATH), "customer_segments.csv"
    )
    df = pd.read_csv(seg_path)

    summary = df.groupby("ltv_tier").agg(
        customer_count=("Customer_ID", "count"),
        avg_spend=("avg_service_cost", "mean"),
        total_revenue=("total_spend", "sum"),
        avg_visits=("total_visits", "mean"),
    ).round(2)

    total_rev = summary["total_revenue"].sum()
    summary["revenue_share_%"] = (summary["total_revenue"] / total_rev * 100).round(1)

    print(summary.to_string())

    # Bar chart of revenue share by tier
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#b0bec5", "#78909c", "#37474f"]
    summary["revenue_share_%"].plot(kind="bar", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Revenue share by customer LTV tier")
    ax.set_ylabel("Revenue share (%)")
    ax.set_xlabel("LTV tier")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    seg_path_out = os.path.join(config.OUTPUTS_DIR, "segmentation_revenue_share.png")
    plt.savefig(seg_path_out, dpi=150)
    plt.close()
    print(f"\n  Chart saved to {seg_path_out}")

    return summary


if __name__ == "__main__":
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    demand_metrics = evaluate_demand_model()
    churn_results = evaluate_churn_model()
    seg_summary = evaluate_segmentation()
    print("\n=== ALL EVALUATIONS COMPLETE ===")
    print(f"Outputs saved to: {config.OUTPUTS_DIR}")