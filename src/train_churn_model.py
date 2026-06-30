import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import shap
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def train_churn_model():
    df = pd.read_csv(config.CUSTOMER_FEATURES_PATH)

    feature_cols = [
        "total_visits", "avg_service_cost", "total_spend", "avg_rating",
        "avg_wait_time", "avg_parts_cost", "avg_labour_cost",
        "tenure_days", "return_visit_rate", "has_insurance", "used_promo"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    print(f"Churn model ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds))

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, config.CHURN_MODEL_PATH)
    joblib.dump(feature_cols, os.path.join(config.MODELS_DIR, "churn_feature_cols.pkl"))
    print(f"Model saved to {config.CHURN_MODEL_PATH}")

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                       X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "shap_summary.png"), dpi=150)
    print(f"SHAP summary plot saved to {config.OUTPUTS_DIR}/shap_summary.png")

    return model, auc


if __name__ == "__main__":
    train_churn_model()