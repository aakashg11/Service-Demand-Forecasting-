import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def train_demand_model():
    df = pd.read_csv(config.DEMAND_DATA_PATH, parse_dates=["Date"])

    # One-hot encode service type since LightGBM needs numeric input
    df = pd.get_dummies(df, columns=["Service_Type"], prefix="svc")

    feature_cols = [c for c in df.columns if c not in ["Date", "Service_Count"]]
    X = df[feature_cols]
    y = df["Service_Count"]

    # Time-based split — train on earlier dates, test on later ones (no shuffling)
    split_date = df["Date"].quantile(0.8, interpolation="nearest")
    train_mask = df["Date"] <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"Demand model RMSE: {rmse:.3f}")
    print(f"Demand model MAE: {mae:.3f}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, config.DEMAND_MODEL_PATH)
    joblib.dump(feature_cols, os.path.join(config.MODELS_DIR, "demand_feature_cols.pkl"))
    print(f"Model saved to {config.DEMAND_MODEL_PATH}")

    return model, rmse, mae


if __name__ == "__main__":
    train_demand_model()