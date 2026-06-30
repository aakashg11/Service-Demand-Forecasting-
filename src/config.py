import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "bobo_benz_workshop.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
DEMAND_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "demand_timeseries.csv")
CUSTOMER_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, "demand_model.pkl")
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")

RANDOM_STATE = 42