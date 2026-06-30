cat > README.md << 'EOF'
# AI-Powered Aftersales Service Demand Forecasting & Client Insight Dashboard

An end-to-end ML pipeline and interactive dashboard for automotive aftersales analytics — built to demonstrate demand forecasting, customer churn prediction, and lifetime value segmentation.

## Setup

```bash
pip install -r requirements.txt
brew install libomp        # macOS only — required by LightGBM
```

## Data

Dataset: [Bobo Benz Workshop — Auto Service Records](https://www.kaggle.com/)

Download the CSV and place it at: data/raw/bobo_benz_workshop.csv

## Run the pipeline

```bash
python3 src/preprocessing.py
python3 src/feature_engineering.py
python3 src/train_demand_model.py
python3 src/train_churn_model.py
python3 src/segmentation.py
python3 src/evaluate.py
```

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

## Models

| Model | Task | Algorithm | Key metric |
|---|---|---|---|
| Demand forecasting | Predict daily service volume per type | LightGBM Regressor | MAE: 0.531 |
| Churn prediction | Flag high-risk customers | LightGBM Classifier + SHAP | ROC-AUC: 0.484* |
| LTV segmentation | Segment customers by lifetime value | Rule-based scoring | Top 20% → 34% revenue |

*AUC reflects proxy label limitation on a 1,000-record simulated dataset. Methodology documented in evaluate.py.

## Tech stack

Python · LightGBM · SHAP · Streamlit · Plotly · scikit-learn · Pandas
EOF