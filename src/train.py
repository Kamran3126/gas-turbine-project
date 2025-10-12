# ----------------- Imports -----------------
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import yaml
import argparse
import json # <-- کتابخانه جدید برای کار با فایل جیسون

warnings.filterwarnings('ignore')

# Function to read params from yaml file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# ----------------- Main Execution Block -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    config = read_params(args.config)

    data_path = config["data_source"]["s3_source"]
    target_col = config["load_data"]["target_col"]
    test_size = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    n_estimators = config["train_model"]["n_estimators"]
    objective = config["train_model"]["objective"]

    df = pd.read_csv(data_path)

    y = df[target_col]
    X = df.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("--> Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective=objective, n_estimators=n_estimators, random_state=random_state)
    xgb_model.fit(X_train, y_train)
    print("--> Model training complete!")

    y_pred = xgb_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- XGBoost Model Evaluation ---")
    print(f"   RMSE: {rmse:.4f} MW")
    print(f"   R² Score: {r2:.4f}")
    print("------------------------------")

    # ----- بخش جدید: ذخیره متریک‌ها در فایل جیسون -----
    metrics = {
        "rmse": rmse,
        "r2_score": r2
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("--> Metrics saved to metrics.json")
    # ---------------------------------------------------