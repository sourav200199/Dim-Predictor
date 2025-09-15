######################################################################
# Model training script
######################################################################

#----------------------------------------------------------------------
# Importing necessary libraries
#----------------------------------------------------------------------
import os
import sys
from scripts.logging.logger import logging
from scripts.exceptions.exceptions import DIMException
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import datetime as dt

#----------------------------------------------------------------------
# General configurations    
#----------------------------------------------------------------------
# Input Data File
DATA_FILE = "extracted_data.csv"  
DATA_PATH = os.path.join(os.getcwd(), "data", DATA_FILE)

# Output Model File
MODELS_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

features = ["TYPE_FREQ", "PARSED_LENGTH", "PARSED_WIDTH", "PARSED_HEIGHT", "PACKAGING_STYLE"]

#----------------------------------------------------------------------
# Candidate models and their hyperparameters
#----------------------------------------------------------------------
param_grids = {
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    ),
    "Ridge": (
        Ridge(random_state=42),
        {
            "alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
        },
    ),
    "KNN": (
        KNeighborsRegressor(),
        {
            "n_neighbors": [3, 5, 10, 20],
            "weights": ["uniform", "distance"],
        },
    ),
}

#----------------------------------------------------------------------
# Main function to execute model training
#----------------------------------------------------------------------
def train_with_best_model(X, y, target_name, model_path):
    try:
        best_model = None
        best_score = float("inf")
        best_name = None

        for model_name, (model, param_grid) in param_grids.items():
            print(f"🔍 Running RandomizedSearchCV for {model_name} on {target_name}...")

            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=5,
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            search.fit(X, y)

            score = -search.best_score_  # since scoring is neg RMSE
            print(f"✅ {model_name} best RMSE for {target_name}: {score:.2f} with params {search.best_params_}")

            if score < best_score:
                best_score = score
                best_model = search.best_estimator_
                best_name = model_name

        print(f"🏆 Best model for {target_name}: {best_name} with RMSE {best_score:.2f}")
        joblib.dump(best_model, model_path)  # Save best model
        print(f"💾 Saved best {target_name} model to {model_path}")
        return best_model
    except Exception as e:
        logging.error(f"Error in training with best model: {e}")
        raise DIMException(f"Error in training with best model: {e}", sys)
    
def train_models(df):
    X = df[features]
    y_len = df["parsed_length"]
    y_wid = df["parsed_width"]
    y_hgt = df["parsed_height"]

    reg_length = train_with_best_model(X, y_len, "Length", "model_length.pkl")
    reg_width = train_with_best_model(X, y_wid, "Width", "model_width.pkl")
    reg_height = train_with_best_model(X, y_hgt, "Height", "model_height.pkl")

    return reg_length, reg_width, reg_height

def evaluate_models(df, out_path="predictions.csv"):
    X = df[features]
    y_len = df["parsed_length"]
    y_wid = df["parsed_width"]
    y_hgt = df["parsed_height"]

    # Load saved models
    reg_length = joblib.load("model_length.pkl")
    reg_width = joblib.load("model_width.pkl")
    reg_height = joblib.load("model_height.pkl")

    # Predictions
    pred_len = reg_length.predict(X)
    pred_wid = reg_width.predict(X)
    pred_hgt = reg_height.predict(X)

    # RMSE calculation
    rmse_len = root_mean_squared_error(y_len, pred_len)
    rmse_wid = root_mean_squared_error(y_wid, pred_wid)
    rmse_hgt = root_mean_squared_error(y_hgt, pred_hgt)

    # Adjusted R2 calculation
    r2 = lambda y, y_pred: 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    adj_r2 = lambda r2, n, p: 1 - (1 - r2) * (n - 1) / (n - p - 1)

    n = len(y_len)
    p = X.shape[1]
    adj_r2_len = adj_r2(r2(y_len, pred_len), n, p)
    adj_r2_wid = adj_r2(r2(y_wid, pred_wid), n, p)
    adj_r2_hgt = adj_r2(r2(y_hgt, pred_hgt), n, p)

    # Save predictions with IDs and actual values
    OUTPUT_DIR = os.path.join(os.getcwd(), "predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Output file name with timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
    results = pd.DataFrame({
        "PRODUCT_ID": df["PRODUCT_ID"],
        "ACTUAL_LENGTH": y_len,
        "PRED_LENGTH": pred_len,
        "ACTUAL_WIDTH": y_wid,
        "PRED_WIDTH": pred_wid,
        "ACTUAL_HEIGHT": y_hgt,
        "PRED_HEIGHT": pred_hgt,
    })
    results.to_csv(out_path, index=False)
    print(f"💾 Predictions saved to {out_path}")

    return results, (rmse_len, rmse_wid, rmse_hgt), (adj_r2_len, adj_r2_wid, adj_r2_hgt)
#----------------------------------------------------------------------
