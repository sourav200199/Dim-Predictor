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

features = ["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]

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
        start_time = dt.datetime.now()
        best_model = None
        best_score = float("inf")
        best_name = None

        for model_name, (model, param_grid) in param_grids.items():

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

            if score < best_score:
                best_score = score
                best_model = search.best_estimator_
                best_name = model_name

        joblib.dump(best_model, model_path)  # Save best model
        end_time = dt.datetime.now()
        logging.info(f"Model training complete. Saved best model ({best_name}) for {target_name} to {model_path}")
        print(f"--> Model training complete in {end_time - start_time}s. Saved best model ({best_name}) for {target_name} to {model_path}")

        return best_model
    except Exception as e:
        logging.error(f"Error in training with best model: {e}")
        raise DIMException(f"Error in training with best model: {e}", sys)
    
def evaluate_models(df, out_path="predictions.csv"):
    print("==> Evaluating model...")

    X = df[features]
    y_len = df["parsed_length"]

    # Load saved models
    reg_length = joblib.load(os.path.join(os.getcwd(), "models", "model_length.pkl"))

    # Predictions
    pred_len = reg_length.predict(X)

    # RMSE calculation
    rmse_len = root_mean_squared_error(y_len, pred_len)

    # Adjusted R2 calculation
    r2 = lambda y, y_pred: 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    adj_r2 = lambda r2, n, p: 1 - (1 - r2) * (n - 1) / (n - p - 1)

    n = len(y_len)
    p = X.shape[1]
    adj_r2_len = adj_r2(r2(y_len, pred_len), n, p)

    # Save predictions with IDs and actual values
    OUTPUT_DIR = os.path.join(os.getcwd(), "predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Output file name with timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
    results = pd.DataFrame({
        "ACTUAL_LENGTH": y_len,
        "PRED_LENGTH": pred_len,
    })

    # Scale to original dimensions before saving
    scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.pkl"))
    scaled_features = df[["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]].copy()
    scaled_features[["parsed_length"]] = pd.DataFrame({
        "parsed_length": pred_len,
    }, index=scaled_features.index)

    original_features = scaler.inverse_transform(scaled_features)
    results[["PRED_LENGTH", "PRED_WIDTH", "PRED_HEIGHT"]] = original_features[:, 1:4]

    results.to_csv(out_path, index=False)
    print(f"==> Predictions saved to {out_path}")

    return results, rmse_len, adj_r2_len
#----------------------------------------------------------------------

def train_models(df):
    X = df[features]
    y_len = df["parsed_length"]

    # Train and save the model
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    reg_length = train_with_best_model(X, y_len, "Length", os.path.join(MODEL_PATH, "model_length.pkl"))

    # Evaluate on training data
    results, rmse_len, adj_r2_len = evaluate_models(df)
    print("\n==> Final Evaluation on Training Set:")
    print(f"##> RMSE : {rmse_len}")
    print(f"##> Adjusted R2 : {adj_r2_len}")

    return reg_length

# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if not os.path.exists(DATA_PATH):
            raise DIMException(f"Data file {DATA_PATH} not found", sys)

        df = pd.read_csv(DATA_PATH)
        print(f"--> Loaded data from {DATA_PATH} with shape {df.shape}\n and columns {df.columns.tolist()}")

        # Train models
        train_models(df)

    except Exception as e:
        logging.error(f"Error in model training script: {e}")
        raise DIMException(f"Error in model training script: {e}", sys)



