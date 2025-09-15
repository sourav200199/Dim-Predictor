######################################################################
# Main Script to execute the entire Pipeline
######################################################################

#----------------------------------------------------------------------
# Importing necessary libraries
#----------------------------------------------------------------------
import os
import datetime as dt
import sys
from scripts.logging.logger import logging
from scripts.exceptions.exceptions import DIMException
import pandas as pd
import joblib
from scripts.pipeline.data_extraction import data_transformation, scale_dimensions, combine_text, extract_width_height, assign_packaging_style
from scripts.pipeline.model_training import train_models, evaluate_models

# ------------------------
# Helper: check if models exist
# ------------------------
def models_exist():
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    os.makedirs(MODEL_PATH, exist_ok=True)

    return all(os.path.exists(os.path.join(MODEL_PATH, f"{dim}_model.pkl")) for dim in ["length", "width", "height"]) and os.path.exists(os.path.join(MODEL_PATH, "scaler.pkl"))


# ------------------------
# Option 1: Train models
# ------------------------
def option_train(fname):
    try:
        if models_exist():
            overwrite = input("Models already exist. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Training aborted. Existing models retained.")
                return

        DATA_PATH = os.path.join(os.getcwd(), "data", fname)
        if not os.path.exists(DATA_PATH):
            print(f"Data file {DATA_PATH} not found.")
            raise DIMException(f"Data file {DATA_PATH} not found", sys)
        
        # Step 1: Data Extraction and Transformation
        data_transformation(fname, is_train=True)

        # Step 2: Model Training
        train_models("extracted_data.csv")

        print("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error in training option: {e}")
        raise DIMException(f"Error in training option: {e}", sys)

#----------------------------------------------------------------------
# Option 2: Batch predictions
# ----------------------------------------------------------------------
def option_predict(fname):
    try:
        if not models_exist():
            print("⚠️ No models found. Please train models first (Option 1).")
            return

        file_path = input("Enter the path of the CSV file to predict: ").strip()
        DATA_PATH = os.path.join(os.getcwd(), "data", fname)
        if not os.path.exists(DATA_PATH):
            print(f"Data file {DATA_PATH} not found.")
            raise DIMException(f"Data file {DATA_PATH} not found", sys)
        
        # Step 1: Data Extraction and Transformation
        df = data_transformation(fname, is_train=False)

        # Step 2: Load the Models and Predict
        m_len = joblib.load(os.path.join(os.getcwd(), "models", "model_length.pkl"))
        m_wid = joblib.load(os.path.join(os.getcwd(), "models", "model_width.pkl"))
        m_hgt = joblib.load(os.path.join(os.getcwd(), "models", "model_height.pkl"))

        # Step 3: Predictions in scaled space
        pred_len_scaled = m_len.predict(df)
        pred_wid_scaled = m_wid.predict(df)
        pred_hgt_scaled = m_hgt.predict(df)

        # Step 4: Inverse transform to original scale
        scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.pkl"))
        scaled_features = df[["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]].copy()
        scaled_features[["parsed_length", "parsed_width", "parsed_height"]] = pd.DataFrame({
            "parsed_length": pred_len_scaled,
            "parsed_width": pred_wid_scaled,
            "parsed_height": pred_hgt_scaled
        }, index=scaled_features.index)

        original_features = scaler.inverse_transform(scaled_features)
        df[["pred_length", "pred_width", "pred_height"]] = original_features[:, 1:4]

        # Step 5: Save predictions
        OUTPUT_DIR = os.path.join(os.getcwd(), "predictions")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # Step 6: Evaluate RMSE and Adjusted R² if actuals are available
        if all(col in df.columns for col in ["parsed_length", "parsed_width", "parsed_height"]):
            results, rmse_values, adj_r2_values = evaluate_models(df)
            print("\n📊 Final Evaluation on Prediction Set:")
            for metric, value in zip(["RMSE", "Adjusted R²"], [rmse_values, adj_r2_values]):
                print(f"  {metric}:")
                for dim, val in zip(["Length", "Width", "Height"], value):
                    print(f"    {dim}: {val:.4f}")
        else:
            print("Actual dimensions not available in the dataset. Skipping evaluation.")
            print("📊 Final Evaluation on Prediction Set:")
            print("  RMSE:")
            for dim in ["Length", "Width", "Height"]:
                print(f"    {dim}: N/A")
            print("  Adjusted R²:")
            for dim in ["Length", "Width", "Height"]:
                print(f"    {dim}: N/A")

    except Exception as e:
        logging.error(f"Error in prediction option: {e}")
        raise DIMException(f"Error in prediction option: {e}", sys)





