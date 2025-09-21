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

    return all(os.path.exists(os.path.join(MODEL_PATH, f"model_{dim}.pkl")) for dim in ["length", "width", "height"]) and os.path.exists(os.path.join(MODEL_PATH, "scaler.pkl"))


# ------------------------
# Option 1: Train models
# ------------------------
def option_train():
    try:
        if models_exist():
            overwrite = input("Models already exist. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Training aborted. Existing models retained.")
                return

        fname = input("Enter the filename of the CSV (with .csv extension): ").strip()  
        fname = fname if fname.endswith('.csv') else fname + '.csv'
        DATA_PATH = os.path.join(os.getcwd(), "data", fname)
        if not os.path.exists(DATA_PATH):
            print(f"Data file {DATA_PATH} not found.")
            raise DIMException(f"Data file {DATA_PATH} not found", sys)
        
        print("==> File uploaded successfully. \n==> Starting data transformation...")
        
        # Step 1: Data Extraction and Transformation
        df = data_transformation(fname, is_train=True)
        print("==> Starting model training...")

        # Step 2: Model Training
        start_time = dt.datetime.now()
        train_models(df)
        end_time = dt.datetime.now()
        print(f"==> Model training completed in {end_time - start_time}s.")

    except Exception as e:
        logging.error(f"Error in training option: {e}")
        raise DIMException(f"Error in training option: {e}", sys)

#----------------------------------------------------------------------
# Option 2: Batch predictions
# ----------------------------------------------------------------------
def option_predict():
    try:
        if not models_exist():
            print("** No models found. Please train models first (Option 1). **")
            return

        # ---------- Step 1: Load input CSV ----------
        fname = input("Enter the filename of the CSV (with .csv extension): ").strip()
        fname = fname if fname.endswith('.csv') else fname + '.csv'
        DATA_PATH = os.path.join(os.getcwd(), "data", fname)
        if not os.path.exists(DATA_PATH):
            raise DIMException(f"Data file {DATA_PATH} not found", sys)

        # ---------- Step 2: Data Transformation ----------
        print("==> File uploaded successfully. \n==> Starting data transformation...")
        print("** Please note that the rows having missing dimensions will be dropped. **")

        df = data_transformation(fname, is_train=False)

        # ---------- Step 3: Load models ----------
        m_len = joblib.load(os.path.join(os.getcwd(), "models", "model_length.pkl"))

        # ---------- Step 4: Predict (scaled space) ----------
        pred_len_scaled = m_len.predict(df)

        # ---------- Step 5: Inverse transform ----------
        scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.pkl"))

        # Copy scaled features
        scaled_features = df[["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]].copy()

        # Replace parsed_length with predicted values
        scaled_features.loc[:, "parsed_length"] = pred_len_scaled

        # Inverse transform back to original scale
        original_features = scaler.inverse_transform(scaled_features)

        # Add predictions to df
        df["pred_length"] = original_features[:, 1]   # Length
        df["pred_width"]  = original_features[:, 2]   # Width
        df["pred_height"] = original_features[:, 3]   # Height

        # ---------- Step 6: Save predictions ----------
        OUTPUT_DIR = os.path.join(os.getcwd(), "predictions")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        print(f"==> Predictions saved to {output_path}")

        # ---------- Step 7: Evaluate only if actuals exist ----------
        if {"parsed_length", "parsed_width", "parsed_height"}.issubset(df.columns) and not df["parsed_length"].isna().all():
            results, rmse_values, adj_r2_values = evaluate_models(df, out_path=output_path)
            print("\n==> Final Evaluation on Prediction Set:")
            print(f"  RMSE: {rmse_values:.4f}")
            print(f"  Adjusted R²: {adj_r2_values:.4f}")
        else:
            print("\n==> Final Evaluation skipped: No actual dimensions available.")

    except Exception as e:
        logging.error(f"Error in prediction option: {e}")
        raise DIMException(f"Error in prediction option: {e}", sys)

#----------------------------------------------------------------------
# Option 3: Single prediction
#----------------------------------------------------------------------
def option_single_predict():
    if not models_exist():
        print("** No models found. Please train models first (Option 1). **")
        return

    try:
        # Collect raw input
        pid = input("Enter PRODUCT_ID: ").strip()
        title = input("Enter TITLE: ").strip()
        bullet = input("Enter BULLET_POINTS: ").strip()
        desc = input("Enter DESCRIPTION: ").strip()
        ptype = int(input("Enter PRODUCT_TYPE_ID: ").strip())
        length = float(input("Enter PRODUCT_LENGTH: ").strip())

        # Apply transformations
        combined_text = combine_text(pd.Series({
            "TITLE": title,
            "BULLET_POINTS": bullet,
            "DESCRIPTION": desc
        }))
        width, height = extract_width_height(combined_text)

        combined_text = pd.Series(combined_text)
        data = pd.DataFrame([{
            "PRODUCT_ID": pid,
            "PRODUCT_TYPE_ID": ptype,
            "TYPE_FREQ": 1,
            "parsed_length": length,
            "parsed_width": width if width else 0,
            "parsed_height": height if height else 0,
            "PACKAGING_STYLE": assign_packaging_style(combined_text)
        }])

        # Load scaler and features
        scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.pkl"))
        feature_cols = ["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]

        # Scale only these columns
        scaled_array = scaler.transform(data[feature_cols])
        scaled_data = pd.DataFrame(scaled_array, columns=feature_cols)

        # Load models
        reg_length = joblib.load(os.path.join(os.getcwd(), "models", "model_length.pkl"))

        # Predictions in scaled space (no warnings now)
        pred_len_scaled = reg_length.predict(scaled_data)[0]

        print(f"Scaled Predictions → Length: {pred_len_scaled:.9f}")

        # Replace into scaled row and inverse transform
        scaled_row = scaled_data.copy()
        scaled_row.loc[0, "parsed_length"] = pred_len_scaled

        # Inverse transform
        original_row = scaler.inverse_transform(scaled_row)
        pred_len = original_row[0, list(scaled_row.columns).index("parsed_length")]

        msg = f"📦 Predicted Box Dimensions → Length: {pred_len:.2f}"
        print(msg)

    except Exception as e:
        logging.error(f"Invalid input: {e}")
        raise DIMException(f"Invalid input: {e}", sys)
    
#----------------------------------------------------------------------
# Main Execution
#----------------------------------------------------------------------
def main_menu(n):
    if n == 1:
        option_train()
    elif n == 2:
        option_predict()
    elif n == 3:
        option_single_predict()
    elif n == 4:
        print("Exiting the program.")
        return
    else:
        print("Invalid choice. Please select a valid option.")

    print("\n" + "="*50)
    print("📦 Dimensional Prediction System")
    print("="*50)
    print("Select an option:")
    print("1. Train Models")
    print("2. Batch Predictions")
    print("3. Single Prediction")
    print("4. Exit")

    choice = int(input("Enter your choice (1-4): "))
    print("="*50)

    main_menu(choice)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("📦 Dimensional Prediction System")
    print("="*50)
    print("Select an option:")
    print("1. Train Models")
    print("2. Batch Predictions")
    print("3. Single Prediction")
    print("4. Exit")

    choice = int(input("Enter your choice (1-4): "))
    print("="*50)
    main_menu(choice)


