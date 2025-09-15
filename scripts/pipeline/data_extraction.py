######################################################################
# Data extraction and transformation script
######################################################################

#----------------------------------------------------------------------
# Importing the necessary packages
#----------------------------------------------------------------------
import os
import sys
import re
import numpy as np
import pandas as pd
import datetime as dt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scripts.logging.logger import logging
from scripts.exceptions.exceptions import DIMException

#----------------------------------------------------------------------
# General configurations
#----------------------------------------------------------------------
# Output File Path
OUTPUT_FILE = 'extracted_data.csv'
OUTPUT_PATH = os.path.join(os.getcwd(), 'data', OUTPUT_FILE)

# For named dimensions
width_pattern = re.compile(r'width[^a-zA-Z0-9]*(\d+(?:\.\d+)?)[\s"]*(inch|inches|in|cm|mm|feet|ft)?', re.IGNORECASE)
height_pattern = re.compile(r'height[^a-zA-Z0-9]*(\d+(?:\.\d+)?)[\s"]*(inch|inches|in|cm|mm|feet|ft)?', re.IGNORECASE)

# For general patterns like 40"x50" or 40 x 50cm
dim_pair_pattern = re.compile(r'(\d+(?:\.\d+)?)[\s"]*[x×*][\s"]*(\d+(?:\.\d+)?)(?:[\s"]*(inch|inches|in|cm|mm|feet|ft))?', re.IGNORECASE)

#----------------------------------------------------------------------
# Writing the necessary functions
#----------------------------------------------------------------------

# Function to convert all the measurements to mm
def to_mm(value, unit):
    if not unit:
        return value
    unit = unit.lower()
    if unit in ['cm']: return value * 10
    if unit in ['inch', 'inches', 'in']: return value * 25.4
    if unit in ['feet', 'ft']: return value * 304.8
    return value

# Function to append the width and height to the row
def combine_text(row):
    try:
        parts = [str(row.get('TITLE', ''))]
        if pd.notna(row.get('BULLET_POINTS')): parts.append(str(row['BULLET_POINTS']))
        if pd.notna(row.get('DESCRIPTION')): parts.append(str(row['DESCRIPTION']))
        return ' '.join(parts)
    except Exception as e:
        logging.error(f"Error combining text: {e}")
        raise DIMException(f"Error combining text: {e}", sys)

# Function to extract width and height from the combined text
def extract_width_height(text):
    try:
        # First try named patterns
        w_match = width_pattern.search(text)
        h_match = height_pattern.search(text)
    
        width = to_mm(float(w_match.group(1)), w_match.group(2)) if w_match else None
        height = to_mm(float(h_match.group(1)), h_match.group(2)) if h_match else None
    
        # If still missing, try 40x50 style pattern
        if width is None or height is None:
            pair_match = dim_pair_pattern.search(text)
            if pair_match:
                val1, val2, unit = pair_match.groups()
                val1 = to_mm(float(val1), unit)
                val2 = to_mm(float(val2), unit)
                # Assign missing values only
                if width is None: width = val1
                if height is None: height = val2
    
        return pd.Series([width if width else np.nan, height if height else np.nan])
    except Exception as e:
        logging.error(f"Error extracting width and height: {e}")
        raise DIMException(f"Error extracting width and height: {e}", sys)
    
# Function to define the packaging style
def assign_packaging_style(row):
    try:
        text = str(row.get("TITLE", "")).lower() + " " + str(row.get("DESCRIPTION", "")).lower()

        # 0: rolled, 1: disassembled, 2: box-like
        if any(word in text for word in ["thread", "yarn", "rope", "wire", "cable"]):
            return 0
        elif any(word in text for word in ["poster", "fabric", "cloth", "mat", "sheet"]):
            return 0
        elif any(word in text for word in ["sofa", "table", "chair", "bed", "furniture", "wardrobe"]):
            return 1
        else:
            return 2   # default fallback
    except Exception as e:
        logging.error(f"Error assigning packaging style: {e}")
        raise DIMException(f"Error assigning packaging style: {e}", sys)

# Function to transform the data in chunks
def transform_dimensions(chunks): 
    try: 
        # Final df to store each transformed chunks
        final_df = pd.DataFrame()

        for chunk in chunks: 
            chunk['combined_text'] = chunk.apply(combine_text, axis=1) 
            chunk[['parsed_width', 'parsed_height']] = chunk['combined_text'].apply(extract_width_height) 
            chunk.rename(columns={"PRODUCT_LENGTH": "parsed_length"}, inplace=True) 

            chunk_filtered = chunk[['PRODUCT_TYPE_ID', 'TITLE', 'DESCRIPTION', 'parsed_length', 'parsed_width', 'parsed_height']].dropna() 
            final_df = pd.concat([final_df, chunk_filtered], ignore_index=True)

        return final_df
    except Exception as e: 
        logging.error(f"Error during data transformation: {e}") 
        raise DIMException(f"Error during data transformation: {e}", sys)

# Function to count the frequerncy of packaging styles
def count_product_type_freq(df, col):
    try:
        type_freq = df[col].value_counts()
        df["TYPE_FREQ"] = df[col].map(type_freq)

        return df
    except Exception as e:
        logging.error(f"Error counting product type frequency: {e}")
        raise DIMException(f"Error counting product type frequency: {e}", sys)

# Function to scale the int and float dimensions to a 0-1 range
def scale_dimensions(df, is_train=True, scaler_path="scaler.pkl"):
    try:
        X = df[["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]]
        SCALER_PATH = os.path.join(os.getcwd(), 'models', scaler_path)

        if is_train:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, SCALER_PATH)  # save fitted scaler
            print(f"💾 Saved MinMaxScaler to {SCALER_PATH}")
        else:
            scaler = joblib.load(SCALER_PATH)
            X_scaled = scaler.transform(X)

        # Replace scaled values in dataframe
        df_scaled = df.copy()
        df_scaled[["TYPE_FREQ", "parsed_length", "parsed_width", "parsed_height", "PACKAGING_STYLE"]] = X_scaled

        return df_scaled

    except Exception as e:
        logging.error(f"Error scaling dimensions: {e}")
        raise DIMException(f"Error scaling dimensions: {e}", sys)
#----------------------------------------------------------------------
# Main function to execute the script
#----------------------------------------------------------------------
def data_transformation(DATA_FILE, is_train):
    try:
        DATA_PATH = os.path.join(os.getcwd(), 'data', DATA_FILE)

        if not os.path.exists(DATA_PATH):
            raise DIMException(f"Data file {DATA_PATH} not found", sys)

        start_time = dt.datetime.now()
        chunks = pd.read_csv(DATA_PATH, chunksize=10000)
        extract_end_time = dt.datetime.now()
        logging.info(f"Data extraction started at {start_time} and ended at {extract_end_time}. Processing chunks...")

        final_df = transform_dimensions(chunks)
        final_df["PACKAGING_STYLE"] = final_df.apply(assign_packaging_style, axis=1)
        
        final_df = count_product_type_freq(final_df, "PRODUCT_TYPE_ID")
        logging.info(f"Product type frequency counted and added to the dataframe.")
        
        # Drop the columns title and description to keep only necessary columns
        final_df = final_df[['TYPE_FREQ', 'parsed_length', 'parsed_width', 'parsed_height', 'PACKAGING_STYLE']]

        # Scale the dimensions
        final_df = scale_dimensions(final_df, is_train=is_train)
        logging.info(f"Dimensions scaled using MinMaxScaler.")

        end_time = dt.datetime.now()
        logging.info(f"Data transformation completed. Total time taken: {(end_time - extract_end_time).total_seconds():.2f} seconds")

        final_df.to_csv(OUTPUT_PATH, index=False)
        elapsed_time = dt.datetime.now() - start_time
        logging.info(f"Transformed data saved to {OUTPUT_PATH}. Elapsed time: {elapsed_time.total_seconds():.2f} seconds")
        print(f"Data transformation completed in {elapsed_time.total_seconds():.2f} seconds.")

        return final_df

    except Exception as e:
        logging.error(f"Initialization error: {e}")
        raise DIMException(f"Initialization error: {e}", sys)
