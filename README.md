# Dim-Predictor

### 1. Introduction
This probleem was a part of the Amazon ML Challenge. The task is to predict the length of cardboard boxes using available product metadata such as title, bullet points, description, product type, and other attributes.

The core challenge was to extract dimensions from unstructured product descriptions, process them consistently, and then train robust models that generalize well to unseen products.

This repository implements a complete pipeline:
* Data extraction & transformation
* Feature engineering
* Scaling & preprocessing
* Model training & hyperparameter tuning
* Saving/loading the best model
* Predictions on new data

### 2. Features
* End-to-end ML pipeline with ETL + ML + Prediction.
* Extracts length, width, height from messy product descriptions.
* Uses packaging style categorization as an additional feature.
* Applies Min-Max Scaling (fit on train, transform on test).
* Supports multiple models with RandomizedSearchCV for hyperparameter tuning.
* Automatically saves the best model and reuses it for predictions.
* Provides a menu-driven CLI interface for easy usage.
* Outputs predictions with product IDs in clean CSV format.

### 3. Steps to run
**Clone the repository and setup:**
* git clone https://github.com/sourav200199/Dim-Predictor.git
* cd Dim-Predictor/Dim-Predictor
* python -m venv venv
* source venv/bin/activate   # Linux/Mac
* venv\Scripts\activate      # Windows
* pip install -r requirements.txt

**Run the Pipeline:**
* python main.py

### 4. Future Improvements
* Add support for predicting width & height in addition to length.
* Experiment with deep learning models (e.g., LSTMs/BERT for text features).
* Deploy as a Flask/FastAPI service for real-time predictions.
* Implement a Dockerized setup for easier reproducibility.