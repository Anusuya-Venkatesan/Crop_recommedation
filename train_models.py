# train_models.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

# ========================
# File Paths
# ========================
DATA_FILE = "data/crop_data.csv"
MODEL_DIR = "models"
CROP_MODEL_FILE = os.path.join(MODEL_DIR, "crop_model.pkl")
REG_MODEL_FILE = os.path.join(MODEL_DIR, "regression_model.pkl")
WS_COLS_FILE = os.path.join(MODEL_DIR, "ws_cols.pkl")
GROWTH_STAGES_FILE = os.path.join(MODEL_DIR, "growth_stages.pkl")

# ========================
# Utility Functions
# ========================
def ensure_model_dir_exists():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def preprocess_water_sources(ws_series):
    all_sources = set()
    ws_list = []

    for val in ws_series:
        items = str(val).split('|')
        ws_list.append(items)
        all_sources.update(items)

    all_sources = list(all_sources)

    ws_encoded = []
    for items in ws_list:
        ws_encoded.append([1 if src in items else 0 for src in all_sources])

    ws_encoded_df = pd.DataFrame(ws_encoded, columns=[f'water_{src}' for src in all_sources])
    return ws_encoded_df, all_sources

# ========================
# Main Training Function
# ========================
def train_models():
    ensure_model_dir_exists()

    # Load dataset
    data = pd.read_csv(DATA_FILE, header=0)

    # Convert numeric columns to numeric types
    numeric_cols = ['Duration (Days)', 'Investment (₹)', 'Profit (₹)']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing **required columns**
    required_cols = ['Soil Type', 'Area Type', 'Cropping Type', 'Crop Category', 'Crop'] + numeric_cols
    data = data.dropna(subset=required_cols)

    # Reset index
    data = data.reset_index(drop=True)

    # Input features and targets
    X = data[['Soil Type', 'Area Type', 'Water Sources', 'Cropping Type', 'Crop Category']]
    y_crop = data['Crop']
    y_numeric = data[numeric_cols]

    # Preprocess water sources
    ws_df, ws_cols = preprocess_water_sources(X['Water Sources'])
    X_encoded = pd.get_dummies(X.drop('Water Sources', axis=1))
    X_encoded = pd.concat([X_encoded, ws_df], axis=1)

    # Reset index to ensure same length
    X_encoded = X_encoded.reset_index(drop=True)
    y_crop = y_crop.reset_index(drop=True)
    y_numeric = y_numeric.reset_index(drop=True)

    # -------------------
    # Crop Classification Model
    # -------------------
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_encoded, y_crop)
    joblib.dump(crop_model, CROP_MODEL_FILE)

    # Save water sources
    joblib.dump(ws_cols, WS_COLS_FILE)

    # -------------------
    # Numeric Regression Model
    # -------------------
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_encoded, y_numeric)
    joblib.dump(reg_model, REG_MODEL_FILE)

    # -------------------
    # Growth Stages
    # -------------------
    growth_stages = {
        "Tomato": ["Sowing", "Germination", "Vegetative", "Flowering", "Fruiting", "Harvest"],
        "Wheat": ["Sowing", "Tillering", "Stem Elongation", "Heading", "Maturity", "Harvest"],
        "Mango": ["Flowering", "Fruit Set", "Fruit Development", "Maturity", "Harvest"],
        "Pulses": ["Sowing", "Germination", "Vegetative", "Flowering", "Harvest"],
        "Rice": ["Sowing", "Transplanting", "Vegetative", "Reproductive", "Maturity", "Harvest"],
        "Maize": ["Sowing", "Vegetative", "Tasseling", "Silking", "Maturity", "Harvest"],
        "Sugarcane": ["Planting", "Tillering", "Grand Growth", "Maturity", "Harvest"],
        "Banana": ["Planting", "Vegetative", "Flowering", "Fruiting", "Harvest"],
    }
    joblib.dump(growth_stages, GROWTH_STAGES_FILE)

    print("✅ Models trained and saved successfully!")

# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    train_models()
