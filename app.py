# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# ========================
# Flask App Initialization
# ========================
app = Flask(__name__)
CORS(app)

# ========================
# File Paths
# ========================
MODEL_DIR = "models"
CROP_MODEL_FILE = os.path.join(MODEL_DIR, "crop_model.pkl")
REG_MODEL_FILE = os.path.join(MODEL_DIR, "regression_model.pkl")
GROWTH_STAGES_FILE = os.path.join(MODEL_DIR, "growth_stages.pkl")
WS_COLS_FILE = os.path.join(MODEL_DIR, "ws_cols.pkl")

# ========================
# Load Models & Metadata
# ========================
crop_model = joblib.load(CROP_MODEL_FILE)
reg_model = joblib.load(REG_MODEL_FILE)
growth_stages = joblib.load(GROWTH_STAGES_FILE)

# Load water sources list (fallback if missing)
if os.path.exists(WS_COLS_FILE):
    ws_cols = joblib.load(WS_COLS_FILE)
else:
    ws_cols = ["Rainfed", "Canal", "Borewell", "River", "Pond", "Irrigated"]

# ========================
# Frontend → Backend Key Mapping
# ========================
KEY_MAP = {
    "areaType": "Area Type",
    "soilType": "Soil Type",
    "waterSources": "Water Sources",
    "croppingType": "Cropping Type",
    "cropCategory": "Crop Category",
    "previousCrop": "Previous Crop",
    "landAcres": "Land Acres",
}

# ========================
# Helper Functions
# ========================
def preprocess_request(data):
    """
    Convert incoming JSON request into model-ready DataFrame.
    Handles one-hot encoding and water source binary features.
    """
    # Normalize keys
    normalized = {KEY_MAP.get(k, k): v for k, v in data.items()}

    # Extract main categorical features
    input_dict = {
        "Soil Type": normalized.get("Soil Type"),
        "Area Type": normalized.get("Area Type"),
        "Cropping Type": normalized.get("Cropping Type"),
        "Crop Category": normalized.get("Crop Category"),
    }
    df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    df = pd.get_dummies(df)

    # Ensure missing columns from training are added
    for col in crop_model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Encode water sources
    ws_values = normalized.get("Water Sources", [])
    if isinstance(ws_values, str):
        ws_values = [ws_values]
    for src in ws_cols:
        df[f"water_{src}"] = 1 if src in ws_values else 0

    # Ensure column order matches training
    df = df[crop_model.feature_names_in_]
    return df

def calculate_roi(profit, investment):
    """Calculate ROI percentage safely."""
    if investment > 0:
        return f"{round((profit / investment) * 100, 2)}%"
    return "N/A"

# ========================
# Flask Routes
# ========================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Preprocess input
        df_input = preprocess_request(data)

        # Crop prediction
        predicted_crop = crop_model.predict(df_input)[0]

        # Numeric predictions
        duration, investment, profit = reg_model.predict(df_input)[0]

        # Fetch growth stages
        stages = growth_stages.get(predicted_crop, ["Stage1", "Stage2", "Harvest"])

        # Prepare response
        response = {
            "recommendedCrop": predicted_crop,
            "cropDurationDays": int(duration),
            "investment": round(float(investment), 2),
            "profit": round(float(profit), 2),
            "roi": calculate_roi(profit, investment),
            "growthStages": stages,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# Main Entry
# ========================
if __name__ == "__main__":
    app.run(debug=True)
