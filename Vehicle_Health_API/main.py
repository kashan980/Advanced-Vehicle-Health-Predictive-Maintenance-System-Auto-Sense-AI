from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
#  App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="AutoSense AI — Vehicle Health API",
    description="Predictive maintenance using an Isolation Forest anomaly detection model.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  Load Model & Scaler
# ──────────────────────────────────────────────
try:
    model = joblib.load("obd_anomaly_model.pkl")
    scaler = joblib.load("obd_scaler.pkl")
    print("✅ Isolation Forest model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler = None, None


# ──────────────────────────────────────────────
#  Schemas
# ──────────────────────────────────────────────
class VehicleData(BaseModel):
    Engine_RPM: float = Field(..., ge=0, le=10000, description="Engine rotational speed in RPM")
    Coolant_Temp: float = Field(..., ge=0, le=200, description="Coolant temperature in °C")
    Engine_Load: float = Field(..., ge=0, le=100, description="Engine load percentage (0–100)")
    Vehicle_Speed: float = Field(..., ge=0, le=300, description="Vehicle speed in km/h")


class PredictionResponse(BaseModel):
    status: str
    requires_maintenance: bool
    anomaly_score: float
    health_score: float
    received_data: dict
    thresholds: dict


# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────
@app.get("/", tags=["Health Check"])
def read_root():
    return {
        "service": "AutoSense AI — Vehicle Health API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_health(data: VehicleData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        feature_names = ["Engine_RPM", "Coolant_Temp", "Engine_Load", "Vehicle_Speed"]
        input_df = pd.DataFrame([data.dict()])[feature_names]

        # Scale using the same scaler fitted during training
        scaled_data = scaler.transform(input_df)

        # Isolation Forest: 1 = inlier (healthy), -1 = outlier (fault)
        prediction = model.predict(scaled_data)[0]

        # decision_function gives a raw anomaly score
        # More negative = more anomalous. We normalise to [0, 1].
        raw_score = model.decision_function(scaled_data)[0]

        # Estimate a health score: map the decision_function to [0, 1]
        # Typical range is roughly [-0.3, 0.3]; we clamp and normalise.
        normalised = float(np.clip((raw_score + 0.3) / 0.6, 0.0, 1.0))
        health_score = round(normalised, 4)
        anomaly_score = round(1.0 - normalised, 4)

        is_fault = prediction == -1

        if is_fault:
            status = "CRITICAL FAULT DETECTED"
        elif health_score < 0.7:
            status = "Warning — Degraded Performance"
        else:
            status = "Engine Operating Normally"

        return PredictionResponse(
            status=status,
            requires_maintenance=is_fault,
            anomaly_score=anomaly_score,
            health_score=health_score,
            received_data=data.dict(),
            thresholds={
                "RPM_normal": "800–3500",
                "Coolant_normal": "80–105°C",
                "Load_normal": "20–75%",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")