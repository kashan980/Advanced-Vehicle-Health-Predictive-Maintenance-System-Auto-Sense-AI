# AutoSense AI — Project Documentation

**Course:** Artificial Intelligence | CS6-1  
**Instructor:** Sir Junaid Khan  
**Group Members:** Kashan Shahid (53686) · Sinwan Haider (56275)  
**Institution:** Riphah International University

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Workflow](#2-system-workflow)
3. [Libraries Used](#3-libraries-used)
4. [Machine Learning Component](#4-machine-learning-component)
5. [Backend API — Functions & Inputs/Outputs](#5-backend-api--functions--inputsoutputs)
6. [Mobile Application — Functions & Inputs/Outputs](#6-mobile-application--functions--inputsoutputs)
7. [Execution Steps](#7-execution-steps)
8. [Project File Structure](#8-project-file-structure)
9. [API Reference](#9-api-reference)

---

## 1. Project Overview

AutoSense AI is a full-stack predictive vehicle health monitoring system. It reads four engine sensor values (OBD-II data), runs them through a trained machine learning model, and tells you whether the vehicle is **healthy**, **overheating**, or in a **critical fault** state — before a breakdown actually happens.

The system has three layers:

```
[ Flutter Mobile App ]  →  HTTP POST  →  [ FastAPI Server ]  →  [ Isolation Forest ML Model ]
     Android UI                              Python Backend            scikit-learn .pkl file
```

---

## 2. System Workflow

```
Step 1: User opens the Flutter app on Android
         ↓
Step 2: User selects a simulation preset OR enters custom values via sliders
         ↓
Step 3: App packages 4 sensor values as JSON and sends HTTP POST to /predict
         ↓
Step 4: FastAPI server receives the request and validates all field ranges
         ↓
Step 5: Values are scaled using the pre-fitted StandardScaler (obd_scaler.pkl)
         ↓
Step 6: Scaled values are passed to the Isolation Forest model (obd_anomaly_model.pkl)
         ↓
Step 7: Model returns:
         • prediction  →  +1 (inlier/healthy)  or  -1 (outlier/fault)
         • decision_function score  →  raw anomaly score (float)
         ↓
Step 8: API normalises the raw score → health_score [0.0 to 1.0]
         ↓
Step 9: JSON response sent back to Flutter app
         ↓
Step 10: App animates the health gauge, updates sensor cards, appends history chart
```

---

## 3. Libraries Used

### Python Backend

| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.110 | Web framework for building the REST API |
| `uvicorn` | ≥ 0.29 | ASGI server that runs the FastAPI app |
| `pydantic` | ≥ 2.0 | Data validation — ensures sensor values are within allowed ranges |
| `scikit-learn` | ≥ 1.4 | Provides the Isolation Forest algorithm and StandardScaler |
| `joblib` | ≥ 1.3 | Saves and loads the trained model and scaler as .pkl files |
| `pandas` | ≥ 2.0 | Creates the DataFrame structure that the model expects as input |
| `numpy` | ≥ 1.26 | Used for the `clip()` function in score normalisation |

### Python Training Notebook (Google Colab)

| Library | Purpose |
|---------|---------|
| `numpy` | Generating synthetic sensor data using random distributions |
| `pandas` | Creating and manipulating the training DataFrame |
| `scikit-learn` | `IsolationForest`, `StandardScaler`, `train_test_split` |
| `matplotlib` | Plotting data distributions and anomaly visualisations |
| `joblib` | Exporting the trained model and scaler as .pkl files |

### Flutter Mobile App (Dart)

| Package | Version | Purpose |
|---------|---------|---------|
| `flutter` | SDK | Core UI framework |
| `http` | ^1.2.0 | Sends HTTP POST requests to the FastAPI server |
| `dart:convert` | Built-in | Encodes/decodes JSON (jsonEncode, jsonDecode) |
| `dart:math` | Built-in | `pi` constant used in drawing the arc gauge |

---

## 4. Machine Learning Component

### How the Dataset Was Created

The training data is **synthetically generated** in the Google Colab notebook. Since labelled real-world OBD-II fault data is not publicly available, synthetic data is generated within the known normal operating ranges of a healthy petrol engine.

```python
import numpy as np
import pandas as pd

n_samples = 5000

data = pd.DataFrame({
    'Engine_RPM':    np.random.normal(loc=2000, scale=500, size=n_samples).clip(800, 3500),
    'Coolant_Temp':  np.random.normal(loc=92,   scale=8,   size=n_samples).clip(80, 105),
    'Engine_Load':   np.random.normal(loc=45,   scale=15,  size=n_samples).clip(20, 75),
    'Vehicle_Speed': np.random.normal(loc=70,   scale=30,  size=n_samples).clip(0, 180),
})
```

**Normal operating ranges used for training:**

| Sensor | Mean | Range in Training Data |
|--------|------|------------------------|
| Engine RPM | 2000 rpm | 800 – 3500 |
| Coolant Temp | 92°C | 80 – 105°C |
| Engine Load | 45% | 20 – 75% |
| Vehicle Speed | 70 km/h | 0 – 180 km/h |

### Why Isolation Forest?

Isolation Forest is an **unsupervised anomaly detection** algorithm. It does not need labelled fault examples. It works by building random decision trees and measuring how quickly each data point gets "isolated" (separated from all others).

- **Normal points** take many tree splits to isolate → low anomaly score
- **Anomalous points** are isolated in very few splits → high anomaly score

This makes it perfect when you only have healthy data to train on — which is the realistic situation for most vehicle maintenance scenarios.

### Score Normalisation Formula

The raw `decision_function` output from Isolation Forest ranges approximately from -0.3 (highly anomalous) to +0.3 (very normal). This is normalised to [0, 1]:

```
health_score = clip( (raw_score + 0.3) / 0.6,  min=0.0,  max=1.0 )
anomaly_score = 1 - health_score
```

### Model Files Produced

| File | Description |
|------|-------------|
| `obd_anomaly_model.pkl` | The trained Isolation Forest model |
| `obd_scaler.pkl` | The fitted StandardScaler (must match training data) |

---

## 5. Backend API — Functions & Inputs/Outputs

**File:** `Vehicle_Health_API/main.py`  
**Run with:** `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

---

### `read_root()`

| Item | Detail |
|------|--------|
| **Endpoint** | `GET /` |
| **Purpose** | Health check — confirms the server is running and model is loaded |
| **Input** | None |
| **Output** | `{ "service": "AutoSense AI", "version": "2.0.0", "status": "running", "model_loaded": true }` |
| **When to use** | Open `http://localhost:8000` in a browser to verify the server is running |

---

### `predict_health(data: VehicleData)`

| Item | Detail |
|------|--------|
| **Endpoint** | `POST /predict` |
| **Purpose** | Core inference function — runs the ML model on sensor values |
| **Input** | JSON body with 4 sensor fields (see below) |
| **Output** | JSON with status, health score, anomaly score (see below) |

**Input fields (VehicleData Pydantic model):**

| Field | Type | Min | Max | Unit | Example |
|-------|------|-----|-----|------|---------|
| `Engine_RPM` | float | 0 | 10000 | rpm | 2100.5 |
| `Coolant_Temp` | float | 0 | 200 | °C | 92.0 |
| `Engine_Load` | float | 0 | 100 | % | 35.5 |
| `Vehicle_Speed` | float | 0 | 300 | km/h | 65.0 |

**Output fields (PredictionResponse):**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `status` | string | Human-readable health status | "Engine Operating Normally" |
| `requires_maintenance` | bool | True if fault detected | false |
| `anomaly_score` | float | 0–1, higher = more anomalous | 0.08 |
| `health_score` | float | 0–1, higher = healthier | 0.92 |
| `received_data` | dict | Echo of input values | {...} |
| `thresholds` | dict | Normal operating ranges | {...} |

**Internal logic step by step:**

```
1. Pydantic validates all 4 input fields and rejects out-of-range values
2. pd.DataFrame created from the 4 sensor values with correct column names
3. scaler.transform() applied — same scaler fitted during training
4. model.predict() → returns +1 (healthy) or -1 (fault)
5. model.decision_function() → returns raw float score
6. health_score = clip((raw + 0.3) / 0.6, 0, 1)
7. Status string selected based on health_score and prediction
8. PredictionResponse object returned as JSON
```

**Status logic:**

```python
if prediction == -1:        → "CRITICAL FAULT DETECTED"
elif health_score < 0.7:    → "Warning — Degraded Performance"
else:                       → "Engine Operating Normally"
```

---

### Error Handling

| Situation | HTTP Code | Response |
|-----------|-----------|----------|
| Model files not found at startup | 503 | "Model not loaded. Check server logs." |
| Invalid sensor value (out of range) | 422 | Pydantic validation error detail |
| Any inference exception | 500 | "Inference error: {message}" |

---

## 6. Mobile Application — Functions & Inputs/Outputs

**File:** `vehicle_health_app/lib/main.dart`  
**Platform:** Android (Flutter 3 + Dart)

---

### `VehicleApiService.predict(SensorReading reading)`

| Item | Detail |
|------|--------|
| **Purpose** | Sends sensor data to the FastAPI server and returns a parsed result |
| **Input** | `SensorReading` object with 4 double fields |
| **Output** | `PredictionResult` object (status, message, detail, anomalyScore) |
| **Timeout** | 10 seconds |
| **Error** | Throws `Exception` if non-200 response received |
| **URL** | `http://10.0.2.2:8000/predict` (emulator) |

> **Note:** `10.0.2.2` is Android emulator's alias for `localhost`. For a real device on the same WiFi, replace with your PC's LAN IP (e.g. `192.168.1.5`).

---

### `_DashboardScreenState._runPrediction(SensorReading reading)`

| Item | Detail |
|------|--------|
| **Purpose** | Orchestrates the full API call cycle and updates all UI state |
| **Input** | `SensorReading` object |
| **Output** | Updates `_status`, `_healthScore`, `_history`, `_statusMessage`, `_statusDetail` |
| **Side effects** | Resets and replays score animation; appends to history list |

---

### `_GaugePainter.paint(Canvas canvas, Size size)`

| Item | Detail |
|------|--------|
| **Purpose** | Draws the arc health gauge using Flutter's Canvas API |
| **Input** | `progress` (0.0–1.0), `color`, `backgroundColor` |
| **Output** | Renders two arcs on canvas — background ring + coloured foreground ring |
| **Algorithm** | Draws arc from -135° sweeping 270° total; foreground sweeps `270° × progress` |

---

### `_ChartPainter.paint(Canvas canvas, Size size)`

| Item | Detail |
|------|--------|
| **Purpose** | Draws the health history line chart |
| **Input** | `List<double>` of health scores (0.0–1.0), up to 12 values |
| **Output** | Cubic Bezier smooth line + gradient fill beneath it |
| **Algorithm** | Spaced evenly on X axis; Y = `height - (score × height)`; control points at midpoints |

---

### `ManualInputSheet` (Bottom Sheet Widget)

| Item | Detail |
|------|--------|
| **Purpose** | Lets user manually set all 4 sensor values using sliders |
| **Input** | Initial values from current dashboard state |
| **Output** | Calls `onSubmit(SensorReading)` with new values when button tapped |
| **Sliders** | RPM: 500–8000 · Temp: 50–140 · Load: 0–100 · Speed: 0–220 |

---

### Simulation Presets

| Preset | RPM | Temp | Load | Speed | Expected Result |
|--------|-----|------|------|-------|-----------------|
| Healthy Engine | 2100 | 92°C | 35% | 65 km/h | Green gauge, ~90% health |
| Engine Overheating | 3800 | 112°C | 78% | 30 km/h | Amber warning |
| Critical Fault | 6500 | 125°C | 95% | 0 km/h | Red gauge, fault detected |

---

## 7. Execution Steps

### Step 1 — Train the Model (Google Colab)

```
1. Open python_proj_ai_.ipynb in Google Colab
2. Run all cells in order
3. The notebook will generate:
     • obd_anomaly_model.pkl
     • obd_scaler.pkl
4. Download both files to your PC
5. Place them inside the Vehicle_Health_API/ folder
```

### Step 2 — Start the Python Backend

```bash
# Navigate to the API folder
cd Vehicle_Health_API

# Install required packages
pip install fastapi uvicorn scikit-learn joblib pandas numpy

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Verify it works — open in browser:
# http://localhost:8000
# http://localhost:8000/docs   ← Auto-generated API documentation
```

### Step 3 — Run the Flutter App

```bash
# Navigate to Flutter project
cd vehicle_health_app

# Install Flutter dependencies
flutter pub get

# Make sure an Android emulator is running in Android Studio
# Then launch the app
flutter run

# OR open Android Studio → Open the project → Run (green play button)
```

### Step 4 — Test the System

```
1. App opens with the AutoSense AI splash screen
2. Dashboard loads showing the health gauge
3. Tap any simulation preset button:
   • "Healthy Engine" → gauge should show green, ~90%
   • "Engine Overheating" → gauge should show amber
   • "Critical Fault" → gauge should show red, fault detected
4. Tap the "Manual Input" FAB → adjust sliders → tap "Run AI Analysis"
5. Pull down on the dashboard to refresh with last values
```

---

## 8. Project File Structure

```
AutoSense-AI/
│
├── README.md                          ← GitHub repository overview
├── DOCUMENTATION.md                   ← This file
│
├── Vehicle_Health_API/                ← Python backend
│   ├── main.py                        ← FastAPI server (core logic)
│   ├── obd_anomaly_model.pkl          ← Trained Isolation Forest
│   ├── obd_scaler.pkl                 ← Fitted StandardScaler
│   └── requirements.txt               ← Python dependencies
│
├── python_proj_ai_.ipynb              ← Google Colab training notebook
│                                         (dataset generation + model training)
│
└── vehicle_health_app/                ← Flutter Android app
    ├── lib/
    │   └── main.dart                  ← Complete Flutter UI + API client
    ├── pubspec.yaml                   ← Flutter dependencies
    └── android/                       ← Android build configuration
```

---

## 9. API Reference

### Test the API manually (without the app)

Once the server is running, open `http://localhost:8000/docs` in your browser. FastAPI auto-generates a Swagger UI where you can test the `/predict` endpoint directly.

**Example curl request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Engine_RPM": 2100.5,
           "Coolant_Temp": 92.0,
           "Engine_Load": 35.5,
           "Vehicle_Speed": 65.0
         }'
```

**Example response (healthy):**

```json
{
  "status": "Engine Operating Normally",
  "requires_maintenance": false,
  "anomaly_score": 0.08,
  "health_score": 0.92,
  "received_data": {
    "Engine_RPM": 2100.5,
    "Coolant_Temp": 92.0,
    "Engine_Load": 35.5,
    "Vehicle_Speed": 65.0
  },
  "thresholds": {
    "RPM_normal": "800–3500",
    "Coolant_normal": "80–105°C",
    "Load_normal": "20–75%"
  }
}
```

**Example response (critical fault):**

```json
{
  "status": "CRITICAL FAULT DETECTED",
  "requires_maintenance": true,
  "anomaly_score": 0.89,
  "health_score": 0.11,
  "received_data": {
    "Engine_RPM": 6500.0,
    "Coolant_Temp": 125.0,
    "Engine_Load": 95.0,
    "Vehicle_Speed": 0.0
  },
  "thresholds": {
    "RPM_normal": "800–3500",
    "Coolant_normal": "80–105°C",
    "Load_normal": "20–75%"
  }
}
```

---

*AutoSense AI — Riphah International University | CS6-1 | 2025*
