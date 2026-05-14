# 🚗 AutoSense AI — Predictive Vehicle Health Monitoring

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-3.x-blue?logo=flutter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> AI-powered predictive maintenance system that monitors vehicle engine health in real-time using an Isolation Forest anomaly detection model, served through a FastAPI backend and visualised in a Flutter Android application.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Team](#team)

---

## About the Project

AutoSense AI detects vehicle engine anomalies **before** they become critical failures. It reads four OBD-II sensor parameters and uses an unsupervised machine learning model to classify the vehicle's health state as:

- ✅ **Healthy** — all parameters within normal operating range
- ⚠️ **Warning** — degraded performance detected
- 🔴 **Critical Fault** — anomaly detected, maintenance required

This project was built as a semester project for the **Artificial Intelligence** course at Riphah International University.

---

## System Architecture

```
┌─────────────────────┐         HTTP POST          ┌──────────────────────┐
│   Flutter App       │  ──────────────────────►  │   FastAPI Server     │
│   (Android)         │  ◄──────────────────────   │   (Python)           │
│                     │         JSON Response       │                      │
│  • Health Gauge     │                             │  • Input Validation  │
│  • Sensor Cards     │                             │  • StandardScaler    │
│  • History Chart    │                             │  • Isolation Forest  │
│  • Sim Presets      │                             │  • Score Normalise   │
└─────────────────────┘                             └──────────────────────┘
                                                              │
                                                    ┌─────────▼────────────┐
                                                    │   ML Model Files     │
                                                    │  • .pkl model        │
                                                    │  • .pkl scaler       │
                                                    └──────────────────────┘
```

---

## Features

- 🎯 **Real-time anomaly detection** using Isolation Forest (unsupervised ML)
- 📊 **Animated health gauge** — arc ring that animates between readings
- 🟢 **Live sensor cards** — green/red status indicator per sensor
- 📈 **Health history chart** — rolling line chart of last 12 readings
- 🎮 **3 simulation presets** — Healthy / Overheating / Critical Fault
- 🎛️ **Manual input sliders** — set any arbitrary sensor values
- ⚡ **FastAPI backend** — ~12ms average inference latency
- 🌙 **Dark UI** — professional instrument-cluster aesthetic

---

## Dataset

The model is trained on **synthetically generated OBD-II data** representing a healthy engine under various driving conditions (idle, city, highway, acceleration).

| Sensor | Unit | Normal Range | Training Distribution |
|--------|------|-------------|----------------------|
| Engine RPM | rpm | 800 – 3,500 | Normal(μ=2000, σ=500) |
| Coolant Temperature | °C | 80 – 105 | Normal(μ=92, σ=8) |
| Engine Load | % | 20 – 75 | Normal(μ=45, σ=15) |
| Vehicle Speed | km/h | 0 – 180 | Normal(μ=70, σ=30) |

**Total samples:** 5,000 healthy readings  
**Reason for synthetic data:** Labelled real-world OBD-II fault datasets are proprietary and not publicly available. Training only on healthy data and detecting deviations is standard practice for unsupervised vehicle anomaly detection.

The full data generation and training process is in [`python_proj_ai_.ipynb`](./python_proj_ai_.ipynb).

---

## Machine Learning Model

**Algorithm:** Isolation Forest (scikit-learn)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 200 | More trees = more stable score |
| `contamination` | 0.05 | Assumes ~5% real-world anomaly rate |
| `random_state` | 42 | Reproducibility |

**Score normalisation:**
```
health_score = clip( (decision_function + 0.3) / 0.6,  0.0,  1.0 )
anomaly_score = 1 - health_score
```

**Test results:**

| Scenario | Samples | Correct | Accuracy |
|----------|---------|---------|----------|
| Healthy readings | 500 | 487 | 97.4% |
| Overheating fault | 25 | 23 | 92.0% |
| Critical fault | 25 | 25 | 100% |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training | Python · NumPy · pandas · scikit-learn · Google Colab |
| Backend API | FastAPI · Uvicorn · Pydantic · joblib |
| Mobile App | Flutter 3 · Dart · http package |
| Data Persistence | joblib .pkl serialisation |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Flutter SDK 3.x
- Android Studio (with emulator) or a physical Android device
- Google Colab (for training) or the pre-trained .pkl files

---

### Step 1 — Train the Model

Open [`python_proj_ai_.ipynb`](./python_proj_ai_.ipynb) in Google Colab, run all cells, then download:
- `obd_anomaly_model.pkl`
- `obd_scaler.pkl`

Place both files inside `Vehicle_Health_API/`.

---

### Step 2 — Start the Backend

```bash
cd Vehicle_Health_API

# Install dependencies
pip install fastapi uvicorn scikit-learn joblib pandas numpy

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify at: [http://localhost:8000](http://localhost:8000)  
Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Step 3 — Run the Flutter App

```bash
cd vehicle_health_app

# Install Flutter packages
flutter pub get

# Run on Android emulator (make sure one is running in Android Studio)
flutter run
```

> **Real device:** Replace `10.0.2.2` with your PC's local IP address in `lib/main.dart` line:
> ```dart
> static const String _baseUrl = 'http://YOUR_PC_IP:8000';
> ```

---

## Project Structure

```
AutoSense-AI/
│
├── README.md                          ← You are here
├── DOCUMENTATION.md                   ← Full technical documentation
│
├── python_proj_ai_.ipynb              ← Dataset generation + model training
│
├── Vehicle_Health_API/                ← Python FastAPI backend
│   ├── main.py                        ← API server + inference logic
│   ├── obd_anomaly_model.pkl          ← Trained Isolation Forest model
│   ├── obd_scaler.pkl                 ← Fitted StandardScaler
│   └── requirements.txt               ← pip dependencies
│
└── vehicle_health_app/                ← Flutter Android application
    ├── lib/
    │   └── main.dart                  ← Full app (UI + API client)
    ├── pubspec.yaml                   ← Flutter dependencies
    └── android/                       ← Android configuration
```

---

## API Endpoints

### `GET /`
Returns server status.

```json
{
  "service": "AutoSense AI — Vehicle Health API",
  "version": "2.0.0",
  "status": "running",
  "model_loaded": true
}
```

### `POST /predict`

**Request:**
```json
{
  "Engine_RPM": 2100.5,
  "Coolant_Temp": 92.0,
  "Engine_Load": 35.5,
  "Vehicle_Speed": 65.0
}
```

**Response (healthy):**
```json
{
  "status": "Engine Operating Normally",
  "requires_maintenance": false,
  "anomaly_score": 0.08,
  "health_score": 0.92
}
```

**Response (fault):**
```json
{
  "status": "CRITICAL FAULT DETECTED",
  "requires_maintenance": true,
  "anomaly_score": 0.89,
  "health_score": 0.11
}
```

---

## Screenshots

> App running on Android emulator showing three states:

| Healthy | Overheating | Critical Fault |
|---------|-------------|----------------|
| Green gauge ~92% | Amber gauge ~52% | Red gauge ~12% |


---

## Team

| Name | Roll Number | Contribution |
|------|-------------|-------------|
| Kashan Shahid | 53686 | ML model training, FastAPI backend, Flutter UI |
| Sinwan Haider | 56275 | Data generation, testing, documentation |

**Course:** Artificial Intelligence — CS6-1  
**Instructor:** Sir Junaid Khan  
**Institution:** Riphah International University  
**Year:** 2025

---
