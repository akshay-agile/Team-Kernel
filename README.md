# AquaIntelligence – AI Pool Risk Detection Platform

AquaIntelligence is an **AI-powered insurance intelligence platform** that detects swimming pools from aerial images and evaluates insurance risk automatically.

The system combines **computer vision, machine learning, IoT monitoring, and fraud detection** to assist insurance companies in underwriting and claims verification.

---

## Key Features

### AI Pool Detection

* Detect swimming pools from aerial/satellite images using **YOLO object detection**
* Classify pool **structure type** (Inground / Aboveground)
* Detect **pool cover status** (Covered / Uncovered)
* Generate **annotated images with bounding boxes**
* Automatic **risk scoring based on pool characteristics**

### Risk Assessment Engine

* Property-level risk scoring based on:

  * Pool presence
  * Pool size
  * Pool cover status
  * Pool structure type
  * Number of pools
* Generates **LOW / MEDIUM / HIGH risk levels**
* Provides **explanations for risk factors**

### Time-Series Fraud Detection

* Compare **before and after satellite images**
* Detect:

  * Newly added pools
  * Removed pools
  * Unchanged pools
* Identify **possible insurance fraud cases**

### IoT Inspection System

* Accepts **ESP8266 sensor readings**
* Automatically classifies **risk level from sensor data**
* Sends **SMS alerts using Twilio** when risk is high
* Stores IoT inspection history

### Document Fraud Detection

* Upload property documents (PDF or image)
* Extract text using **Tesseract OCR**
* Detect suspicious documents using:

  * Rule-based validation
  * **Isolation Forest anomaly detection**
* Generates **fraud risk score and recommendations**

### Claims Management System

* Policyholders can **submit insurance claims**
* Insurers can:

  * View claims
  * Update claim status
  * Flag potential fraud
* Claim tracking system for policyholders

### Policyholder Registration

* Register property and declare pool information
* Store policyholder profile and pool details
* Link claims to registered policyholders

### Communication System

* Send **email notifications via SMTP**
* Send **SMS alerts via Twilio**
* Maintain communication logs

### Reporting & GeoJSON Export

* Generate **AI underwriting reports**
* Export detected pools as **GeoJSON spatial data**
* Structured JSON reports for insurance analytics

---

# Tech Stack

### Frontend

* React (Vite)
* Recharts (analytics dashboards)
* JavaScript
* HTML / CSS

### Backend

* Python
* FastAPI
* SQLAlchemy
* SQLite Database

### Machine Learning

* YOLO (pool detection)
* MobileNetV2 (pool structure & cover classification)
* Isolation Forest (fraud detection)

### Other Technologies

* OpenCV
* Tesseract OCR
* Twilio SMS API
* SMTP Email
* ESP8266 IoT integration

---

# Project Structure

```
project/
│
├── frontend/                 # React dashboard
│   ├── src/
│   ├── public/
│   └── package.json
│
├── dataset/                  # Sample images for testing
├── api.py                    # FastAPI backend
├── cover_classifier.pth      # Cover detection model
├── structure_classifier.pth  # Structure classification model
├── requirements.txt          # Python dependencies
└── README.md
```

---

# Installation

## 1. Clone the Repository

```
git clone https://github.com/your-username/aquaintelligence.git
cd aquaintelligence
```

---

# Backend Setup

### Install Python Dependencies

If `requirements.txt` is already included:

```
pip install -r requirements.txt
```

If you want to generate it from the environment:

```
pip freeze > requirements.txt
pip install -r requirements.txt
```

---

### Run the Backend API

```
uvicorn api:app --reload --port 5000
```

API documentation will be available at:

```
http://localhost:5000/docs
```

---

# Frontend Setup

```
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# System Workflow

1. User uploads satellite image.
2. AI detects swimming pools using YOLO.
3. Pools are classified by structure and cover type.
4. Risk scoring engine evaluates property risk.
5. Fraud detection compares historical images.
6. IoT sensor data provides additional inspection data.
7. Reports and alerts are generated for insurers.

---

# Sample Dataset

A small **sample dataset (25 images)** is included in the `dataset/` folder to test the detection system.

---

# Author

Akshay
