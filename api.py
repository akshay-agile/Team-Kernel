"""
AquaIntelligence — Complete Backend API (v3.1)
===============================================
Fixes applied vs v3.0:
  - PyTorch 2.6 compatibility: torch.load patched to use weights_only=False
    (safe — weights come from your own training run)
  - SQLAlchemy 2.0 compatibility: declarative_base imported from sqlalchemy.orm

Run:
    uvicorn api:app --reload --port 5000

Docs:
    http://localhost:5000/docs

Endpoints:
    GET  /                              Health check + model status
    GET  /api/overview                  Dashboard summary stats

    POST /api/detect                    Single image → pool detection
    POST /api/compare                   T1 + T2 → time-series fraud detection

    POST /api/underwriting/validate     Declaration vs detection cross-check

    GET  /api/claims                    List all claims  (insurer)
    POST /api/claims                    Submit new claim (policyholder)
    GET  /api/claims/{id}               Get single claim
    PUT  /api/claims/{id}/status        Update claim status (insurer)

    POST /api/iot/reading               ESP8266 sensor reading
    GET  /api/iot/readings              Sensor history
    GET  /api/iot/latest                Latest reading

    POST /api/documents/analyze         OCR + ML fraud analysis
    GET  /api/documents/{claim_id}      Documents for a claim

    POST /api/policyholder/register     Register + pool declaration
    GET  /api/policyholder/{id}         Get policyholder profile
    PUT  /api/policyholder/{id}         Update profile

    POST /api/comms/email               Send email to policyholder
    POST /api/comms/sms                 Send Twilio SMS
    GET  /api/comms/log                 Communications log

    POST /api/report/generate           Generate full JSON underwriting report
    POST /api/geojson/export            Export detected pools as GeoJSON
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import cv2
import math
import json
import base64
import uuid
import smtplib
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI + PYDANTIC + SQLALCHEMY
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, Text, DateTime
)
# FIX: use sqlalchemy.orm.declarative_base (SQLAlchemy 2.0 compatible)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL HEAVY IMPORTS  (graceful fallback when weights are absent)
# ─────────────────────────────────────────────────────────────────────────────
try:
    # FIX: Patch torch.load BEFORE ultralytics is imported so that the YOLO
    # checkpoint (which contains custom ultralytics globals) can be loaded with
    # PyTorch 2.6+.  This is safe because the weights come from your own
    # training run.  Do NOT apply this patch to untrusted checkpoints.
    import torch

    _original_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _patched_torch_load
    # ── now it is safe to import ultralytics ──────────────────────────────────
    from ultralytics import YOLO
    import torch.nn as nn
    from torchvision import transforms, models as tv_models

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARN] PyTorch / Ultralytics not installed — demo fallback active")

try:
    import pytesseract
    from PIL import Image as PILImage
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] Tesseract / pdf2image not installed — OCR disabled")

try:
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] scikit-learn not installed — ML anomaly detection disabled")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARN] twilio not installed — SMS will be logged as QUEUED")


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (all values readable from environment variables)
# ═════════════════════════════════════════════════════════════════════════════
MODEL_PATH                = os.getenv("YOLO_MODEL_PATH",      "runs/detect/train/weights/best.pt")
COVER_MODEL_PATH          = os.getenv("COVER_MODEL_PATH",     "cover_classifier.pth")
STRUCTURE_MODEL_PATH      = os.getenv("STRUCTURE_MODEL_PATH", "structure_classifier.pth")
CONF_THRESHOLD            = float(os.getenv("CONF_THRESHOLD", "0.4"))
DIST_THRESHOLD_PROPERTY   = int(os.getenv("DIST_PROPERTY",    "150"))
DIST_THRESHOLD_TIMESERIES = int(os.getenv("DIST_TIMESERIES",  "120"))
IOU_THRESHOLD             = float(os.getenv("IOU_THRESHOLD",  "0.5"))

TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID",  "")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",   "")
TWILIO_FROM   = os.getenv("TWILIO_FROM_NUMBER",  "")
INSURER_PHONE = os.getenv("INSURER_PHONE",        "")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aquaintelligence.db")
UPLOAD_DIR   = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DATABASE  (SQLAlchemy + SQLite by default, swap DATABASE_URL for Postgres)
# ═════════════════════════════════════════════════════════════════════════════
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


class ClaimDB(Base):
    __tablename__ = "claims"
    id              = Column(String,  primary_key=True,
                              default=lambda: f"CLM-{str(uuid.uuid4())[:8].upper()}")
    policyholder_id = Column(String,  nullable=True)
    name            = Column(String)
    email           = Column(String,  nullable=True)
    phone           = Column(String,  nullable=True)
    property_addr   = Column(String)
    claim_type      = Column(String)
    incident_date   = Column(String)
    estimated_loss  = Column(Float,   default=0)
    description     = Column(Text)
    pool_declared   = Column(Boolean, default=False)
    witness_name    = Column(String,  nullable=True)
    witness_contact = Column(String,  nullable=True)
    status          = Column(String,  default="PENDING")
    risk_level      = Column(String,  default="UNKNOWN")
    risk_score      = Column(Float,   default=0)
    fraud_flag      = Column(Boolean, default=False)
    uw_notes        = Column(Text,    nullable=True)
    submitted_at    = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow,
                              onupdate=datetime.utcnow)


class PolicyholderDB(Base):
    __tablename__ = "policyholders"
    id              = Column(String,  primary_key=True,
                              default=lambda: f"PH-{str(uuid.uuid4())[:8].upper()}")
    name            = Column(String)
    email           = Column(String)
    phone           = Column(String)
    policy_number   = Column(String)
    property_addr   = Column(String)
    pin_code        = Column(String,  nullable=True)
    pool_declared   = Column(Boolean, default=False)
    pool_type       = Column(String,  nullable=True)
    pool_covered    = Column(Boolean, nullable=True)
    pool_year       = Column(String,  nullable=True)
    pool_area_sqft  = Column(Float,   nullable=True)
    pool_notes      = Column(Text,    nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class IoTReadingDB(Base):
    __tablename__ = "iot_readings"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    sensor_value   = Column(Integer)
    risk_level     = Column(String)
    inspector_name = Column(String,  nullable=True)
    notes          = Column(Text,    nullable=True)
    claim_id       = Column(String,  nullable=True)
    property_addr  = Column(String,  nullable=True)
    sms_sent       = Column(Boolean, default=False)
    recorded_at    = Column(DateTime, default=datetime.utcnow)


class DocumentDB(Base):
    __tablename__ = "documents"
    id              = Column(String,  primary_key=True,
                              default=lambda: str(uuid.uuid4()))
    claim_id        = Column(String,  nullable=True)
    policyholder_id = Column(String,  nullable=True)
    filename        = Column(String)
    doc_type        = Column(String,  default="UNKNOWN")
    cert_number     = Column(String,  nullable=True)
    owner_name      = Column(String,  nullable=True)
    risk_score      = Column(Float,   default=0)
    risk_level      = Column(String,  default="UNKNOWN")
    ml_flag         = Column(String,  default="UNKNOWN")
    recommendation  = Column(String,  nullable=True)
    risk_reasons    = Column(Text,    nullable=True)
    status          = Column(String,  default="UNDER_REVIEW")
    uploaded_at     = Column(DateTime, default=datetime.utcnow)


class CommLogDB(Base):
    __tablename__ = "comm_log"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    comm_type = Column(String)          # EMAIL | SMS
    recipient = Column(String)
    subject   = Column(String,  nullable=True)
    body      = Column(Text)
    status    = Column(String,  default="SENT")
    claim_id  = Column(String,  nullable=True)
    sent_at   = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════
device           = None
yolo_model       = None
cover_model      = None
structure_model  = None
cover_transform  = None
structure_transform = None
if_model         = None

if MODELS_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        print(f"[INFO] YOLO model loaded: {MODEL_PATH}")
    else:
        print(f"[WARN] YOLO weights not found at '{MODEL_PATH}' — demo fallback active")

    def _load_mobilenet(path: str):
        m = tv_models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            print(f"[INFO] Classifier loaded: {path}")
        else:
            print(f"[WARN] Classifier weights not found at '{path}' — random weights used")
        return m.to(device).eval()

    cover_model     = _load_mobilenet(COVER_MODEL_PATH)
    structure_model = _load_mobilenet(STRUCTURE_MODEL_PATH)

    _tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    cover_transform     = _tfm
    structure_transform = _tfm

if ML_AVAILABLE:
    # Train Isolation Forest on representative stamp-duty / cert data
    if_model = IsolationForest(contamination=0.2, random_state=42)
    _training_data = np.array([
        [20, 15, 6, 2022],
        [50, 18, 8, 2021],
        [100, 20, 10, 2020],
        [30, 16, 7, 2023],
        [500, 0, 0, 1990],   # anomaly example
    ])
    if_model.fit(_training_data)
    print("[INFO] Isolation Forest trained")


# ═════════════════════════════════════════════════════════════════════════════
#  CORE DETECTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _detect_cover(crop: np.ndarray):
    """Classify pool crop as Covered / Uncovered using MobileNetV2."""
    if not MODELS_AVAILABLE or cover_model is None or crop is None or crop.size == 0:
        return "Uncovered", 0.75
    tensor = cover_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(cover_model(tensor), dim=1)
        conf, pred = torch.max(probs, 1)
    return ["Covered", "Uncovered"][pred.item()], round(conf.item(), 3)


def _classify_pool(crop: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
    """
    Returns (structure, cover_status, cover_confidence, bbox_area_px2).
    Falls back to sensible defaults when model weights are absent.
    """
    if not MODELS_AVAILABLE or structure_model is None or crop is None or crop.size == 0:
        return "Inground", "Uncovered", 0.75, (xmax - xmin) * (ymax - ymin)

    tensor = structure_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred = torch.max(torch.softmax(structure_model(tensor), dim=1), 1)

    structure    = ["Inground", "Aboveground"][pred.item()]
    cover, conf  = _detect_cover(crop)
    bbox_area    = (xmax - xmin) * (ymax - ymin)
    return structure, cover, conf, bbox_area


def _group_by_property(pools: list, threshold: int = DIST_THRESHOLD_PROPERTY) -> list:
    """
    Cluster detected pools into property groups using spatial proximity.
    Pools whose centers are within `threshold` pixels belong to the same property.
    """
    groups: list = []
    for pool in pools:
        placed = False
        for group in groups:
            for member in group:
                if math.dist(pool["center"], member["center"]) < threshold:
                    group.append(pool)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([pool])
    return groups


def _score_risk(pool_list: list):
    """
    Property-level risk scoring engine.
    Factors: pool presence, cover status, structure type, size, confidence, count.
    Returns (risk_level, risk_score, risk_reasons).
    """
    score, reasons = 0, []

    for pool in pool_list:
        score += 30
        reasons.append("Pool detected on property (+30)")

        if pool["cover"] == "Uncovered":
            score += 25
            reasons.append("Uncovered — drowning/liability risk (+25)")
        elif pool["cover"] == "Covered":
            score += 5
            reasons.append("Covered — reduced exposure (+5)")
        else:
            score += 10
            reasons.append("Cover status unknown (+10)")

        if pool["structure"] == "Aboveground":
            score += 15
            reasons.append("Above-ground — structural instability risk (+15)")
        elif pool["structure"] == "Inground":
            score += 8
            reasons.append("In-ground pool — property integration risk (+8)")
        else:
            score += 10
            reasons.append("Structure type unknown (+10)")

        area = pool.get("area", 0)
        if area > 5000:
            score += 20
            reasons.append(f"Large pool ({area}px²) — high exposure (+20)")
        elif area > 3000:
            score += 10
            reasons.append(f"Medium pool ({area}px²) — moderate exposure (+10)")
        else:
            score += 3
            reasons.append(f"Small pool ({area}px²) — limited exposure (+3)")

        if pool.get("confidence", 1.0) < 0.5:
            score += 5
            reasons.append(f"Low detection confidence ({pool['confidence']}) (+5)")

    if len(pool_list) > 1:
        score += 20
        reasons.append(f"Multiple pools ({len(pool_list)}) on property — compounded risk (+20)")

    if score == 0:
        return "LOW", 0, ["No pool detected — no pool-related risk"]

    level = "LOW" if score < 60 else "MEDIUM" if score < 100 else "HIGH"
    return level, score, reasons


def _compare_pool_states(pools_t1: list, pools_t2: list):
    """
    Time-series pool comparison.
    Returns (added, removed, unchanged) pool lists.
    """
    added, removed, unchanged = [], [], []
    matched_t2 = set()

    for p1 in pools_t1:
        found = False
        for i, p2 in enumerate(pools_t2):
            if i in matched_t2:
                continue
            if math.dist(p1["center"], p2["center"]) < DIST_THRESHOLD_TIMESERIES:
                unchanged.append({"t1": p1, "t2": p2})
                matched_t2.add(i)
                found = True
                break
        if not found:
            removed.append(p1)

    for i, p2 in enumerate(pools_t2):
        if i not in matched_t2:
            added.append(p2)

    return added, removed, unchanged


def _calculate_iou(box1: list, box2: list) -> float:
    """Intersection-over-Union for two [xmin,ymin,xmax,ymax] boxes."""
    xi1 = max(box1[0], box2[0]);  yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]);  yi2 = min(box1[3], box2[3])
    inter  = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union  = area1 + area2 - inter
    return inter / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE  (single image)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(image_path: str):
    """
    Full detection pipeline for one image file.
    1. YOLO pool detection
    2. Per-pool structure + cover classification
    3. Property grouping by spatial proximity
    4. Risk scoring per property group
    Returns: (detected_pools, property_risks)
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return [], []

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return [], []

    detected_pools: list = []

    if yolo_model is not None:
        results = yolo_model(image_path, conf=CONF_THRESHOLD)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                crop  = image[ymin:ymax, xmin:xmax]
                if crop is None or crop.size == 0:
                    continue
                structure, cover, cover_conf, area = _classify_pool(
                    crop, xmin, ymin, xmax, ymax
                )
                detected_pools.append({
                    "confidence":  round(conf, 3),
                    "structure":   structure,
                    "cover":       cover,
                    "cover_conf":  cover_conf,
                    "area":        int(area),
                    "center":      [(xmin + xmax) // 2, (ymin + ymax) // 2],
                    "bbox":        [xmin, ymin, xmax, ymax],
                })
    else:
        # ── Demo fallback when YOLO weights are absent ────────────────────────
        h, w = image.shape[:2]
        detected_pools = [{
            "confidence":  0.88,
            "structure":   "Inground",
            "cover":       "Uncovered",
            "cover_conf":  0.92,
            "area":        6240,
            "center":      [w // 3, h // 3],
            "bbox":        [w // 4, h // 4, w // 2, h // 2],
        }]
        print("[INFO] Demo fallback: returning synthetic pool detection")

    groups = _group_by_property(detected_pools)
    property_risks = []
    for i, group in enumerate(groups):
        risk_level, risk_score, risk_reasons = _score_risk(group)
        property_risks.append({
            "property_id":  f"P-{i+1:03d}",
            "pools":        len(group),
            "risk_level":   risk_level,
            "risk_score":   risk_score,
            "risk_reasons": risk_reasons,
        })

    return detected_pools, property_risks


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED IMAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _annotate_detection(image_path: str, pools: list, property_risks: list) -> str:
    """
    Draws risk-coded bounding boxes and returns base64-encoded JPEG.
    Color: RED = HIGH, ORANGE = MEDIUM, GREEN = LOW risk.
    """
    image = cv2.imread(image_path)
    if image is None:
        return ""

    risk_colors = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 200, 0)}

    for i, pool in enumerate(pools):
        col       = risk_colors.get("HIGH" if pool["cover"] == "Uncovered" else "LOW", (0, 165, 255))
        risk_tag  = "HIGH RISK" if pool["cover"] == "Uncovered" else "LOW RISK"
        xmin, ymin, xmax, ymax = pool["bbox"]

        overlay = image.copy()
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), col, -1)
        image = cv2.addWeighted(overlay, 0.12, image, 0.88, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), col, 2)

        label = f"Pool {i+1} | {pool['structure']} | {risk_tag}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(ymin - 8, th + 8)
        cv2.rectangle(image, (xmin, ly - th - 6), (xmin + tw + 6, ly), (30, 30, 30), -1)
        cv2.putText(image, label, (xmin + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _annotate_comparison(image_path: str, added: list, removed: list, unchanged: list) -> str:
    """
    Draws change-detection overlays and returns base64-encoded JPEG.
    GREEN = unchanged, RED = added (fraud), BLUE = removed.
    """
    image = cv2.imread(image_path)
    if image is None:
        return ""

    for pair in unchanged:
        p = pair["t2"]
        xmin, ymin, xmax, ymax = p["bbox"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        cv2.putText(image, "UNCHANGED", (xmin, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 0), 1, cv2.LINE_AA)

    for p in added:
        xmin, ymin, xmax, ymax = p["bbox"]
        ov = image.copy()
        cv2.rectangle(ov, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)
        image = cv2.addWeighted(ov, 0.15, image, 0.85, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(image, "NEW POOL — FRAUD RISK", (xmin, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 1, cv2.LINE_AA)

    for p in removed:
        xmin, ymin, xmax, ymax = p["bbox"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 80, 0), 2)
        cv2.putText(image, "REMOVED", (xmin, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 80, 0), 1, cv2.LINE_AA)

    legend = [("UNCHANGED", (0, 220, 0)), ("ADDED (Fraud)", (0, 0, 255)), ("REMOVED", (255, 80, 0))]
    lx, ly = 10, 10
    cv2.rectangle(image, (lx, ly), (lx + 260, ly + 12 + len(legend) * 24), (20, 20, 20), -1)
    for m, (lbl, col) in enumerate(legend):
        y = ly + 24 + m * 24
        cv2.rectangle(image, (lx + 8, y - 12), (lx + 22, y + 2), col, -1)
        cv2.putText(image, lbl, (lx + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)

    h, w = image.shape[:2]
    stats = f"Added: {len(added)}  |  Removed: {len(removed)}  |  Unchanged: {len(unchanged)}"
    cv2.rectangle(image, (0, h - 32), (w, h), (20, 20, 20), -1)
    cv2.putText(image, stats, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def underwriting_validation(applicant_declares_pool: bool, detected_pools: list) -> dict:
    """Cross-validates applicant declaration with AI detection results."""
    n = len(detected_pools)
    if not applicant_declares_pool and n > 0:
        return {"status": "NON_DISCLOSED_POOL",
                "message": f"Applicant declared no pool, but {n} pool(s) detected.",
                "severity": "HIGH", "detected_count": n}
    if applicant_declares_pool and n == 0:
        return {"status": "DECLARED_BUT_NOT_VISIBLE",
                "message": "Applicant declared a pool but none detected in imagery.",
                "severity": "MEDIUM", "detected_count": 0}
    return {"status": "CONSISTENT",
            "message": f"Declaration matches imagery — {n} pool(s) detected and consistent.",
            "severity": "LOW", "detected_count": n}


def claims_validation(pools_before: list, pools_after: list) -> dict:
    """Detects post-loss pool construction (fraud signal)."""
    added, removed, unchanged = _compare_pool_states(pools_before, pools_after)
    if len(added) > 0 and len(unchanged) == 0:
        return {"status": "POOL_NOT_PRESENT_BEFORE_LOSS",
                "message": "Pool detected post-loss but absent pre-loss — potential fraud. Escalate to SIU.",
                "severity": "CRITICAL",
                "added": len(added), "removed": len(removed), "unchanged": len(unchanged)}
    if len(unchanged) > 0:
        return {"status": "POOL_EXISTED_PRE_LOSS",
                "message": f"Pool confirmed before loss date. {len(unchanged)} matching pool(s) found.",
                "severity": "LOW",
                "added": len(added), "removed": len(removed), "unchanged": len(unchanged)}
    return {"status": "NO_POOL_INVOLVED",
            "message": "No pool involvement in either image set.",
            "severity": "LOW", "added": 0, "removed": 0, "unchanged": 0}


def get_recommended_actions(uw_result: dict, claims_result: dict, property_risks: list) -> list:
    """Generates actionable underwriting / claims recommendations."""
    actions = []

    if uw_result.get("status") == "NON_DISCLOSED_POOL":
        actions += [
            "⚠️  FLAG: Pool not declared — request updated disclosure form",
            "📋  Schedule physical or IoT-assisted site inspection",
            "💰  Recalculate premium with pool liability endorsement",
            "🔒  Hold policy binding until disclosure resolved",
        ]
    elif uw_result.get("status") == "DECLARED_BUT_NOT_VISIBLE":
        actions += [
            "🔍  Pool declared but not visible — request recent photos",
            "📅  Verify imagery date vs application date",
            "📋  Consider scheduling field inspection",
        ]
    elif uw_result.get("status") == "CONSISTENT":
        actions.append("✅  Declaration consistent with imagery — no disclosure action required")

    if claims_result and claims_result.get("status") == "POOL_NOT_PRESENT_BEFORE_LOSS":
        actions += [
            "🚨  FRAUD ALERT: Escalate to Special Investigations Unit (SIU)",
            "📁  Preserve pre-loss and post-loss imagery as legal evidence",
            "🛑  Suspend claim payment pending full investigation",
        ]
    elif claims_result and claims_result.get("status") == "POOL_EXISTED_PRE_LOSS":
        actions += [
            "✅  Pool confirmed pre-loss — claim eligible for coverage review",
            "📐  Assess pool damage extent relative to policy limits",
        ]

    for pr in property_risks:
        if pr["risk_level"] == "HIGH":
            actions += [
                f"🔴  Property {pr['property_id']}: HIGH RISK (score {pr['risk_score']}) — require safety inspection",
                f"📄  Property {pr['property_id']}: Attach pool liability endorsement",
            ]
        elif pr["risk_level"] == "MEDIUM":
            actions.append(
                f"🟡  Property {pr['property_id']}: MEDIUM RISK (score {pr['risk_score']}) — flag for review"
            )

    if not actions:
        actions.append("✅  No immediate action required")
    return actions


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(image_path: str, detected_pools: list, property_risks: list,
                    uw_result: dict, claims_result: dict = None,
                    time_comparison: dict = None, metrics: dict = None) -> dict:
    """Builds a structured JSON underwriting report."""
    priority = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    overall  = max((p["risk_level"] for p in property_risks),
                   key=lambda x: priority.get(x, 0), default="LOW")

    actions = get_recommended_actions(
        uw_result,
        claims_result or {"status": "NOT_EVALUATED"},
        property_risks,
    )

    report = {
        "report_id":     f"RPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "generated_at":  datetime.utcnow().isoformat(),
        "image_analyzed": image_path,
        "summary": {
            "total_pools_detected":  len(detected_pools),
            "properties_analyzed":   len(property_risks),
            "overall_risk_level":    overall,
            "disclosure_status":     uw_result.get("status"),
            "disclosure_severity":   uw_result.get("severity"),
        },
        "pools_detected": [
            {
                "pool_id":            i + 1,
                "structure":          p["structure"],
                "cover":              p["cover"],
                "cover_confidence":   p.get("cover_conf"),
                "detection_confidence": p["confidence"],
                "area_px2":           p["area"],
                "center":             p["center"],
                "bbox":               p["bbox"],
            }
            for i, p in enumerate(detected_pools)
        ],
        "property_risk_assessment": [
            {
                "property_id":   pr["property_id"],
                "pools_on_property": pr["pools"],
                "risk_level":    pr["risk_level"],
                "risk_score":    pr["risk_score"],
                "risk_reasons":  pr.get("risk_reasons", []),
            }
            for pr in property_risks
        ],
        "underwriting_validation": uw_result,
        "claims_validation":       claims_result or "Not evaluated",
        "recommended_actions":     actions,
    }

    if metrics:
        report["model_performance_metrics"] = metrics
    if time_comparison:
        report["time_series_comparison"] = {
            "added_pools":      time_comparison.get("added", 0),
            "removed_pools":    time_comparison.get("removed", 0),
            "unchanged_pools":  time_comparison.get("unchanged", 0),
            "change_detected":  time_comparison.get("added", 0) > 0 or time_comparison.get("removed", 0) > 0,
        }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# GEOJSON EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_geojson(detected_pools: list) -> dict:
    """Exports detected pools as a GeoJSON FeatureCollection (pixel coords)."""
    features = []
    for i, pool in enumerate(detected_pools):
        xmin, ymin, xmax, ymax = pool["bbox"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[xmin, ymin], [xmax, ymin],
                                  [xmax, ymax], [xmin, ymax], [xmin, ymin]]]
            },
            "properties": {
                "pool_id":               i + 1,
                "structure_type":        pool["structure"],
                "cover_status":          pool["cover"],
                "cover_confidence":      pool.get("cover_conf"),
                "detection_confidence":  pool["confidence"],
                "area_px2":              pool["area"],
                "center_x":              pool["center"][0],
                "center_y":              pool["center"][1],
                "risk_indicator":        "HIGH" if pool["cover"] == "Uncovered" else "MEDIUM",
            }
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated_at":      datetime.utcnow().isoformat(),
            "total_pools":       len(features),
            "coordinate_system": "pixel (convert to WGS84 using GeoTIFF metadata)",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCR + DOCUMENT FRAUD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _ocr_extract(file_bytes: bytes, filename: str) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        if filename.lower().endswith(".pdf"):
            pages = convert_from_bytes(file_bytes)
            return "\n".join(pytesseract.image_to_string(p) for p in pages)
        else:
            img = PILImage.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[OCR Error] {e}")
        return ""


def _extract_doc_fields(text: str) -> dict:
    import re
    cert  = re.search(r'Certificate No\.?\s*[:\-]?\s*(\S+)',         text, re.IGNORECASE)
    date  = re.search(r'Certificate Issued Date\s*[:\-]?\s*([\dA-Za-z:\- ]+)', text, re.IGNORECASE)
    owner = re.search(r'Purchased by\s*[:\-]?\s*([A-Z ,]+)',         text, re.IGNORECASE)
    stamp = re.search(r'Stamp Duty Amount\(Rs\.\)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
    return {
        "certificate_number": cert.group(1)          if cert  else None,
        "issued_date":        date.group(1).strip()  if date  else None,
        "purchased_by":       owner.group(1).strip() if owner else None,
        "stamp_duty_amount":  stamp.group(1)         if stamp else None,
    }


def _score_doc(fields: dict):
    score, reasons = 0, []
    if not fields.get("certificate_number"):
        score += 30; reasons.append("Missing certificate number (+30)")
    if not fields.get("purchased_by"):
        score += 25; reasons.append("Missing owner name (+25)")
    if not fields.get("stamp_duty_amount"):
        score += 20; reasons.append("Missing stamp duty amount (+20)")
    if not fields.get("issued_date"):
        score += 15; reasons.append("Missing issued date (+15)")
    cert = fields.get("certificate_number")
    if cert and not cert.startswith("IN-"):
        score += 20; reasons.append("Suspicious certificate format (+20)")
    stamp = fields.get("stamp_duty_amount")
    if stamp:
        try:
            if int(stamp) < 10:
                score += 15; reasons.append("Unusually low stamp duty (+15)")
        except ValueError:
            score += 10; reasons.append("Invalid stamp duty value (+10)")
    level = "Low" if score <= 20 else "Medium" if score <= 50 else "High"
    rec   = ("Safe to Process" if level == "Low"
             else "Manual Review Required" if level == "Medium"
             else "High Fraud Risk — Investigate Immediately")
    return score, level, rec, reasons


def _ml_doc_score(fields: dict):
    if not ML_AVAILABLE:
        return "Normal", 0.0
    stamp     = int(fields.get("stamp_duty_amount") or 0)
    cert_len  = len(fields.get("certificate_number") or "")
    owner_len = len(fields.get("purchased_by") or "")
    year      = 0
    if fields.get("issued_date"):
        try:
            year = int(str(fields["issued_date"]).strip().split("-")[-1][:4])
        except Exception:
            pass
    feat = np.array([[stamp, cert_len, owner_len, year]])
    pred  = if_model.predict(feat)[0]
    score = float(if_model.decision_function(feat)[0])
    return ("Anomaly Detected" if pred == -1 else "Normal"), round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# COMMUNICATIONS
# ─────────────────────────────────────────────────────────────────────────────
def _send_twilio_sms(to: str, body: str) -> bool:
    if not TWILIO_AVAILABLE or not TWILIO_SID:
        print(f"[SMS] Queued (Twilio not configured): to={to}")
        return False
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=body, from_=TWILIO_FROM, to=to)
        print(f"[SMS] Sent to {to}")
        return True
    except Exception as e:
        print(f"[Twilio Error] {e}")
        return False


def _send_email(to: str, subject: str, body: str) -> bool:
    if not SMTP_USER:
        print(f"[EMAIL] Queued (SMTP not configured): to={to}, subject={subject}")
        return False
    try:
        msg           = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = to
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print(f"[EMAIL] Sent to {to}: {subject}")
        return True
    except Exception as e:
        print(f"[SMTP Error] {e}")
        return False


def _iot_level(val: int) -> str:
    return "LOW" if val < 300 else "MEDIUM" if val < 700 else "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def _save_upload(upload: UploadFile) -> str:
    ext  = os.path.splitext(upload.filename or "file.jpg")[-1] or ".jpg"
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════
class UWRequest(BaseModel):
    applicant_declares_pool: bool
    detected_pool_count: int


class ClaimSubmitRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    property_addr: str
    claim_type: str
    incident_date: str
    estimated_loss: float = 0
    description: str
    pool_declared: bool = False
    witness_name: Optional[str] = None
    witness_contact: Optional[str] = None
    policyholder_id: Optional[str] = None


class ClaimStatusUpdate(BaseModel):
    status: str
    risk_level: Optional[str]   = None
    risk_score: Optional[float] = None
    fraud_flag: Optional[bool]  = None
    uw_notes:   Optional[str]   = None


class IoTReadingRequest(BaseModel):
    sensor_value:    int
    inspector_name:  Optional[str] = None
    notes:           Optional[str] = None
    claim_id:        Optional[str] = None
    property_addr:   Optional[str] = None
    send_sms_if_high: bool = True


class PolicyholderRequest(BaseModel):
    name:           str
    email:          str
    phone:          str
    policy_number:  str
    property_addr:  str
    pin_code:       Optional[str]   = None
    pool_declared:  bool            = False
    pool_type:      Optional[str]   = None
    pool_covered:   Optional[bool]  = None
    pool_year:      Optional[str]   = None
    pool_area_sqft: Optional[float] = None
    pool_notes:     Optional[str]   = None


class EmailRequest(BaseModel):
    to:       str
    subject:  str
    body:     str
    claim_id: Optional[str] = None


class SMSRequest(BaseModel):
    to:       str
    message:  str
    claim_id: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "AquaIntelligence API",
    description = "AI-powered swimming pool detection, underwriting & claims intelligence",
    version     = "3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT / HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "AquaIntelligence API",
        "version": "3.1.0",
        "status":  "online",
        "models": {
            "yolo":               yolo_model is not None,
            "cover_classifier":   cover_model is not None,
            "structure_classifier": structure_model is not None,
            "isolation_forest":   if_model is not None,
            "ocr":                OCR_AVAILABLE,
            "twilio":             TWILIO_AVAILABLE and bool(TWILIO_SID),
            "smtp":               bool(SMTP_USER),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/overview")
def get_overview(db: Session = Depends(get_db)):
    """Aggregated stats for the insurer overview dashboard."""
    claims = db.query(ClaimDB).all()
    iot    = db.query(IoTReadingDB).order_by(IoTReadingDB.recorded_at.desc()).limit(20).all()

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    fraud_count = 0
    for c in claims:
        risk_counts[c.risk_level] = risk_counts.get(c.risk_level, 0) + 1
        if c.fraud_flag:
            fraud_count += 1

    iot_vals = [r.sensor_value for r in iot]

    return {
        "total_claims":         len(claims),
        "pending_claims":       sum(1 for c in claims if c.status == "PENDING"),
        "fraud_flags":          fraud_count,
        "high_risk_properties": risk_counts.get("HIGH", 0),
        "risk_distribution":    risk_counts,
        "latest_iot_value":     iot_vals[0] if iot_vals else 0,
        "latest_iot_level":     _iot_level(iot_vals[0]) if iot_vals else "UNKNOWN",
        "iot_history":          iot_vals[:8][::-1],
        "recent_claims": [
            {
                "id":           c.id,
                "name":         c.name,
                "property":     c.property_addr,
                "status":       c.status,
                "risk_level":   c.risk_level,
                "fraud_flag":   c.fraud_flag,
                "submitted_at": c.submitted_at.isoformat(),
            }
            for c in sorted(claims, key=lambda x: x.submitted_at, reverse=True)[:6]
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    """
    Detect swimming pools in a single aerial/satellite image.

    Multipart field: `image` (JPEG / PNG)

    Returns:
        pools[]              — per-pool detection details
        property_risks[]     — grouped risk assessment
        annotated_image      — base64 JPEG with bounding boxes
        summary              — totals and overall risk
    """
    path = _save_upload(image)
    try:
        pools, risks  = run_pipeline(path)
        annotated_b64 = _annotate_detection(path, pools, risks)
        overall = max(
            (r["risk_level"] for r in risks),
            key=lambda x: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(x, 0),
            default="LOW",
        )
        return {
            "pools":           pools,
            "property_risks":  risks,
            "annotated_image": annotated_b64,
            "summary": {
                "total_pools":    len(pools),
                "properties":     len(risks),
                "overall_risk":   overall,
            },
        }
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/api/compare")
async def compare(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
):
    """
    Time-series change detection: T1 (pre-loss) vs T2 (post-loss).

    Multipart fields: `image_t1`, `image_t2`

    Returns:
        t1 / t2              — per-image detection results + annotated images
        comparison           — added / removed / unchanged counts
        claims_validation    — fraud risk assessment
    """
    p1 = _save_upload(image_t1)
    p2 = _save_upload(image_t2)
    try:
        pools_t1, risks_t1 = run_pipeline(p1)
        pools_t2, risks_t2 = run_pipeline(p2)

        added, removed, unchanged = _compare_pool_states(pools_t1, pools_t2)
        claims_result = claims_validation(pools_t1, pools_t2)

        ann_t1 = _annotate_detection(p1, pools_t1, risks_t1)
        ann_t2 = _annotate_comparison(p2, added, removed, unchanged)

        return {
            "t1": {
                "pools":            pools_t1,
                "property_risks":   risks_t1,
                "annotated_image":  ann_t1,
            },
            "t2": {
                "pools":            pools_t2,
                "property_risks":   risks_t2,
                "annotated_image":  ann_t2,
            },
            "comparison": {
                "added":     len(added),
                "removed":   len(removed),
                "unchanged": len(unchanged),
            },
            "claims_validation": claims_result,
        }
    finally:
        for p in (p1, p2):
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
# UNDERWRITING
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/underwriting/validate")
def underwriting_validate(req: UWRequest):
    """
    Cross-check applicant pool declaration against AI detection count.

    Body: { "applicant_declares_pool": false, "detected_pool_count": 2 }
    """
    dummy_pools = [{}] * req.detected_pool_count
    result      = underwriting_validation(req.applicant_declares_pool, dummy_pools)

    actions = []
    if result["status"] == "NON_DISCLOSED_POOL":
        actions = [
            "⚠️  FLAG: Pool not declared — request updated disclosure form",
            "📋  Schedule IoT-assisted site inspection",
            "💰  Recalculate premium — apply pool liability endorsement",
            "🔒  Hold policy binding until disclosure resolved",
            "📱  Notify underwriter via SMS (Twilio)",
        ]
    elif result["status"] == "DECLARED_BUT_NOT_VISIBLE":
        actions = [
            "🔍  Pool declared but not visible — request recent photos from applicant",
            "📅  Check if imagery is outdated relative to application date",
            "📋  Schedule field inspection to confirm",
        ]
    else:
        actions = ["✅  Declaration consistent with imagery — apply standard pool endorsement"]

    return {**result, "recommended_actions": actions}


# ─────────────────────────────────────────────────────────────────────────────
# CLAIMS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/claims")
def list_claims(
    status:     Optional[str]  = Query(None),
    risk_level: Optional[str]  = Query(None),
    fraud_only: bool           = Query(False),
    db: Session = Depends(get_db),
):
    """List all claims (insurer view). Supports ?status=PENDING, ?risk_level=HIGH, ?fraud_only=true"""
    q = db.query(ClaimDB)
    if status:
        q = q.filter(ClaimDB.status == status)
    if risk_level:
        q = q.filter(ClaimDB.risk_level == risk_level)
    if fraud_only:
        q = q.filter(ClaimDB.fraud_flag == True)  # noqa: E712
    return [
        {
            "id":            c.id,
            "name":          c.name,
            "email":         c.email,
            "phone":         c.phone,
            "property":      c.property_addr,
            "claim_type":    c.claim_type,
            "incident_date": c.incident_date,
            "estimated_loss": c.estimated_loss,
            "pool_declared": c.pool_declared,
            "status":        c.status,
            "risk_level":    c.risk_level,
            "risk_score":    c.risk_score,
            "fraud_flag":    c.fraud_flag,
            "uw_notes":      c.uw_notes,
            "submitted_at":  c.submitted_at.isoformat(),
        }
        for c in q.order_by(ClaimDB.submitted_at.desc()).all()
    ]


@app.post("/api/claims", status_code=201)
def submit_claim(req: ClaimSubmitRequest, db: Session = Depends(get_db)):
    """Policyholder submits a new insurance claim."""
    claim = ClaimDB(
        name=req.name, email=req.email, phone=req.phone,
        property_addr=req.property_addr, claim_type=req.claim_type,
        incident_date=req.incident_date, estimated_loss=req.estimated_loss,
        description=req.description, pool_declared=req.pool_declared,
        witness_name=req.witness_name, witness_contact=req.witness_contact,
        policyholder_id=req.policyholder_id,
        status="PENDING",
    )
    db.add(claim); db.commit(); db.refresh(claim)
    return {
        "claim_id":     claim.id,
        "status":       "PENDING",
        "message":      "Claim submitted. AI satellite verification will be completed within 24 hours.",
        "submitted_at": claim.submitted_at.isoformat(),
    }


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    """Get single claim with documents, IoT readings, and progress step."""
    c = db.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Claim not found")

    docs = db.query(DocumentDB).filter(DocumentDB.claim_id == claim_id).all()
    iot  = db.query(IoTReadingDB).filter(IoTReadingDB.claim_id == claim_id).all()

    progress_map = {
        "PENDING": 1, "AI_VERIFICATION": 2, "IOT_INSPECTION": 3,
        "UNDERWRITING_REVIEW": 4, "APPROVED": 5, "REJECTED": 5, "FRAUD_SUSPECTED": 5,
    }
    return {
        "id":            c.id,
        "name":          c.name,
        "email":         c.email,
        "phone":         c.phone,
        "property":      c.property_addr,
        "claim_type":    c.claim_type,
        "incident_date": c.incident_date,
        "estimated_loss": c.estimated_loss,
        "description":   c.description,
        "pool_declared": c.pool_declared,
        "status":        c.status,
        "risk_level":    c.risk_level,
        "risk_score":    c.risk_score,
        "fraud_flag":    c.fraud_flag,
        "uw_notes":      c.uw_notes,
        "submitted_at":  c.submitted_at.isoformat(),
        "documents": [{"id": d.id, "filename": d.filename, "status": d.status} for d in docs],
        "iot_readings": [
            {"value": r.sensor_value, "level": r.risk_level, "at": r.recorded_at.isoformat()}
            for r in iot
        ],
        "progress_step": progress_map.get(c.status, 1),
    }


@app.put("/api/claims/{claim_id}/status")
def update_claim_status(claim_id: str, req: ClaimStatusUpdate,
                         db: Session = Depends(get_db)):
    """Insurer updates claim status, risk level, fraud flag, and UW notes."""
    c = db.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Claim not found")
    c.status     = req.status
    c.updated_at = datetime.utcnow()
    if req.risk_level is not None: c.risk_level = req.risk_level
    if req.risk_score is not None: c.risk_score = req.risk_score
    if req.fraud_flag is not None: c.fraud_flag = req.fraud_flag
    if req.uw_notes   is not None: c.uw_notes   = req.uw_notes
    db.commit()
    return {"message": "Claim updated", "claim_id": claim_id, "new_status": req.status}


# ─────────────────────────────────────────────────────────────────────────────
# IOT  (ESP8266 drone / inspector readings)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/iot/reading")
def submit_iot_reading(req: IoTReadingRequest, db: Session = Depends(get_db)):
    """
    Submit an ESP8266 sensor reading.
    Automatically sends Twilio SMS to INSURER_PHONE when risk level is HIGH.
    """
    if not (0 <= req.sensor_value <= 1023):
        raise HTTPException(status_code=422, detail="sensor_value must be 0–1023")

    level    = _iot_level(req.sensor_value)
    sms_sent = False

    if level == "HIGH" and req.send_sms_if_high and INSURER_PHONE:
        msg = (
            f"AquaIntelligence ALERT: HIGH risk IoT reading ({req.sensor_value}) "
            f"at {req.property_addr or 'unknown property'}. "
            f"Inspector: {req.inspector_name or 'N/A'}. "
            f"Claim: {req.claim_id or 'N/A'}. Immediate review required."
        )
        sms_sent = _send_twilio_sms(INSURER_PHONE, msg)

    row = IoTReadingDB(
        sensor_value=req.sensor_value, risk_level=level,
        inspector_name=req.inspector_name, notes=req.notes,
        claim_id=req.claim_id, property_addr=req.property_addr,
        sms_sent=sms_sent,
    )
    db.add(row); db.commit(); db.refresh(row)

    return {
        "reading_id":   row.id,
        "sensor_value": req.sensor_value,
        "risk_level":   level,
        "led_pct":      round((req.sensor_value / 1023) * 100, 1),
        "sms_sent":     sms_sent,
        "message":      f"Reading recorded. Risk: {level}." +
                        (" Twilio SMS dispatched." if sms_sent else ""),
    }


@app.get("/api/iot/readings")
def get_iot_readings(limit: int = 50, db: Session = Depends(get_db)):
    """Sensor reading history (most recent first)."""
    rows = db.query(IoTReadingDB).order_by(IoTReadingDB.recorded_at.desc()).limit(limit).all()
    return [
        {
            "id":            r.id,
            "sensor_value":  r.sensor_value,
            "risk_level":    r.risk_level,
            "inspector_name": r.inspector_name,
            "notes":         r.notes,
            "claim_id":      r.claim_id,
            "property_addr": r.property_addr,
            "sms_sent":      r.sms_sent,
            "recorded_at":   r.recorded_at.isoformat(),
        }
        for r in rows
    ]


@app.get("/api/iot/latest")
def get_iot_latest(db: Session = Depends(get_db)):
    """Latest sensor reading plus recent history array (for the chart)."""
    row  = db.query(IoTReadingDB).order_by(IoTReadingDB.recorded_at.desc()).first()
    hist = db.query(IoTReadingDB).order_by(IoTReadingDB.recorded_at.desc()).limit(8).all()
    if not row:
        return {"sensor_value": 0, "risk_level": "UNKNOWN", "history": []}
    return {
        "sensor_value": row.sensor_value,
        "risk_level":   row.risk_level,
        "led_pct":      round((row.sensor_value / 1023) * 100, 1),
        "recorded_at":  row.recorded_at.isoformat(),
        "history":      [r.sensor_value for r in reversed(hist)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/documents/analyze")
async def analyze_document(
    file:            UploadFile       = File(...),
    claim_id:        Optional[str]    = None,
    policyholder_id: Optional[str]    = None,
    db: Session = Depends(get_db),
):
    """
    Upload property document (PDF or image).
    Pipeline: Tesseract OCR → field extraction → rule scoring → Isolation Forest flag.
    """
    raw    = await file.read()
    text   = _ocr_extract(raw, file.filename)
    fields = _extract_doc_fields(text)

    rule_score, risk_level, recommendation, reasons = _score_doc(fields)
    ml_flag, ml_score = _ml_doc_score(fields)

    final_score = rule_score
    if ml_flag == "Anomaly Detected":
        final_score += 25
        reasons.append("ML Isolation Forest flagged as anomalous (+25)")

    doc = DocumentDB(
        claim_id=claim_id, policyholder_id=policyholder_id,
        filename=file.filename,
        cert_number=fields.get("certificate_number"),
        owner_name=fields.get("purchased_by"),
        risk_score=final_score,
        risk_level=risk_level,
        ml_flag=ml_flag,
        recommendation=recommendation,
        risk_reasons=json.dumps(reasons),
        status="VERIFIED" if risk_level == "Low" else "UNDER_REVIEW",
    )
    db.add(doc); db.commit(); db.refresh(doc)

    return {
        "document_id":          doc.id,
        "filename":             file.filename,
        "extracted_fields":     fields,
        "ocr_text_preview":     text[:400] if text else "(OCR not available — install pytesseract)",
        "rule_based_risk_score": rule_score,
        "ml_anomaly_score":     ml_score,
        "ml_flag":              ml_flag,
        "final_risk_score":     final_score,
        "risk_level":           risk_level,
        "recommendation":       recommendation,
        "risk_reasons":         reasons,
        "status":               doc.status,
    }


@app.get("/api/documents/{claim_id}")
def get_documents(claim_id: str, db: Session = Depends(get_db)):
    """List all documents attached to a claim."""
    docs = db.query(DocumentDB).filter(DocumentDB.claim_id == claim_id).all()
    return [
        {
            "id":           d.id,
            "filename":     d.filename,
            "risk_score":   d.risk_score,
            "risk_level":   d.risk_level,
            "ml_flag":      d.ml_flag,
            "recommendation": d.recommendation,
            "risk_reasons": json.loads(d.risk_reasons) if d.risk_reasons else [],
            "status":       d.status,
            "uploaded_at":  d.uploaded_at.isoformat(),
        }
        for d in docs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# POLICYHOLDER
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/policyholder/register", status_code=201)
def register_policyholder(req: PolicyholderRequest, db: Session = Depends(get_db)):
    """Policyholder registers property and submits pool declaration."""
    ph = PolicyholderDB(
        name=req.name, email=req.email, phone=req.phone,
        policy_number=req.policy_number, property_addr=req.property_addr,
        pin_code=req.pin_code, pool_declared=req.pool_declared,
        pool_type=req.pool_type, pool_covered=req.pool_covered,
        pool_year=req.pool_year, pool_area_sqft=req.pool_area_sqft,
        pool_notes=req.pool_notes,
    )
    db.add(ph); db.commit(); db.refresh(ph)
    return {
        "policyholder_id": ph.id,
        "message": "Registration successful. Insurer will verify via AI satellite detection within 2–3 business days.",
        "pool_declared": req.pool_declared,
    }


@app.get("/api/policyholder/{ph_id}")
def get_policyholder(ph_id: str, db: Session = Depends(get_db)):
    """Get policyholder profile with linked claims."""
    ph = db.query(PolicyholderDB).filter(PolicyholderDB.id == ph_id).first()
    if not ph:
        raise HTTPException(status_code=404, detail="Policyholder not found")
    claims = db.query(ClaimDB).filter(ClaimDB.policyholder_id == ph_id).all()
    return {
        "id":           ph.id,
        "name":         ph.name,
        "email":        ph.email,
        "phone":        ph.phone,
        "policy_number": ph.policy_number,
        "property_addr": ph.property_addr,
        "pool_declared": ph.pool_declared,
        "pool_type":    ph.pool_type,
        "pool_covered": ph.pool_covered,
        "pool_year":    ph.pool_year,
        "pool_area_sqft": ph.pool_area_sqft,
        "pool_notes":   ph.pool_notes,
        "created_at":   ph.created_at.isoformat(),
        "claims": [{"id": c.id, "status": c.status, "type": c.claim_type} for c in claims],
    }


@app.put("/api/policyholder/{ph_id}")
def update_policyholder(ph_id: str, req: PolicyholderRequest,
                         db: Session = Depends(get_db)):
    """Update policyholder profile and pool declaration."""
    ph = db.query(PolicyholderDB).filter(PolicyholderDB.id == ph_id).first()
    if not ph:
        raise HTTPException(status_code=404, detail="Policyholder not found")
    for k, v in req.dict().items():
        setattr(ph, k, v)
    db.commit()
    return {"message": "Profile updated", "policyholder_id": ph_id}


# ─────────────────────────────────────────────────────────────────────────────
# COMMUNICATIONS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/comms/email")
def send_email_endpoint(req: EmailRequest, db: Session = Depends(get_db)):
    """Send email to policyholder (fraud warning, disclosure request, approval)."""
    sent = _send_email(req.to, req.subject, req.body)
    log  = CommLogDB(
        comm_type="EMAIL", recipient=req.to,
        subject=req.subject, body=req.body,
        status="SENT" if sent else "QUEUED",
        claim_id=req.claim_id,
    )
    db.add(log); db.commit()
    return {
        "success":  True,
        "delivery": "sent" if sent else "queued (configure SMTP_USER/SMTP_PASS to deliver)",
        "log_id":   log.id,
        "message":  f"Email {'sent' if sent else 'logged'} to {req.to}",
    }


@app.post("/api/comms/sms")
def send_sms_endpoint(req: SMSRequest, db: Session = Depends(get_db)):
    """Send Twilio SMS to policyholder or insurer."""
    sent = _send_twilio_sms(req.to, req.message)
    log  = CommLogDB(
        comm_type="SMS", recipient=req.to,
        body=req.message,
        status="SENT" if sent else "QUEUED",
        claim_id=req.claim_id,
    )
    db.add(log); db.commit()
    return {
        "success":  True,
        "delivery": "sent" if sent else "queued (configure TWILIO_ACCOUNT_SID to deliver)",
        "log_id":   log.id,
        "message":  f"SMS {'sent' if sent else 'logged'} to {req.to}",
    }


@app.get("/api/comms/log")
def get_comm_log(limit: int = 50, db: Session = Depends(get_db)):
    """Get sent communications log (email + SMS)."""
    rows = db.query(CommLogDB).order_by(CommLogDB.sent_at.desc()).limit(limit).all()
    return [
        {
            "id":       r.id,
            "type":     r.comm_type,
            "to":       r.recipient,
            "subject":  r.subject,
            "body":     r.body[:120],
            "status":   r.status,
            "claim_id": r.claim_id,
            "sent_at":  r.sent_at.isoformat(),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# REPORT + GEOJSON
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/report/generate")
async def report_generate(
    image:                  UploadFile = File(...),
    applicant_declares_pool: bool      = False,
    image_t1:               Optional[UploadFile] = File(None),
):
    """
    One-shot endpoint: upload an image (and optional T1 for comparison),
    run the full pipeline, and return the complete structured JSON report.
    """
    path = _save_upload(image)
    p1   = _save_upload(image_t1) if image_t1 else None
    try:
        pools, risks = run_pipeline(path)
        uw_result    = underwriting_validation(applicant_declares_pool, pools)

        claims_result = None
        time_comp     = None
        if p1:
            pools_t1, _ = run_pipeline(p1)
            added, removed, unchanged = _compare_pool_states(pools_t1, pools)
            claims_result = claims_validation(pools_t1, pools)
            time_comp     = {"added": len(added), "removed": len(removed), "unchanged": len(unchanged)}

        report = generate_report(
            image_path=image.filename,
            detected_pools=pools,
            property_risks=risks,
            uw_result=uw_result,
            claims_result=claims_result,
            time_comparison=time_comp,
        )
        return report
    finally:
        for p in filter(None, [path, p1]):
            if os.path.exists(p):
                os.remove(p)


@app.post("/api/geojson/export")
async def geojson_export(image: UploadFile = File(...)):
    """
    Detect pools in image and return results as a GeoJSON FeatureCollection.
    Pixel coordinates — convert to WGS84 using image GeoTIFF metadata in production.
    """
    path = _save_upload(image)
    try:
        pools, _ = run_pipeline(path)
        return export_geojson(pools)
    finally:
        if os.path.exists(path):
            os.remove(path)