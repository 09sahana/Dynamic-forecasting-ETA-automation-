"""
api/app.py
==========
FastAPI application for Phase 7 ETA Prediction Service.
Provides REST endpoints, WebSocket streaming, and hosts the real-time operations dashboard.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.journey_aggregator import aggregate_journey_eta
from api.schemas import (
    FullJourneyResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from api.stream_manager import stream_manager
from src.config import BASE_DIR
from src.explainability import explain_prediction
from src.predict import predict_delay_and_eta

METADATA_PATH = BASE_DIR / "models" / "metadata.json"
DASHBOARD_DIR = BASE_DIR / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stream_manager.start()
    yield
    # Shutdown
    await stream_manager.stop()


app = FastAPI(
    title="Indian Railways ETA Prediction Service",
    description="Productionized ML-powered dynamic ETA prediction with calibrated confidence intervals and leakage-audited features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_version": "v6",
        "service": "Indian Railways ETA Prediction API",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/metrics")
def model_metrics() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    obs = {
        "train_no": req.train_no,
        "journey_date": req.date,
        "station_code": req.station,
        "previous_station_delay": req.previous_station_delay,
        "temperature_c": req.temperature_c,
        "precipitation_mm": req.precipitation_mm,
        "visibility_m": req.visibility_m,
        "event_type": req.event_type or "NONE",
        "event_elapsed_min": req.event_elapsed_min or 0.0,
    }

    result = predict_delay_and_eta(
        train_no=req.train_no,
        journey_date=req.date,
        station_code=req.station,
        previous_station_delay=req.previous_station_delay,
        event_type=req.event_type or "NONE",
        event_elapsed_min=req.event_elapsed_min or 0.0,
    )

    # Resolve chronic tier string (T1-T5)
    from src.predict import load_inference_bundle
    bundle = load_inference_bundle()
    fp = bundle.get("feature_pipeline")
    tier_num = getattr(fp, "chronic_delay_tier_map", {}).get(req.train_no, 3)
    tier_str = f"T{tier_num}_worst" if tier_num == 5 else (f"T{tier_num}_best" if tier_num == 1 else f"T{tier_num}")

    result["chronic_delay_tier"] = tier_str
    explanations = explain_prediction(obs, result)

    return {
        "train_no": req.train_no,
        "journey_date": req.date,
        "station_code": req.station,
        "scheduled_arrival": result.get("scheduled_arrival", "00:00:00"),
        "predicted_delay_min": result.get("predicted_delay_min"),
        "predicted_delay_range_min": result.get("predicted_delay_range_min"),
        "predicted_eta": result.get("predicted_eta"),
        "predicted_eta_range": result.get("predicted_eta_range"),
        "confidence": result.get("confidence", "Medium"),
        "severe_delay_risk": result.get("severe_delay_risk", 0.0),
        "chronic_delay_tier": tier_str,
        "chronic_tier_is_fallback": result.get("chronic_tier_is_fallback", 0),
        "event_type": result.get("event_type", "NONE"),
        "event_elapsed_min": result.get("event_elapsed_min", 0.0),
        "status": result.get("status", "ON_TIME"),
        "explanations": explanations,
    }


@app.get("/train/{train_no}/eta", response_model=FullJourneyResponse)
def full_journey_eta(
    train_no: int,
    date: str = Query(default_factory=lambda: datetime.now().strftime("%Y-%m-%d")),
    current_station: str = Query(None, description="Current train station code"),
    current_delay: float = Query(0.0, description="Current delay in minutes"),
    event_type: str = Query("NONE", description="Active event type if any"),
    event_elapsed: float = Query(0.0, description="Elapsed incident duration"),
):
    return aggregate_journey_eta(
        train_no=train_no,
        journey_date=date,
        current_station_code=current_station,
        current_delay_min=current_delay,
        event_type=event_type,
        event_elapsed_min=event_elapsed,
    )


@app.get("/api/stream/status")
def stream_status():
    return {
        "mode": stream_manager.mode,
        "current_train": stream_manager.current_train,
        "active_clients": len(stream_manager.active_connections),
        "running_mae": (
            round(sum(x["error"] for x in stream_manager.mae_tracking) / len(stream_manager.mae_tracking), 2)
            if stream_manager.mae_tracking
            else 0.0
        ),
    }


@app.post("/api/stream/mode")
def set_stream_mode(mode: str = Query(..., pattern="^(simulator|railradar)$"), train_no: int = Query(12952)):
    stream_manager.set_mode(mode, train_no)
    return {"status": "ok", "mode": mode, "train_no": train_no}


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    await stream_manager.connect(websocket)
    try:
        while True:
            # Keep-alive / incoming command listener
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                if "set_mode" in cmd:
                    stream_manager.set_mode(cmd["set_mode"], cmd.get("train_no", 12952))
            except Exception:
                pass
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)


# Mount dashboard frontend if directory exists
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
