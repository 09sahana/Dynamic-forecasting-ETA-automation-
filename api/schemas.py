"""
api/schemas.py
==============
Pydantic data models for the Phase 7 ETA Prediction Service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    train_no: int = Field(..., description="Train number (e.g. 12952)")
    date: str = Field(..., description="Journey date in YYYY-MM-DD format")
    station: str = Field(..., description="Station code (e.g. BRC, ST, BVI)")
    previous_station_delay: float = Field(0.0, description="Running delay at prior station (minutes)")
    temperature_c: Optional[float] = Field(None, description="Ambient temperature (°C)")
    precipitation_mm: Optional[float] = Field(None, description="Hourly rainfall (mm)")
    visibility_m: Optional[float] = Field(None, description="Visibility (meters)")
    event_elapsed_min: Optional[float] = Field(0.0, description="Minutes elapsed since incident began")
    event_type: Optional[str] = Field("NONE", description="Operational event type (CONGESTION, TRAIN_HELD, NONE)")


class ExplanationItem(BaseModel):
    factor: str
    impact: str
    description: str
    direction: str


class PredictResponse(BaseModel):
    train_no: int
    journey_date: str
    station_code: str
    scheduled_arrival: str
    predicted_delay_min: Optional[float]
    predicted_delay_range_min: Optional[List[float]]
    predicted_eta: Optional[str]
    predicted_eta_range: Optional[List[str]]
    confidence: str
    severe_delay_risk: float
    chronic_delay_tier: str
    chronic_tier_is_fallback: int
    event_type: str
    event_elapsed_min: float
    status: str
    explanations: List[ExplanationItem] = []


class FullJourneyStationItem(BaseModel):
    seq: int
    station_code: str
    station_name: Optional[str] = None
    distance_km: float
    scheduled_arrival: str
    predicted_delay_min: Optional[float] = None
    predicted_eta: Optional[str] = None
    confidence: Optional[str] = None
    severe_delay_risk: Optional[float] = None


class FullJourneyResponse(BaseModel):
    train_no: int
    journey_date: str
    origin_station: str
    destination_station: str
    total_distance_km: float
    stations: List[FullJourneyStationItem]
    final_predicted_eta: Optional[str] = None
    final_predicted_delay_min: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    service: str
    timestamp: str
