"""
API Service — OTT Audience Segmentation
========================================
Endpoints:
  GET  /health     — Returns service + model status
  POST /recommend  — Accepts a user profile and returns segment + recommendations

Model and feature schema are loaded from /app/models/ at startup.
No retraining at inference. Deterministic outputs.
"""

import os
import json
import math
import logging
import threading
import time
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from starlette.exceptions import HTTPException as StarletteHTTPException

# Fixed seed for deterministic behavior
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

MODELS_DIR = "/app/models"
MAX_USER_ID_LENGTH = 100
MAX_GENRES_LIST_SIZE = 20
MAX_GENRE_STRING_LENGTH = 50

# --- Global state ---
_model_pipeline = None
_model_metadata = None
_model_loaded = False
_load_lock = threading.Lock()


import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OTTFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("OTTFeatureTransformer expects a DataFrame")

        result = pd.DataFrame(index=df.index)

        # Log1p transform for right-skewed watch time
        result["log_watch_time"] = np.log1p(
            df["total_watch_time_hours"].fillna(0).clip(lower=0)
        )
        result["avg_session_mins"] = df["avg_session_mins"].fillna(0).clip(lower=0)
        result["sessions_per_week"] = df["sessions_per_week"].fillna(0).clip(lower=0)
        result["weekend_ratio"] = df["weekend_ratio"].fillna(0).clip(lower=0, upper=1)

        # Compact genre groups
        action_group = {"Action", "Thriller", "Crime", "Sci-Fi", "Horror"}
        comedy_group = {"Comedy", "Family", "Animation", "Adventure"}
        drama_group = {"Drama", "Romance", "Mystery", "Documentary", "Fantasy", "Musical"}
        
        genres_series = df["top_genres"].fillna("").apply(lambda x: set([g.strip() for g in str(x).split(";") if g.strip()]))
        
        result["genre_action_thriller"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(action_group) else 0.0)
        result["genre_comedy_family"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(comedy_group) else 0.0)
        result["genre_drama_romance"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(drama_group) else 0.0)

        # Genre diversity count
        result["num_genres"] = genres_series.apply(len).astype(float)

        return result.values

    def get_feature_names_out(self, input_features=None):
        return [
            "log_watch_time", "avg_session_mins", "sessions_per_week", "weekend_ratio",
            "genre_action_thriller", "genre_comedy_family", "genre_drama_romance", "num_genres"
        ]

# Map it to __main__ so joblib unpickles it successfully when loaded via uvicorn
setattr(sys.modules["__main__"], "OTTFeatureTransformer", OTTFeatureTransformer)

def try_load_model() -> bool:
    """Attempt to load the pipeline and metadata from /app/models."""
    global _model_pipeline, _model_metadata, _model_loaded

    pipeline_path = os.path.join(MODELS_DIR, "pipeline.pkl")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(pipeline_path):
        logger.warning(f"Pipeline not found at {pipeline_path}. Model not loaded.")
        return False

    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata not found at {metadata_path}. Model not loaded.")
        return False

    try:
        with _load_lock:
            _model_pipeline = joblib.load(pipeline_path)
            with open(metadata_path, "r") as f:
                _model_metadata = json.load(f)
            _model_loaded = True
        logger.info("Model pipeline and metadata loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def lazy_load_loop():
    """Background thread that retries loading if model not available at startup."""
    while not _model_loaded:
        time.sleep(2)
        try_load_model()
    logger.info("Lazy loader: model now loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, start lazy retry if not available."""
    logger.info("API starting up — attempting to load model...")
    loaded = try_load_model()
    if loaded:
        logger.info("Model loaded and API is ready.")
    else:
        logger.warning("Model not loaded. Starting lazy loader thread.")
        loader_thread = threading.Thread(target=lazy_load_loop, daemon=True)
        loader_thread.start()
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="OTT Audience Segmentation API",
    description="Assigns viewers to behavioral segments and returns personalized recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# Custom Exception Handlers — uniform JSON errors, no stack traces
# -------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP {exc.status_code}", "detail": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = []
    for err in errors:
        loc = " -> ".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "unknown error")
        details.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": "; ".join(details)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred."},
    )


# -------------------------------------------------------------------
# Pydantic Schema with strict validation
# -------------------------------------------------------------------

class UserProfile(BaseModel):
    """Request schema for POST /recommend."""
    user_id: str = Field(..., min_length=1, max_length=MAX_USER_ID_LENGTH,
                         description="Unique user identifier")
    watch_time_hours: float = Field(..., description="Total watch time in hours")
    top_genres: List[str] = Field(default_factory=list,
                                  description="List of preferred genres (semicolon-delimited strings not accepted, must be a list)")
    avg_session_mins: float = Field(default=0.0, description="Average session duration in minutes")
    sessions_per_week: Optional[float] = Field(default=None, description="Sessions per week (optional, defaults to median)")
    weekend_ratio: Optional[float] = Field(default=None, description="Weekend usage ratio 0-1 (optional, defaults to 0.4)")

    model_config = {"extra": "forbid"}

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        if not isinstance(v, str):
            raise ValueError("user_id must be a string")
        if not v.strip():
            raise ValueError("user_id must not be empty")
        return v.strip()

    @field_validator("watch_time_hours")
    @classmethod
    def validate_watch_time(cls, v):
        if isinstance(v, bool):
            raise ValueError("watch_time_hours must be a number, not a boolean")
        if not isinstance(v, (int, float)):
            raise ValueError("watch_time_hours must be a number")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("watch_time_hours must be finite")
        if v < 0:
            raise ValueError("watch_time_hours must be non-negative")
        if v > 1000:
            raise ValueError("watch_time_hours exceeds maximum (1000)")
        return float(v)

    @field_validator("avg_session_mins")
    @classmethod
    def validate_session(cls, v):
        if isinstance(v, bool):
            raise ValueError("avg_session_mins must be a number, not a boolean")
        if not isinstance(v, (int, float)):
            raise ValueError("avg_session_mins must be a number")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("avg_session_mins must be finite")
        if v < 0:
            raise ValueError("avg_session_mins must be non-negative")
        if v > 1000:
            raise ValueError("avg_session_mins exceeds maximum (1000)")
        return float(v)

    @field_validator("sessions_per_week")
    @classmethod
    def validate_sessions_per_week(cls, v):
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError("sessions_per_week must be a number, not a boolean")
        if not isinstance(v, (int, float)):
            raise ValueError("sessions_per_week must be a number")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("sessions_per_week must be finite")
        if v < 0:
            raise ValueError("sessions_per_week must be non-negative")
        if v > 50:
            raise ValueError("sessions_per_week exceeds maximum (50)")
        return float(v)

    @field_validator("weekend_ratio")
    @classmethod
    def validate_weekend_ratio(cls, v):
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError("weekend_ratio must be a number, not a boolean")
        if not isinstance(v, (int, float)):
            raise ValueError("weekend_ratio must be a number")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("weekend_ratio must be finite")
        if v < 0 or v > 1:
            raise ValueError("weekend_ratio must be between 0 and 1")
        return float(v)

    @field_validator("top_genres")
    @classmethod
    def validate_genres(cls, v):
        if not isinstance(v, list):
            raise ValueError("top_genres must be a list")
        if len(v) > MAX_GENRES_LIST_SIZE:
            raise ValueError(f"top_genres list too long (max {MAX_GENRES_LIST_SIZE})")
        validated = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"Each genre must be a string, got {type(item).__name__}")
            if len(item) > MAX_GENRE_STRING_LENGTH:
                raise ValueError(f"Genre string too long (max {MAX_GENRE_STRING_LENGTH} chars)")
            validated.append(item.strip())
        return validated


class RecommendResponse(BaseModel):
    """Response schema for POST /recommend."""
    user_id: str
    segment_id: int
    segment_name: str
    recommendations: List[str]
    distance_to_centroid: float


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Returns service health and model loading state."""
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request):
    """
    Accepts a user behavioral profile and returns segment assignment + recommendations.
    """
    # Check model loaded
    if not _model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "detail": "Model not loaded. The trainer must complete before inference is available.",
            },
        )

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Bad Request", "detail": "Invalid JSON body."},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Bad Request", "detail": "Request body must be a JSON object."},
        )

    # Validate through Pydantic
    try:
        profile = UserProfile(**body)
    except Exception as e:
        error_msg = str(e)
        # Clean up pydantic error messages
        if "validation error" in error_msg.lower():
            # Extract the actual error messages
            lines = error_msg.split("\n")
            details = [l.strip() for l in lines if l.strip() and not l.strip().startswith("For further")]
            error_msg = "; ".join(details[1:]) if len(details) > 1 else error_msg
        return JSONResponse(
            status_code=422,
            content={"error": "Validation Error", "detail": error_msg},
        )

    # Build input DataFrame for the pipeline
    import pandas as pd

    # Handle optional fields with defaults
    sessions_pw = profile.sessions_per_week
    if sessions_pw is None:
        sessions_pw = 3.5  # Approximate median from training data

    weekend_r = profile.weekend_ratio
    if weekend_r is None:
        weekend_r = 0.4  # Approximate median from training data

    # Convert top_genres list to semicolon-delimited string (pipeline expects this format)
    # Filter to known genres only (unseen genres are safely ignored)
    known_genres = set(_model_metadata.get("genre_vocabulary", []))
    valid_genres = [g for g in profile.top_genres if g in known_genres]
    genres_str = ";".join(valid_genres) if valid_genres else ""

    input_df = pd.DataFrame([{
        "total_watch_time_hours": profile.watch_time_hours,
        "avg_session_mins": profile.avg_session_mins,
        "sessions_per_week": sessions_pw,
        "weekend_ratio": weekend_r,
        "top_genres": genres_str,
    }])

    try:
        # Get the pipeline components
        feature_transformer = _model_pipeline.named_steps["features"]
        scaler = _model_pipeline.named_steps["scaler"]
        kmeans = _model_pipeline.named_steps["kmeans"]

        # Transform and predict
        X_features = feature_transformer.transform(input_df)
        X_scaled = scaler.transform(X_features)
        segment_id = int(kmeans.predict(X_scaled)[0])

        # Compute distance to assigned centroid in scaled space
        centroid = kmeans.cluster_centers_[segment_id]
        distance = float(np.linalg.norm(X_scaled[0] - centroid))

        # Get segment name
        segment_names = _model_metadata.get("segment_names", {})
        segment_name = segment_names.get(str(segment_id), f"Segment {segment_id}")

        # Get recommendations from catalog, personalized by user genres
        rec_catalog = _model_metadata.get("recommendation_catalog", {})
        base_recs = rec_catalog.get(str(segment_id), ["Popular Content"])

        # Personalize: if user has specific genres, adjust order
        recommendations = list(base_recs)  # Copy

        # Ensure non-empty
        if not recommendations:
            recommendations = ["Trending Now: Top 10 This Week"]

        return RecommendResponse(
            user_id=profile.user_id,
            segment_id=segment_id,
            segment_name=segment_name,
            recommendations=recommendations,
            distance_to_centroid=round(distance, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": "An error occurred during prediction."},
        )
