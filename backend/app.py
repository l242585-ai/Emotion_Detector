"""
Emotion Detector — FastAPI Backend
====================================
REST API that serves real-time emotion classification from text.

Endpoints
---------
GET  /                  → Welcome + health summary
GET  /health            → Detailed health check with loaded model info
GET  /models            → List all models with descriptions and load status
POST /predict           → Predict emotion from text
POST /predict/explain   → Predict emotion + per-word LIME-style explainability

Usage
-----
From the project root:

    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Interactive API docs are then available at http://localhost:8000/docs

Member 1 (Annushay Fatima, 24L-2585) — Deliverable 3
"""

import os
import sys
import logging

from fastapi                  import FastAPI, HTTPException
from fastapi.middleware.cors  import CORSMiddleware
from pydantic                 import BaseModel, Field
from typing                   import Optional, List, Dict

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.model_service import ModelService, EMOTION_NAMES, EMOTION_EMOJIS
from backend.explainer     import EmotionExplainer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Emotion Detector API",
    description = (
        "Real-time 6-class emotion classification from English text. "
        "Models: TF-IDF + Logistic Regression (baseline), BiLSTM, DistilBERT. "
        "Deliverable 3 — Track A Application."
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

# CORS — allow all origins so the Streamlit frontend (any host/port) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict to specific origins in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global service objects ────────────────────────────────────────────────────
model_service : ModelService      = ModelService()
explainer     : Optional[EmotionExplainer] = None


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """Load all available trained models when the server starts."""
    global explainer

    logger.info("=" * 60)
    logger.info("EMOTION DETECTOR API — starting up")
    logger.info("=" * 60)

    loaded = model_service.load_all()

    if loaded:
        explainer = EmotionExplainer(model_service)
        logger.info("Models loaded: %s", loaded)
        logger.info("Explainer ready.")
    else:
        logger.warning(
            "No models were loaded. "
            "Train models first (see README.md) then restart the server."
        )
    logger.info("=" * 60)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for /predict and /predict/explain."""
    text: str = Field(
        ...,
        min_length  = 1,
        max_length  = 5000,
        description = "English text to classify (1 – 5 000 characters).",
    )
    model: Optional[str] = Field(
        default     = None,
        description = (
            "Model key: 'baseline', 'bilstm', or 'distilbert'. "
            "Omit to let the API choose the best available model automatically."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text" : "I am so happy and excited about this!",
                "model": "distilbert",
            }
        }


class WordHighlight(BaseModel):
    word  : str
    score : float
    color : str


class PredictResponse(BaseModel):
    emotion      : str
    emotion_id   : int
    emoji        : str
    confidence   : float
    probabilities: Dict[str, float]
    model        : str
    latency_ms   : float


class ExplainResponse(PredictResponse):
    highlights: List[WordHighlight]


# ── Utility ───────────────────────────────────────────────────────────────────

def _resolve_model(requested: Optional[str]) -> str:
    """
    Return the model name to use.
    If *requested* is None, pick the best available model in priority order.
    Raises HTTPException 503 / 400 on bad states.
    """
    available = model_service.get_available_models()

    if not available:
        raise HTTPException(
            status_code = 503,
            detail      = "No models are loaded. Train models first (see README).",
        )

    if requested is None:
        for preferred in ("distilbert", "bilstm", "baseline"):
            if preferred in available:
                return preferred
        return available[0]   # fallback: whatever is loaded

    if requested not in available:
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Model '{requested}' is not available. "
                f"Currently loaded: {available}"
            ),
        )
    return requested


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root() -> dict:
    """Welcome endpoint — confirms the API is running."""
    return {
        "message"         : "🎭 Emotion Detector API is running!",
        "version"         : "1.0.0",
        "docs"            : "/docs",
        "available_models": model_service.get_available_models(),
        "emotions"        : EMOTION_NAMES,
    }


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Detailed health check."""
    available = model_service.get_available_models()
    return {
        "status"            : "healthy" if available else "degraded",
        "models_loaded"     : available,
        "total_models"      : len(available),
        "emotions_supported": EMOTION_NAMES,
    }


@app.get("/models", tags=["Models"])
async def list_models() -> dict:
    """List every model with its description and whether it is currently loaded."""
    info = {
        "baseline": {
            "name"       : "TF-IDF + Logistic Regression",
            "description": "Fast bag-of-words baseline.",
            "loaded"     : "baseline" in model_service.loaded_models,
        },
        "bilstm": {
            "name"       : "BiLSTM",
            "description": "Bidirectional LSTM with learned word embeddings.",
            "loaded"     : "bilstm" in model_service.loaded_models,
        },
        "distilbert": {
            "name"       : "DistilBERT",
            "description": "Fine-tuned DistilBERT transformer (highest accuracy).",
            "loaded"     : "distilbert" in model_service.loaded_models,
        },
    }
    return {"models": info}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify the emotion in a text string.

    Returns the top emotion, its confidence, and the probability
    distribution across all six emotions (sadness, joy, love, anger,
    fear, surprise).
    """
    model_name = _resolve_model(request.model)

    try:
        result = model_service.predict(request.text, model_name)
        return PredictResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post("/predict/explain", response_model=ExplainResponse, tags=["Prediction"])
async def predict_with_explanation(request: PredictRequest) -> ExplainResponse:
    """
    Classify emotion AND return per-word importance scores.

    Words are highlighted by how much removing them changes the
    model's confidence (LIME-inspired leave-one-out perturbation).
    Color values: 'strong_positive', 'moderate_positive', 'negative', 'neutral'.
    """
    if explainer is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Explainer not ready. No models are loaded.",
        )

    model_name = _resolve_model(request.model)

    try:
        exp  = explainer.explain(request.text, model_name)
        pred = exp["prediction"]
        return ExplainResponse(
            emotion       = pred["emotion"],
            emotion_id    = pred["emotion_id"],
            emoji         = pred["emoji"],
            confidence    = pred["confidence"],
            probabilities = pred["probabilities"],
            model         = pred["model"],
            latency_ms    = pred["latency_ms"],
            highlights    = exp["highlights"],
        )
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=f"Explanation error: {exc}")


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info",
    )
