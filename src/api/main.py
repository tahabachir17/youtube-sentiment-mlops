"""
FastAPI REST application for YouTube Sentiment Analysis.
Serves sentiment inference endpoints, batch processing, monitoring, and CORS for Chrome extensions.
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.schemas import (
    SingleCommentRequest,
    BatchCommentRequest,
    CommentPrediction,
    SentimentStatistics,
    BatchPredictionResponse,
    HealthResponse
)
from src.utils.text_cleaning import clean_text, label_to_sentiment_name

# Setup logging
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "api.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sentiment_api")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_pipeline.joblib")
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads ML model artifacts once at startup
    and shares them across all requests.
    """
    logger.info("Starting up YouTube Sentiment API...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}! Please run train.py first.")
        app.state.pipeline = None
    else:
        try:
            app.state.pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Successfully loaded ML pipeline from: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            app.state.pipeline = None
    yield
    logger.info("Shutting down YouTube Sentiment API...")


app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="Production-ready REST API for real-time sentiment analysis of YouTube comments using MLOps best practices.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS: Allow all origins (specifically Chrome extensions: chrome-extension://*)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with service overview and link to documentation."""
    return {
        "service": "YouTube Sentiment Analysis API",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring uptime and ML model readiness.
    """
    model_loaded = app.state.pipeline is not None
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name="TF-IDF + LogisticRegression (Balanced)",
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        supported_labels=["negative (-1)", "neutral (0)", "positive (1)"]
    )


def compute_prediction_for_text(pipeline, raw_text: str) -> CommentPrediction:
    """Helper to run inference and build structured prediction object."""
    cleaned = clean_text(raw_text)
    # Default fallback for completely empty strings
    eval_text = cleaned if len(cleaned.strip()) > 0 else "neutral"

    # Prediction & probabilities
    pred_label = int(pipeline.predict([eval_text])[0])
    probas = pipeline.predict_proba([eval_text])[0]
    classes = pipeline.classes_

    # Map classes to sentiment probabilities
    prob_dict = {
        "negative": 0.0,
        "neutral": 0.0,
        "positive": 0.0
    }
    for cls_label, prob in zip(classes, probas):
        name = label_to_sentiment_name(cls_label)
        prob_dict[name] = round(float(prob), 4)

    confidence = float(np.max(probas))
    sentiment_str = label_to_sentiment_name(pred_label)

    return CommentPrediction(
        text=raw_text,
        cleaned_text=cleaned,
        sentiment=sentiment_str,
        label=pred_label,
        confidence=round(confidence, 4),
        probabilities=prob_dict
    )


@app.post("/predict", response_model=CommentPrediction, tags=["Inference"])
async def predict_single(request: SingleCommentRequest):
    """
    Analyzes sentiment of a single comment.
    """
    if app.state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train or provide model artifacts."
        )

    t0 = time.perf_counter()
    result = compute_prediction_for_text(app.state.pipeline, request.text)
    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Single inference processed in {duration_ms:.2f}ms | Result: {result.sentiment}")
    return result


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchCommentRequest):
    """
    Analyzes a batch of comments and returns predictions, probabilities,
    and aggregated sentiment statistics.
    """
    pipeline = app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check /health for status."
        )

    raw_comments = request.comments
    if not raw_comments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The comments list cannot be empty."
        )

    t0 = time.perf_counter()

    # Preprocess all comments
    cleaned_comments = [clean_text(c) for c in raw_comments]
    eval_comments = [c if len(c.strip()) > 0 else "neutral" for c in cleaned_comments]

    # Vectorized batch inference
    preds = pipeline.predict(eval_comments)
    probas = pipeline.predict_proba(eval_comments)
    classes = pipeline.classes_

    predictions: List[CommentPrediction] = []
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    confidence_sum = 0.0

    for raw, clean, pred, prob in zip(raw_comments, cleaned_comments, preds, probas):
        pred_label = int(pred)
        sentiment_name = label_to_sentiment_name(pred_label)
        confidence = float(np.max(prob))
        confidence_sum += confidence

        prob_dict = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for cls_label, p in zip(classes, prob):
            prob_dict[label_to_sentiment_name(cls_label)] = round(float(p), 4)

        if pred_label == 1:
            positive_count += 1
        elif pred_label == -1:
            negative_count += 1
        else:
            neutral_count += 1

        predictions.append(
            CommentPrediction(
                text=raw,
                cleaned_text=clean,
                sentiment=sentiment_name,
                label=pred_label,
                confidence=round(confidence, 4),
                probabilities=prob_dict
            )
        )

    total = len(predictions)
    pos_pct = round((positive_count / total) * 100, 2)
    neu_pct = round((neutral_count / total) * 100, 2)
    neg_pct = round((negative_count / total) * 100, 2)
    avg_conf = round(confidence_sum / total, 4)

    # Sentiment score: weighted average of (+1 * pos + 0 * neu + -1 * neg) / total
    sentiment_score = round((positive_count - negative_count) / total, 4)

    if positive_count >= neutral_count and positive_count >= negative_count:
        predominant = "positive"
    elif negative_count >= positive_count and negative_count >= neutral_count:
        predominant = "negative"
    else:
        predominant = "neutral"

    stats = SentimentStatistics(
        total_comments=total,
        positive_count=positive_count,
        neutral_count=neutral_count,
        negative_count=negative_count,
        positive_pct=pos_pct,
        neutral_pct=neu_pct,
        negative_pct=neg_pct,
        sentiment_score=sentiment_score,
        average_confidence=avg_conf,
        predominant_sentiment=predominant
    )

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"Batch of {total} comments processed in {duration_ms}ms "
        f"[Pos: {positive_count}, Neu: {neutral_count}, Neg: {negative_count}]"
    )

    return BatchPredictionResponse(
        status="success",
        processing_time_ms=duration_ms,
        statistics=stats,
        predictions=predictions
    )
