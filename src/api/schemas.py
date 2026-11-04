"""
Pydantic schemas and data validation models for the YouTube Sentiment Analysis API.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class SingleCommentRequest(BaseModel):
    """Schema for single comment prediction."""
    text: str = Field(..., min_length=1, max_length=5000, description="The comment text to analyze")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This tutorial is fantastic! Clear, concise and well-paced. 🔥"
            }
        }
    }


class BatchCommentRequest(BaseModel):
    """Schema for batch comment analysis."""
    comments: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of raw comment strings to analyze in batch"
    )

    @field_validator("comments")
    def check_non_empty_comments(cls, v):
        if not v:
            raise ValueError("The 'comments' list cannot be empty.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "comments": [
                    "This tutorial is fantastic! Clear, concise and well-paced. 🔥",
                    "Horrible video, didn't explain anything properly 👎",
                    "What version of Python is being used here?"
                ]
            }
        }
    }


class CommentPrediction(BaseModel):
    """Individual comment prediction result."""
    text: str
    cleaned_text: str
    sentiment: str = Field(..., description="Sentiment category: 'positive', 'neutral', or 'negative'")
    label: int = Field(..., description="Numeric label: 1 (positive), 0 (neutral), -1 (negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of top class")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across negative, neutral, and positive"
    )


class SentimentStatistics(BaseModel):
    """Aggregated statistical breakdown of the batch."""
    total_comments: int
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    sentiment_score: float = Field(
        ...,
        description="Normalized score from -1.0 (strongly negative) to +1.0 (strongly positive)"
    )
    average_confidence: float
    predominant_sentiment: str


class BatchPredictionResponse(BaseModel):
    """Full batch response including per-comment predictions and statistics."""
    status: str = "success"
    processing_time_ms: float
    statistics: SentimentStatistics
    predictions: List[CommentPrediction]


class HealthResponse(BaseModel):
    """System health check and runtime status."""
    status: str
    model_loaded: bool
    model_name: str
    version: str
    uptime_seconds: float
    supported_labels: List[str]
