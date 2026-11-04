"""
Unit and integration tests for ML sentiment model performance, latency, and edge cases.
"""

import os
import time
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.utils.text_cleaning import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model2.joblib")
VEC_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer2.joblib")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Loads the model2 and vectorizer2 into a callable predict wrapper."""
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    assert os.path.exists(VEC_PATH), f"Vectorizer not found at {VEC_PATH}"
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)

    class ModelWrapper:
        def __init__(self, m, v):
            self.m = m
            self.v = v
            self.classes_ = m.classes_

        def predict(self, texts):
            features = self.v.transform(texts)
            return self.m.predict(features)

        def predict_proba(self, texts):
            features = self.v.transform(texts)
            return self.m.predict_proba(features)

    return ModelWrapper(model, vec)


def test_model_artifact_exists_and_loads(loaded_pipeline):
    """Verifies that the joblib artifact loads and contains expected sklearn steps."""
    assert loaded_pipeline is not None
    assert hasattr(loaded_pipeline, "predict")
    assert hasattr(loaded_pipeline, "predict_proba")
    assert hasattr(loaded_pipeline, "classes_")
    assert set(loaded_pipeline.classes_) == {-1, 0, 1}


def test_test_set_accuracy_criterion(loaded_pipeline):
    """Validates that test accuracy meets the TP criterion (>= 80%)."""
    assert os.path.exists(TEST_DATA_PATH), f"Test set not found at {TEST_DATA_PATH}"
    test_df = pd.read_csv(TEST_DATA_PATH).dropna(subset=["text", "label"])

    # Sample 1000 comments for fast test execution
    sample_df = test_df.sample(n=min(1000, len(test_df)), random_state=42)
    X = sample_df["text"].astype(str).tolist()
    y = sample_df["label"].astype(int).to_numpy()

    y_pred = loaded_pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n[Test Set Sample Accuracy]: {acc*100:.2f}%")
    assert acc >= 0.80, f"Accuracy {acc} is below the 80% threshold required by TP3."


def test_inference_latency_criterion(loaded_pipeline):
    """Validates that batch latency for 50 comments is strictly under 100ms."""
    batch_50 = ["This is an amazing and insightful tutorial, learned a lot!" for _ in range(50)]

    # Warm-up run
    _ = loaded_pipeline.predict_proba(batch_50)

    # Benchmark run
    t0 = time.perf_counter()
    probas = loaded_pipeline.predict_proba(batch_50)
    duration_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[Batch-50 Latency]: {duration_ms:.2f} ms")
    assert duration_ms < 100.0, f"Inference took {duration_ms}ms, exceeding 100ms limit."
    assert probas.shape == (50, 3)


def test_edge_cases(loaded_pipeline):
    """Tests diverse edge cases: emojis, very long text, minimal strings, neutral queries."""
    # 1. Emoji only
    pred_pos_emoji = loaded_pipeline.predict([clean_text("🔥❤️🎉")])[0]
    assert pred_pos_emoji == 1

    pred_neg_emoji = loaded_pipeline.predict([clean_text("🤮👎💩")])[0]
    assert pred_neg_emoji == -1

    # 2. Very long text (3000 chars)
    long_text = "really great tutorial and explanation! " * 80
    pred_long = loaded_pipeline.predict([clean_text(long_text)])[0]
    assert pred_long == 1

    # 3. Neutral question
    neutral_text = "What timestamp does the code start?"
    prob_neutral = loaded_pipeline.predict_proba([clean_text(neutral_text)])[0]
    # Check probabilities sum to ~ 1.0
    assert np.isclose(np.sum(prob_neutral), 1.0)
