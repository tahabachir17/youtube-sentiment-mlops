"""
Evaluation and diagnostic script for trained sentiment model.
Tests saved artifacts on validation sets and diverse edge-case inputs.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Ensure UTF-8 stdout on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.text_cleaning import clean_text, label_to_sentiment_name

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_pipeline.joblib")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")


def run_edge_case_tests(pipeline):
    """Tests the model on various synthetic and realistic edge cases."""
    test_cases = [
        "This video is absolutely incredible, learned so much! 🔥❤️",
        "Terrible tutorial, waste of time and complete nonsense 👎😡",
        "The audio quality is okay, video was uploaded yesterday.",
        "Sooooo goooood loved every single second of it!!!!!",
        "horrible worst explanation ever 0/10",
        "Meh, it's alright I guess. Nothing special.",
        "https://scam-link.com check this out for free money @baduser",
        "👍🎉💯",
        "🤮💩",
        "Could you please share the GitHub repository link?",
        "I was skeptical at first, but this completely changed my mind. Brilliant work!"
    ]

    print("\n" + "=" * 80)
    print("                  MODEL EDGE CASES & INFERENCE TEST")
    print("=" * 80)

    cleaned = [clean_text(t) for t in test_cases]
    preds = pipeline.predict(cleaned)
    probas = pipeline.predict_proba(cleaned)
    classes = pipeline.classes_

    for raw, clean, pred, prob in zip(test_cases, cleaned, preds, probas):
        conf = float(np_max := max(prob))
        sent = label_to_sentiment_name(pred)
        print(f"Raw:        {raw}")
        print(f"Cleaned:    '{clean}'")
        print(f"Prediction: {sent.upper():<8} (Label: {pred:+d}) | Confidence: {conf*100:5.1f}%")
        print("-" * 80)


def evaluate_saved_model():
    """Evaluates the saved pipeline on the holdout test set."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Trained model not found at {PIPELINE_PATH}. Run train.py first.")

    pipeline = joblib.load(PIPELINE_PATH)
    print(f"[+] Loaded pipeline from: {PIPELINE_PATH}")

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    X_test = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Holdout Test Set Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, digits=4))

    run_edge_case_tests(pipeline)


if __name__ == "__main__":
    evaluate_saved_model()
