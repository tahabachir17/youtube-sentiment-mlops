"""
Model training, experiment tracking, hyperparameter tuning, and evaluation script.
Trains and compares multiple classifiers for sentiment analysis on Reddit/YouTube comments.
Saves the best model, vectorizer, confusion matrix plot, and metrics report.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def load_data():
    """Loads preprocessed train and test datasets."""
    train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["text", "label"])
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["label"].astype(int).to_numpy()

    X_test = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    print(f"[+] Loaded {len(X_train)} training examples and {len(X_test)} test examples.")
    return X_train, y_train, X_test, y_test


def evaluate_classifier(name, model, X_test_vec, y_test):
    """Evaluates classifier accuracy and macro F1 score."""
    start_t = time.perf_counter()
    y_pred = model.predict(X_test_vec)
    inference_time = (time.perf_counter() - start_t) * 1000

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    print(f"| {name:<30} | Acc: {acc*100:6.2f}% | Macro F1: {f1:0.4f} | Inf. Time: {inference_time:6.1f}ms |")
    return {
        "name": name,
        "model": model,
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec)
    }


def benchmark_latency(model, vectorizer, X_test, batch_size=50, repetitions=50):
    """Measures inference latency for a batch of 50 comments."""
    sample_batch = X_test[:batch_size]
    if len(sample_batch) < batch_size:
        sample_batch = (sample_batch * ((batch_size // len(sample_batch)) + 1))[:batch_size]

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        # Transform + predict_proba
        feats = vectorizer.transform(sample_batch)
        _ = model.predict_proba(feats)
        times.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(times))
    p95_latency = float(np.percentile(times, 95))
    print(f"\n[+] Latency Benchmark (Batch of {batch_size} comments):")
    print(f"    Average: {avg_latency:.2f} ms")
    print(f"    95th percentile: {p95_latency:.2f} ms")
    print(f"    Criteria (< 100ms): {'PASSED [OK]' if avg_latency < 100 else 'FAILED'}")
    return avg_latency, p95_latency


def train_and_evaluate():
    """Main training, model comparison, tuning, and export pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    print("\n[*] Initializing and fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[+] TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} features")

    print("\n" + "=" * 80)
    print("                      ALGORITHM COMPARISON")
    print("=" * 80)

    candidates = {
        "Logistic Regression (lbfgs, C=1.0)": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Logistic Regression (balanced, C=1.5)": LogisticRegression(
            C=1.5, max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"
        ),
        "Support Vector Machine (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=42, dual="auto", max_iter=2000),
            cv=3
        ),
        "Random Forest (n_estimators=100)": RandomForestClassifier(
            n_estimators=100, max_depth=30, n_jobs=-1, random_state=42
        ),
        "SGDClassifier (log_loss, alpha=1e-5)": SGDClassifier(
            loss="log_loss", alpha=1e-5, max_iter=1000, random_state=42
        ),
        "Complement Naive Bayes": ComplementNB(alpha=0.5)
    }

    results = []
    for name, clf in candidates.items():
        clf.fit(X_train_vec, y_train)
        res = evaluate_classifier(name, clf, X_test_vec, y_test)
        results.append(res)

    print("=" * 80)

    # Hyperparameter tuning on best performing candidate family (Logistic Regression)
    print("\n[*] Fine-tuning best model with GridSearchCV...")
    param_grid = {
        "C": [0.8, 1.2, 1.5, 2.0, 3.0],
        "class_weight": [None, "balanced"],
        "solver": ["lbfgs", "saga"]
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_vec, y_train)

    best_model = grid.best_estimator_
    print(f"[+] Best parameters found: {grid.best_params_}")
    print(f"[+] Best cross-validation accuracy: {grid.best_score_*100:.2f}%")

    # Final evaluation on holdout test set
    y_pred = best_model.predict(X_test_vec)
    y_proba = best_model.predict_proba(X_test_vec)
    final_acc = accuracy_score(y_test, y_pred)
    labels = [-1, 0, 1]
    target_names = ["Negative (-1)", "Neutral (0)", "Positive (1)"]

    clf_report = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names, output_dict=True
    )
    clf_report_text = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names
    )

    print("\n" + "=" * 50)
    print("         FINAL CLASSIFICATION REPORT")
    print("=" * 50)
    print(clf_report_text)
    print(f"Overall Accuracy: {final_acc*100:.2f}%")
    print("=" * 50)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix (Accuracy: {final_acc*100:.2f}%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to: {cm_path}")

    # Latency test
    avg_lat, p95_lat = benchmark_latency(best_model, vectorizer, X_test, batch_size=50)

    # Save artifacts
    print("\n[*] Persisting trained models to disk...")
    # 1. Save separate components
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, "sentiment_model.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(best_model, model_path)

    # 2. Save full unified sklearn pipeline for single-artifact deployment
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])
    pipeline_path = os.path.join(MODELS_DIR, "sentiment_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    print(f"[+] Saved vectorizer: {vectorizer_path}")
    print(f"[+] Saved classifier: {model_path}")
    print(f"[+] Saved complete pipeline: {pipeline_path}")

    # Save metrics JSON
    metrics_summary = {
        "best_algorithm": "Logistic Regression",
        "best_hyperparameters": grid.best_params_,
        "cross_val_accuracy": float(grid.best_score_),
        "test_accuracy": float(final_acc),
        "target_accuracy_met": bool(final_acc >= 0.80),
        "f1_per_class": {
            "negative": float(clf_report["Negative (-1)"]["f1-score"]),
            "neutral": float(clf_report["Neutral (0)"]["f1-score"]),
            "positive": float(clf_report["Positive (1)"]["f1-score"])
        },
        "macro_avg_f1": float(clf_report["macro avg"]["f1-score"]),
        "weighted_avg_f1": float(clf_report["weighted avg"]["f1-score"]),
        "latency_batch_50_avg_ms": avg_lat,
        "latency_batch_50_p95_ms": p95_lat,
        "latency_criteria_met": bool(avg_lat < 100.0),
        "comparison_results": [
            {k: v for k, v in r.items() if k != "model"} for r in results
        ]
    }

    metrics_json_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved metrics summary to: {metrics_json_path}")

    # Save Markdown Evaluation Report
    eval_md_path = os.path.join(REPORTS_DIR, "evaluation_report.md")
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Rapport d'Évaluation du Modèle de Sentiment

## Métriques Globales
- **Accuracy Test** : {final_acc*100:.2f}% (Objectif TP : >= 80% -> {'Validé [OK]' if final_acc >= 0.8 else 'Non atteint'})
- **F1-Score Macro** : {clf_report['macro avg']['f1-score']:.4f}
- **F1-Score Négatif (-1)** : {clf_report['Negative (-1)']['f1-score']:.4f}
- **F1-Score Neutre (0)** : {clf_report['Neutral (0)']['f1-score']:.4f}
- **F1-Score Positif (1)** : {clf_report['Positive (1)']['f1-score']:.4f}
- **Latence d'inférence (batch 50)** : {avg_lat:.2f} ms (Objectif TP : < 100ms -> {'Validé [OK]' if avg_lat < 100 else 'Non atteint'})

## Meilleurs Hyperparamètres
```json
{json.dumps(grid.best_params_, indent=2)}
```

## Rapport de Classification Détaillé
```
{clf_report_text}
```

## Matrice de Confusion
![Matrice de Confusion](confusion_matrix.png)
""")
    print(f"[+] Saved evaluation markdown report to: {eval_md_path}")
    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
