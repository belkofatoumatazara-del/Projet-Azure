"""
Train a binary wine-quality classifier and export a versioned model artifact.

Task:    binary classification  ->  "good wine" (quality >= 7) vs "not good"
Output:  model/model_v<MODEL_VERSION>.pkl   (model + metadata bundle)
         model/schema.json                  (feature contract, reused by API + dispatcher)
         model/metrics.json                  (metrics, copied into the README)

Reproducible by design: fixed seed, pinned requirements.txt, dataset hash recorded.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --- Configuration ------------------------------------------------------------
HERE = Path(__file__).parent
DATA_PATH = HERE / "winequality-red.csv"
MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42
QUALITY_THRESHOLD = 7  # quality >= 7  ->  label 1 ("good")
DATASET_SOURCE = "UCI Wine Quality (red) - mirror: plotly/datasets"


def dataset_hash(path: Path) -> str:
    """SHA-256 of the raw dataset file, for reproducibility tracking."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    np.random.seed(RANDOM_SEED)

    # encoding="utf-8-sig" strips the BOM the mirror ships with
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    feature_names = [c for c in df.columns if c != "quality"]

    X = df[feature_names].values
    y = (df["quality"] >= QUALITY_THRESHOLD).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",  # dataset is imbalanced (~14% "good")
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "positive_rate": round(float(y.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # --- Artifact bundle: model + everything the API needs to serve it --------
    bundle = {
        "model": clf,
        "model_version": MODEL_VERSION,
        "feature_names": feature_names,
        "quality_threshold": QUALITY_THRESHOLD,
        "classes": {"0": "not_good", "1": "good"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_source": DATASET_SOURCE,
        "dataset_sha256_16": dataset_hash(DATA_PATH),
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
    }

    model_path = HERE / f"model_v{MODEL_VERSION}.pkl"
    joblib.dump(bundle, model_path)

    # Schema contract shared with the API (/predict validation) and the
    # dispatcher Function (CSV column validation). Single source of truth.
    schema = {
        "model_version": MODEL_VERSION,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target": "good_wine",
        "threshold": QUALITY_THRESHOLD,
    }
    (HERE / "schema.json").write_text(json.dumps(schema, indent=2))
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
