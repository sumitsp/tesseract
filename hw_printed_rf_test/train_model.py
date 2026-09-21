#!/usr/bin/env python3
"""Train / rebuild models/data.joblib from models/res2.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = Path(__file__).resolve().parent / "models"
CSV_PATH = MODELS_DIR / "res2.csv"
MODEL_PATH = MODELS_DIR / "data.joblib"
LABELS = {
    "Printed_extended",
    "Handwritten_extended",
    "Mixed_extended",
    "Other_extended",
}


def train(csv_path: Path = CSV_PATH, model_path: Path = MODEL_PATH) -> Path:
    df = pd.read_csv(csv_path)
    if "res" not in df.columns:
        df = pd.read_csv(csv_path, header=None, names=["a1", "a2", "a3", "a4", "a5", "res"])
    df = df[df["res"].isin(LABELS)].copy()
    X = df[["a1", "a2", "a3", "a4", "a5"]].astype(float).to_numpy()
    y = df["res"].astype(str).to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, model_path)
    print(f"Wrote {model_path} ({model_path.stat().st_size} bytes) from {len(df)} rows")
    print(f"Classes: {list(clf.classes_)}")
    return model_path


if __name__ == "__main__":
    train()
