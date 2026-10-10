"""Train the scam classifier.

    python -m app.train

Uses the seed data plus any anonymised, consented reports in data/reports.csv
(columns: timestamp,label,text  where label is 'scam' or 'safe').
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from .seed_data import SAFE, SCAM

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parent / "data" / "reports.csv"
MODEL_PATH = ROOT / "model.joblib"


def load_reports() -> tuple[list[str], list[int]]:
    X, y = [], []
    if REPORTS.exists():
        with REPORTS.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") in ("scam", "safe") and row.get("text"):
                    X.append(row["text"])
                    y.append(1 if row["label"] == "scam" else 0)
    return X, y


def build_pipeline():
    # Character n-grams work across English, Tamil script and Tanglish spellings
    # without needing a Tamil tokenizer.
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )


def main() -> None:
    X = SCAM + SAFE
    y = [1] * len(SCAM) + [0] * len(SAFE)
    rx, ry = load_reports()
    X += rx
    y += ry
    print(f"Training on {len(X)} messages ({sum(y)} scam, {len(y) - sum(y)} safe; {len(rx)} from reports)")

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
    print(f"5-fold F1: {scores.mean():.2f} (+/- {scores.std():.2f})  <- optimistic: tiny synthetic seed set")

    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
