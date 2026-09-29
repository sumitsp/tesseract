"""Simple coefficient-based evidence for linear TF-IDF models."""

from __future__ import annotations

from src.features.ocr_features import top_evidence
from src.models.classifiers import FlatClassifier


def explain_page(
    model: FlatClassifier,
    ocr_text: str,
    predicted_class: str,
    limit: int = 5,
) -> list[str]:
    """Prefer pattern evidence; add top TF-IDF coefficients when available."""
    evidence = top_evidence(ocr_text, limit=limit)
    if model.bundle is None or not hasattr(model.clf, "coef_"):
        return evidence
    try:
        classes = list(model.clf.classes_)
        if predicted_class not in classes:
            return evidence
        ci = classes.index(predicted_class)
        coef = model.clf.coef_
        # One-vs-rest or multinomial
        row = coef[ci] if coef.ndim == 2 else coef
        X = model.bundle.transform([ocr_text])
        # Contribution ≈ x_i * w_i for active features
        x = X.toarray()[0]
        scores = x * row
        idx = scores.argsort()[::-1]
        names = model.bundle.feature_names
        for i in idx[:limit]:
            if scores[i] <= 0:
                break
            name = names[i]
            if name.startswith("n:"):
                continue
            token = name.split(":", 1)[-1]
            if len(token) < 3:
                continue
            tag = f"tfidf:{token}"
            if tag not in evidence:
                evidence.append(tag)
            if len(evidence) >= limit:
                break
    except Exception:
        pass
    return evidence[:limit]
