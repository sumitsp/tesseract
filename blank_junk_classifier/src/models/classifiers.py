"""TF-IDF baseline and embedding classifiers (KEEP / BLANK / JUNK)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.features.tfidf_vectorizer import TfidfFeatureBundle, fit_tfidf_bundle
from src.preprocessing.dataset import PageRecord
from src.preprocessing.taxonomy import coarse_bucket

ClassifierKind = Literal["logistic_regression", "linear_svc"]


def _class_weight(y: list[str], keep_cost: float = 50.0) -> dict[str, float]:
    """Up-weight KEEP so KEEP→JUNK/BLANK errors are costly."""
    weights: dict[str, float] = {}
    for c in set(y):
        weights[c] = float(keep_cost) if c == "KEEP" else 1.0
    return weights


def _flags_from_rows(rows: list[PageRecord]) -> list[str]:
    flags: list[str] = []
    for r in rows:
        flag = r.flag
        if flag is None:
            raise ValueError(f"{r.page_id}: REVIEW rows cannot be training targets")
        flags.append(flag)
    return flags


@dataclass
class FlatClassifier:
    name: str
    labels: list[str]
    bundle: TfidfFeatureBundle | None = None
    clf: Any = None
    embedding_model_name: str | None = None
    embedder: Any = None
    extra: dict = field(default_factory=dict)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._transform(texts)
        return self.clf.predict_proba(X)

    def predict_labels(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.labels[int(i)] for i in np.argmax(proba, axis=1)]

    def _transform(self, texts: list[str]):
        if self.bundle is not None:
            return self.bundle.transform(texts)
        assert self.embedder is not None
        return np.asarray(self.embedder.encode(texts, show_progress_bar=False))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FlatClassifier":
        return joblib.load(path)


def train_tfidf_flat(
    train_rows: list[PageRecord],
    *,
    classifier: ClassifierKind = "logistic_regression",
    retain_cost: float = 50.0,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_max_features: int = 40000,
    char_max_features: int = 40000,
    min_df: int = 1,
) -> FlatClassifier:
    texts = [r.ocr_text for r in train_rows]
    y = _flags_from_rows(train_rows)
    labels = sorted(set(y))
    if len(labels) < 2:
        raise ValueError(
            f"Need ≥2 flags to train; got {labels}. "
            "Dataset still lacks KEEP / BLANK / JUNK diversity."
        )
    bundle = fit_tfidf_bundle(
        texts,
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        min_df=min_df,
    )
    X = bundle.transform(texts)
    weights = _class_weight(y, retain_cost)
    if classifier == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=weights,
            solver="lbfgs",
        )
        clf.fit(X, y)
    elif classifier == "linear_svc":
        base = LinearSVC(class_weight=weights, max_iter=5000)
        clf = CalibratedClassifierCV(base, cv=min(3, _min_class_count(y)))
        clf.fit(X, y)
    else:
        raise ValueError(classifier)
    return FlatClassifier(
        name=f"tfidf_{classifier}",
        labels=list(clf.classes_),
        bundle=bundle,
        clf=clf,
    )


def _min_class_count(y: list[str]) -> int:
    from collections import Counter

    return max(2, min(Counter(y).values()))


def train_embedding_flat(
    train_rows: list[PageRecord],
    *,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    retain_cost: float = 50.0,
) -> FlatClassifier:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers to train the embedding classifier: "
            "pip install sentence-transformers"
        ) from exc

    texts = [r.ocr_text for r in train_rows]
    y = _flags_from_rows(train_rows)
    labels = sorted(set(y))
    if len(labels) < 2:
        raise ValueError(f"Need ≥2 flags to train; got {labels}")

    embedder = SentenceTransformer(model_name)
    X = np.asarray(embedder.encode(texts, show_progress_bar=False))
    weights = _class_weight(y, retain_cost)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=weights,
        solver="lbfgs",
    )
    clf.fit(X, y)
    return FlatClassifier(
        name=f"embed_{model_name.split('/')[-1]}",
        labels=list(clf.classes_),
        clf=clf,
        embedding_model_name=model_name,
        embedder=embedder,
    )


@dataclass
class HierarchicalClassifier:
    """Coarse KEEP vs DROP, then BLANK vs JUNK on the drop branch."""

    name: str
    coarse: FlatClassifier
    fine_drop: FlatClassifier
    labels: list[str]

    def predict_proba_page_types(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        label_set = list(self.labels)
        mat = np.zeros((len(texts), len(label_set)), dtype=float)
        coarse_proba = self.coarse.predict_proba(texts)
        coarse_labels = list(self.coarse.labels)
        c_idx = {c: i for i, c in enumerate(coarse_labels)}

        for i, text in enumerate(texts):
            p_keep = float(coarse_proba[i, c_idx["KEEP"]]) if "KEEP" in c_idx else 0.0
            p_drop = float(coarse_proba[i, c_idx["DROP"]]) if "DROP" in c_idx else 0.0
            if p_keep >= p_drop:
                mat[i, label_set.index("KEEP")] = max(p_keep, 1e-6)
            else:
                fine = self.fine_drop.predict_proba([text])[0]
                for j, lab in enumerate(self.fine_drop.labels):
                    mat[i, label_set.index(lab)] = p_drop * float(fine[j])
            s = mat[i].sum()
            if s > 0:
                mat[i] /= s
            else:
                mat[i, label_set.index("KEEP")] = 1.0
        best = [label_set[int(np.argmax(mat[i]))] for i in range(len(texts))]
        return best, mat


def train_hierarchical(
    train_rows: list[PageRecord],
    *,
    retain_cost: float = 50.0,
) -> HierarchicalClassifier:
    texts: list[str] = []
    y_coarse: list[str] = []
    for r in train_rows:
        bucket = coarse_bucket(r.primary_class)
        if bucket == "REVIEW":
            continue
        texts.append(r.ocr_text)
        y_coarse.append("KEEP" if bucket == "KEEP" else "DROP")

    if len(set(y_coarse)) < 2:
        raise ValueError(
            "Hierarchical coarse stage needs both KEEP and DROP examples; "
            f"got {sorted(set(y_coarse))}"
        )

    bundle = fit_tfidf_bundle(texts)
    X = bundle.transform(texts)
    coarse_clf = LogisticRegression(
        max_iter=2000,
        class_weight={"KEEP": retain_cost, "DROP": 1.0},
    )
    coarse_clf.fit(X, y_coarse)
    coarse = FlatClassifier(
        name="coarse_keep_drop",
        labels=list(coarse_clf.classes_),
        bundle=bundle,
        clf=coarse_clf,
    )

    drop_rows = [
        r for r in train_rows if coarse_bucket(r.primary_class) in {"BLANK", "JUNK"}
    ]
    if len({r.flag for r in drop_rows}) < 2:
        raise ValueError("Hierarchical fine-drop stage needs both BLANK and JUNK")
    fine_drop = train_tfidf_flat(drop_rows, retain_cost=1.0)
    labels = ["KEEP"] + list(fine_drop.labels)
    return HierarchicalClassifier(
        name="hierarchical_tfidf",
        coarse=coarse,
        fine_drop=fine_drop,
        labels=labels,
    )
