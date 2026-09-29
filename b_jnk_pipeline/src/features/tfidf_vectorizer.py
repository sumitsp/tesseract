"""TF-IDF + numeric feature vectorizer for OCR pages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.features.ocr_features import FEATURE_NAMES, feature_vector
from src.preprocessing.text_normalize import normalize_for_ngrams


@dataclass
class TfidfFeatureBundle:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    numeric_scaler: StandardScaler
    feature_names: list[str]

    def transform(self, texts: list[str]) -> sparse.spmatrix:
        normed = [normalize_for_ngrams(t) for t in texts]
        Xw = self.word_vectorizer.transform(normed)
        Xc = self.char_vectorizer.transform(normed)
        Xn = np.vstack([feature_vector(t) for t in texts])
        Xn = self.numeric_scaler.transform(Xn)
        return sparse.hstack([Xw, Xc, sparse.csr_matrix(Xn)], format="csr")


def fit_tfidf_bundle(
    texts: list[str],
    *,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_max_features: int = 40000,
    char_max_features: int = 40000,
    min_df: int = 1,
) -> TfidfFeatureBundle:
    normed = [normalize_for_ngrams(t) for t in texts]
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=word_max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=char_max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    word.fit(normed)
    char.fit(normed)
    Xn = np.vstack([feature_vector(t) for t in texts])
    scaler = StandardScaler()
    scaler.fit(Xn)
    names = (
        [f"w:{t}" for t in word.get_feature_names_out()]
        + [f"c:{t}" for t in char.get_feature_names_out()]
        + [f"n:{n}" for n in FEATURE_NAMES]
    )
    return TfidfFeatureBundle(
        word_vectorizer=word,
        char_vectorizer=char,
        numeric_scaler=scaler,
        feature_names=names,
    )
