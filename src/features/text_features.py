from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class MovieTitleTextEmbedder:
    """
    Deterministic TF-IDF + TruncatedSVD text embedder for movie titles.
    Runs efficiently on CPU with low memory overhead (~5 MB for 20k titles).
    """

    def __init__(
        self,
        tfidf_max_features: int = 5000,
        svd_components: int = 64,
        random_state: int = 42,
    ):
        self.tfidf_max_features = tfidf_max_features
        self.svd_components = svd_components
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.svd = TruncatedSVD(
            n_components=self.svd_components,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, titles: List[str]) -> "MovieTitleTextEmbedder":
        titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
        tfidf_matrix = self.vectorizer.fit_transform(titles_clean)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        return self

    def transform(self, titles: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("MovieTitleTextEmbedder must be fitted before calling transform.")
        titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
        tfidf_matrix = self.vectorizer.transform(titles_clean)
        embeddings = self.svd.transform(tfidf_matrix)
        return embeddings.astype(np.float32)

    def fit_transform(self, titles: List[str]) -> np.ndarray:
        self.fit(titles)
        return self.transform(titles)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "tfidf_max_features": self.tfidf_max_features,
                "svd_components": self.svd_components,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MovieTitleTextEmbedder":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedder artifact not found at '{path}'")
        data = joblib.load(path)
        embedder = cls(
            tfidf_max_features=data.get("tfidf_max_features", 5000),
            svd_components=data.get("svd_components", 64),
            random_state=data.get("random_state", 42),
        )
        embedder.vectorizer = data["vectorizer"]
        embedder.svd = data["svd"]
        embedder.is_fitted = data.get("is_fitted", True)
        return embedder
