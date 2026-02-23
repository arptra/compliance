from __future__ import annotations

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TrainingConfig


class TextVectorizers:
    def __init__(self, cfg: TrainingConfig):
        self.word = TfidfVectorizer(
            ngram_range=cfg.vectorizer.word_ngram,
            min_df=cfg.vectorizer.min_df,
            max_df=cfg.vectorizer.max_df,
            max_features=cfg.vectorizer.max_features_word,
            sublinear_tf=True,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=cfg.vectorizer.char_ngram,
            min_df=cfg.vectorizer.min_df,
            max_df=cfg.vectorizer.max_df,
            max_features=cfg.vectorizer.max_features_char,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        xw = self.word.fit_transform(texts)
        xc = self.char.fit_transform(texts)
        return hstack([xw, xc])

    def transform(self, texts):
        xw = self.word.transform(texts)
        xc = self.char.transform(texts)
        return hstack([xw, xc])
