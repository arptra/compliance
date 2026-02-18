from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List

import pandas as pd

from app.stopwords_ru import RU_STOPWORDS

LOGGER = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{8,}\d")
DIGIT_RE = re.compile(r"\d+")
NON_LETTER_RE = re.compile(r"[^a-zа-яё<>\s]", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


class TextPreprocessor:
    def __init__(self, cfg: dict):
        p_cfg = cfg["preprocess"]
        self.use_razdel = bool(p_cfg.get("use_razdel", True))
        self.lemmatize = bool(p_cfg.get("lemmatize", False))
        self.max_token_len = int(p_cfg.get("max_token_len", 30))
        self._tokenize_fn = self._build_tokenizer()
        self._morph = self._build_morph() if self.lemmatize else None

    def _build_tokenizer(self):
        if self.use_razdel:
            try:
                from razdel import tokenize as rz_tokenize

                return lambda text: [t.text for t in rz_tokenize(text)]
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("razdel unavailable (%s), fallback to regex", exc)
        return lambda text: re.findall(r"[a-zа-яё<>]+", text, flags=re.IGNORECASE)

    def _build_morph(self):
        try:
            import pymorphy2

            return pymorphy2.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("pymorphy2 unavailable (%s), lemmatization disabled", exc)
            self.lemmatize = False
            return None

    @staticmethod
    def clean_text(text: str) -> str:
        text = (text or "").lower()
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
        text = PHONE_RE.sub(" ", text)
        text = DIGIT_RE.sub(" <num> ", text)
        text = NON_LETTER_RE.sub(" ", text)
        text = SPACE_RE.sub(" ", text).strip()
        return text

    @lru_cache(maxsize=100_000)
    def _lemma(self, token: str) -> str:
        if not self._morph:
            return token
        return self._morph.parse(token)[0].normal_form

    def tokenize(self, text: str) -> List[str]:
        tokens = self._tokenize_fn(text)
        out = []
        for token in tokens:
            token = token.strip()
            if not token or len(token) > self.max_token_len:
                continue
            if token in RU_STOPWORDS:
                continue
            if self.lemmatize:
                token = self._lemma(token)
            out.append(token)
        return out

    def preprocess_series(self, s: pd.Series) -> List[str]:
        processed: List[str] = []
        for item in s.fillna("").astype(str):
            cleaned = self.clean_text(item)
            tokens = self.tokenize(cleaned)
            processed.append(" ".join(tokens))
        return processed
