from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from app.stopwords_ru import RU_STOPWORDS

LOGGER = logging.getLogger(__name__)

DEFAULT_PREPROCESS = {
    "lemmatize": False,
    "use_razdel": True,
    "max_token_len": 30,
    "min_token_len": 3,
    "remove_placeholders": True,
    "placeholder_patterns": [
        r"<[^>]{1,80}>",
        r"\*{2,}",
        r"(?i)\b(x{2,}|х{2,}|_+|#+)\b",
        r"\b\d{4,}\b",
        r"\b[0-9a-f]{8,}\b",
        r"\b[0-9a-f]{8}-[0-9a-f-]{8,}\b",
        r"(?i)\b(ticket|case|id|uuid|guid)\b\s*[:=]?\s*\w+",
    ],
    "remove_urls_emails_phones": True,
    "normalize_numbers": "remove",
    "number_token": "<num>",
    "keep_cyrillic_latin_only": True,
    "drop_tokens_with_digits": True,
    "drop_tokens_with_underscores": True,
    "drop_repeated_char_tokens": True,
    "extra_stopwords_path": "configs/extra_stopwords.txt",
    "deny_tokens_path": "configs/deny_tokens.txt",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{8,}\d")
DIGIT_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")


class TextPreprocessor:
    """RU-friendly text cleaner/tokenizer with aggressive artifact filtering."""

    def __init__(self, cfg: dict):
        p_cfg = {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}
        self.cfg = p_cfg
        self.use_razdel = bool(p_cfg.get("use_razdel", True))
        self.lemmatize = bool(p_cfg.get("lemmatize", False))
        self.max_token_len = int(p_cfg.get("max_token_len", 30))
        self.min_token_len = int(p_cfg.get("min_token_len", 3))
        self.placeholder_patterns = [re.compile(p) for p in p_cfg.get("placeholder_patterns", [])]
        self.extra_stopwords = self._load_word_set(p_cfg.get("extra_stopwords_path"))
        self.deny_tokens = self._load_word_set(p_cfg.get("deny_tokens_path"))

        self._tokenizer_mode = self._build_tokenizer_mode()
        self._morph = self._build_morph() if self.lemmatize else None

    @staticmethod
    def _load_word_set(path: str | None) -> set[str]:
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            return set()
        words = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
        return words

    def _build_tokenizer_mode(self) -> str:
        if self.use_razdel:
            try:
                import razdel  # noqa: F401
                return "razdel"
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("razdel unavailable (%s), fallback to regex", exc)
        return "regex"

    @staticmethod
    def _regex_tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zа-яё0-9_\-]+", text, flags=re.IGNORECASE)

    @staticmethod
    def _razdel_tokenize(text: str) -> List[str]:
        from razdel import tokenize as rz_tokenize

        return [t.text for t in rz_tokenize(text)]

    def _build_morph(self):
        try:
            import pymorphy2

            return pymorphy2.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("pymorphy2 unavailable (%s), lemmatization disabled", exc)
            self.lemmatize = False
            return None

    def strip_placeholders(self, text: str) -> str:
        if not self.cfg.get("remove_placeholders", True):
            return text
        out = text
        for pat in self.placeholder_patterns:
            out = pat.sub(" ", out)
        return out

    def clean_text(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"(?i)\b(ticket|case|id|uuid|guid)\b\s*[:=]?\s*[\w-]+", " ", text)
        text = self.strip_placeholders(text)

        if self.cfg.get("remove_urls_emails_phones", True):
            text = URL_RE.sub(" ", text)
            text = EMAIL_RE.sub(" ", text)
            text = PHONE_RE.sub(" ", text)

        norm_numbers = self.cfg.get("normalize_numbers", "remove")
        if norm_numbers == "token":
            token = self.cfg.get("number_token") or "<num>"
            text = DIGIT_RE.sub(f" {token} ", text)
        else:
            text = DIGIT_RE.sub(" ", text)

        if self.cfg.get("keep_cyrillic_latin_only", True):
            text = re.sub(r"[^a-zа-яё\s_<>{}\-]", " ", text, flags=re.IGNORECASE)

        text = SPACE_RE.sub(" ", text).strip()
        return text

    @lru_cache(maxsize=100_000)
    def _lemma(self, token: str) -> str:
        if not self._morph:
            return token
        return self._morph.parse(token)[0].normal_form

    def is_good_token(self, tok: str) -> bool:
        t = (tok or "").strip().lower()
        if not t:
            return False
        if len(t) < self.min_token_len or len(t) > self.max_token_len:
            return False
        if t in RU_STOPWORDS or t in self.extra_stopwords or t in self.deny_tokens:
            return False
        if t in {"id", "uuid", "guid", "ticket", "case"}:
            return False
        if self.cfg.get("drop_tokens_with_digits", True) and any(ch.isdigit() for ch in t):
            return False
        if self.cfg.get("drop_tokens_with_underscores", True) and "_" in t:
            return False
        if self.cfg.get("drop_repeated_char_tokens", True) and REPEATED_CHAR_RE.search(t):
            return False
        if re.fullmatch(r"[-*#<>{}]+", t):
            return False
        if self.cfg.get("keep_cyrillic_latin_only", True) and not re.search(r"[a-zа-яё]", t, flags=re.IGNORECASE):
            return False
        for pat in self.placeholder_patterns:
            if pat.search(t):
                return False
        return True

    def tokenize(self, text: str) -> List[str]:
        tokens = self._razdel_tokenize(text) if self._tokenizer_mode == "razdel" else self._regex_tokenize(text)
        out: List[str] = []
        for token in tokens:
            token = token.strip().lower()
            if self.lemmatize and token:
                token = self._lemma(token)
            if self.is_good_token(token):
                out.append(token)
        return out

    def analyzer(self, text: str) -> List[str]:
        return self.tokenize(self.clean_text(text))

    def preprocess_series(self, s: pd.Series) -> List[str]:
        return [" ".join(self.analyzer(item)) for item in s.fillna("").astype(str)]

    def is_good_term(self, term: str) -> bool:
        parts = [p for p in term.lower().split() if p]
        if not parts:
            return False
        return all(self.is_good_token(p) for p in parts)

    def filter_terms(self, terms: Iterable[str]) -> List[str]:
        out: List[str] = []
        for term in terms:
            if self.is_good_term(term):
                out.append(term)
        return out
