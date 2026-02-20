from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from app.utils import dump_json, load_yaml


@dataclass
class SeedResult:
    label: Optional[int]
    reasons: List[str]


class ComplaintSeeder:
    def __init__(self, lexicon_path: str | Path):
        lex = load_yaml(lexicon_path)
        self.strong_patterns = [re.compile(p, re.IGNORECASE) for p in lex.get("strong_complaint_patterns", [])]
        self.non_patterns = [re.compile(p, re.IGNORECASE) for p in lex.get("non_complaint_patterns", [])]

    def seed_label(self, text: str) -> SeedResult:
        reasons: List[str] = []
        for pat in self.strong_patterns:
            if pat.search(text):
                reasons.append(pat.pattern)
        if reasons:
            return SeedResult(label=1, reasons=reasons)

        non_reasons: List[str] = []
        for pat in self.non_patterns:
            if pat.search(text):
                non_reasons.append(pat.pattern)
        if non_reasons:
            return SeedResult(label=0, reasons=non_reasons)

        return SeedResult(label=None, reasons=[])


class ComplaintModel:
    def __init__(self, model_type: str = "logreg", random_state: int = 42):
        self.model_type = model_type
        if model_type == "linearsvc":
            self.model = LinearSVC(random_state=random_state)
        else:
            self.model = LogisticRegression(
                solver="liblinear",
                random_state=random_state,
                max_iter=1000,
            )

    def fit(self, x: csr_matrix, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def score(self, x: csr_matrix) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.decision_function(x)

    def predict(self, x: csr_matrix, threshold: float) -> np.ndarray:
        return (self.score(x) >= threshold).astype(int)


def build_seed_labels(texts: List[str], lexicon_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    seeder = ComplaintSeeder(lexicon_path)
    labels = []
    reason_text = []
    for t in texts:
        res = seeder.seed_label(t)
        labels.append(res.label)
        reason_text.append(";".join(res.reasons))
    return np.array(labels, dtype=object), reason_text


def train_complaint_classifier(
    x_baseline: csr_matrix,
    baseline_texts: List[str],
    cfg: Dict,
    lexicon_path: str | Path,
    model_dir: str | Path,
    reports_dir: str | Path,
) -> Tuple[ComplaintModel, np.ndarray, List[str], Dict]:
    seed_labels, reasons = build_seed_labels(baseline_texts, lexicon_path)
    known_idx = np.where(seed_labels != None)[0]  # noqa: E711
    if len(known_idx) < 20:
        raise ValueError("Too few seeded samples to train complaint classifier")

    y = seed_labels[known_idx].astype(int)
    x = x_baseline[known_idx]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=cfg.get("random_state", 42),
        stratify=y,
    )

    model = ComplaintModel(cfg["complaint"].get("model", "logreg"), cfg.get("random_state", 42))
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val, threshold=cfg["complaint"].get("threshold", 0.5))
    p, r, f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
    metrics = {
        "seed_train_size": int(len(x_train.toarray())) if False else int(x_train.shape[0]),
        "seed_valid_size": int(x_val.shape[0]),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(model_dir) / "complaint_model.joblib")
    dump_json(metrics, Path(reports_dir) / "complaint_seed_metrics.json")
    return model, seed_labels, reasons, metrics
