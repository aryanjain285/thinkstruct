"""Learning-to-rank reranker.

A gradient-boosted regressor over retrieval features, trained on the pooled relevance
judgements. It reorders the candidate set hybrid retrieval produced, so comparison
against the unreranked baseline is controlled.

Chosen over cross-encoder fine-tuning because it trains in seconds on CPU and scores
in microseconds rather than ~90 ms. The trade-off: it only sees the features, never
the text.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patsearch.search.query import Hit
from patsearch.training.features import FEATURE_NAMES, extract_features


@dataclass(slots=True)
class TrainingReport:
    n_train_rows: int
    n_test_rows: int
    n_train_patents: int
    n_test_patents: int
    label_distribution: dict[int, int]
    feature_importance: dict[str, float]
    model: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "params": self.params,
            "rows": {"train": self.n_train_rows, "test": self.n_test_rows},
            "patents": {"train": self.n_train_patents, "test": self.n_test_patents},
            "label_distribution": {str(k): v for k, v in sorted(self.label_distribution.items())},
            "feature_importance": dict(
                sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)
            ),
        }


class LTRModel:
    """Wraps the fitted estimator plus the feature contract it was trained against."""

    def __init__(self, estimator, feature_names: tuple[str, ...] = FEATURE_NAMES) -> None:
        self.estimator = estimator
        self.feature_names = feature_names
        self.model_name = f"ltr-{type(estimator).__name__}"

    def score_hits(self, query: str, hits: list[Hit]) -> list[float]:
        if not hits:
            return []
        X = [extract_features(query, h) for h in hits]
        return [float(v) for v in self.estimator.predict(X)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"estimator": self.estimator, "features": self.feature_names}, fh)

    @classmethod
    def load(cls, path: Path) -> LTRModel:
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        # A model trained against a different feature set would score nonsense.
        if tuple(blob["features"]) != FEATURE_NAMES:
            raise ValueError(
                f"model was trained on {len(blob['features'])} features "
                f"but the code now produces {len(FEATURE_NAMES)}; retrain it"
            )
        return cls(blob["estimator"], tuple(blob["features"]))


class LTRReranker:
    """Reranker protocol adapter. Uses features, so it needs Hits rather than text."""

    def __init__(self, model: LTRModel) -> None:
        self.model = model
        self.model_name = model.model_name

    def score_hits(self, query: str, hits: list[Hit]) -> list[float]:
        return self.model.score_hits(query, hits)

    def score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError(
            "LTRReranker scores from retrieval features, not raw text. "
            "Call score_hits(query, hits), or use rerank() which dispatches correctly."
        )


def split_by_patent(
    patent_ids: list[str], *, test_fraction: float = 0.3, seed: int = 17
) -> tuple[set[str], set[str]]:
    """Partition patents into train/test. Grouping by patent prevents leakage."""
    import random

    uniq = sorted(set(patent_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    cut = max(1, int(len(uniq) * (1 - test_fraction)))
    return set(uniq[:cut]), set(uniq[cut:])


def train(
    rows: list,
    *,
    test_fraction: float = 0.3,
    seed: int = 17,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> tuple[LTRModel, TrainingReport, list]:
    """Fit the ranker. Returns (model, report, held-out rows)."""
    from collections import Counter

    from sklearn.ensemble import GradientBoostingRegressor

    if not rows:
        raise ValueError("no training rows")

    train_pats, test_pats = split_by_patent(
        [r.patent_id for r in rows], test_fraction=test_fraction, seed=seed
    )
    train_rows = [r for r in rows if r.patent_id in train_pats]
    test_rows = [r for r in rows if r.patent_id in test_pats]
    if not train_rows:
        raise ValueError("split produced no training rows")

    X = [r.features for r in train_rows]
    y = [float(r.label) for r in train_rows]

    est = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        subsample=0.9,
    )
    est.fit(X, y)

    importance = {
        name: round(float(v), 4)
        for name, v in zip(FEATURE_NAMES, est.feature_importances_, strict=True)
    }
    report = TrainingReport(
        n_train_rows=len(train_rows),
        n_test_rows=len(test_rows),
        n_train_patents=len(train_pats),
        n_test_patents=len(test_pats),
        label_distribution=dict(Counter(r.label for r in train_rows)),
        feature_importance=importance,
        model="GradientBoostingRegressor",
        params={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "seed": seed,
            "test_fraction": test_fraction,
        },
    )
    return LTRModel(est), report, test_rows


def write_report(report: TrainingReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
