from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
    *,
    tie_credit: float = 0.5,
) -> float:
    """Return pairwise accuracy, assigning ``tie_credit`` to equal scores."""
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("examples, chosen_scores, and rejected_scores must have equal lengths")
    if not 0.0 <= tie_credit <= 1.0:
        raise ValueError("tie_credit must be between 0 and 1")
    if any(not math.isfinite(score) for score in chosen_scores + rejected_scores):
        raise ValueError("scores must contain only finite values")
    if not examples:
        return 0.0
    wins = sum(
        1.0 if chosen > rejected else tie_credit if chosen == rejected else 0.0
        for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True)
    )
    return wins / len(examples)


def deterministic_response_score(text: str) -> float:
    """Score a response with a transparent, model-free CPU baseline.

    The baseline rewards informative length and lexical diversity. It is intended
    for pipeline smoke tests, not as a substitute for a trained reward model.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.casefold())
    if not words:
        return 0.0
    diversity = len(set(words)) / len(words)
    return math.log1p(len(words)) + diversity


def write_metrics(
    metrics: dict[str, float], output_dir: str | Path, filename: str = "metrics.json"
) -> Path:
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("filename must be a JSON filename without directory components")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / filename
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
