from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_PII_PATTERNS = {
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone number": re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)"),
}


def load_jsonl(path: str | Path, *, check_pii: bool = False) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Errors include source line numbers. Prompts must be unique after whitespace and
    case normalization. Set ``check_pii`` to reject common email/phone patterns.
    """
    examples: list[PreferenceExample] = []
    source = Path(path)
    seen_prompts: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                example = PreferenceExample.model_validate(raw)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                raise ValueError(f"{source}: line {line_number}: {exc}") from exc

            prompt_key = " ".join(example.prompt.split()).casefold()
            if prompt_key in seen_prompts:
                first_line = seen_prompts[prompt_key]
                raise ValueError(
                    f"{source}: line {line_number}: duplicate prompt (first seen on line {first_line})"
                )
            seen_prompts[prompt_key] = line_number

            if check_pii:
                combined_text = f"{example.prompt} {example.chosen} {example.rejected}"
                for label, pattern in _PII_PATTERNS.items():
                    if pattern.search(combined_text):
                        raise ValueError(f"{source}: line {line_number}: possible {label} detected")
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    All rows sharing a normalized prompt remain in the same partition.
    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        key = " ".join(example.prompt.split()).casefold()
        groups.setdefault(key, []).append(example)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    validation_group_count = round(len(keys) * validation_ratio)
    validation_keys = set(keys[:validation_group_count])
    train = [example for key in keys if key not in validation_keys for example in groups[key]]
    validation = [example for key in keys if key in validation_keys for example in groups[key]]
    return train, validation
