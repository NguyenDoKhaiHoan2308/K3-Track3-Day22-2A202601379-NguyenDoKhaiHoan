from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("text fields must be strings")
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is not None:
            normalized_chosen = re.sub(r"\s+", " ", chosen).strip().casefold()
            normalized_rejected = re.sub(r"\s+", " ", rejected).strip().casefold()
            similarity = SequenceMatcher(None, normalized_chosen, normalized_rejected).ratio()
            if normalized_chosen == normalized_rejected or similarity >= 0.98:
                raise ValueError("chosen and rejected must be meaningfully different")
        return rejected
