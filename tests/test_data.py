import json

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_load_error_contains_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    valid = {"prompt": "p", "chosen": "good", "rejected": "bad"}
    path.write_text(f"{json.dumps(valid)}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        load_jsonl(path)


def test_duplicate_prompt_is_rejected(tmp_path) -> None:
    row = {"prompt": "Same prompt", "chosen": "good", "rejected": "bad"}
    path = tmp_path / "duplicates.jsonl"
    path.write_text("\n".join((json.dumps(row), json.dumps(row))), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt"):
        load_jsonl(path)


def test_split_keeps_prompt_groups_together() -> None:
    examples = [
        PreferenceExample(prompt="Repeated", chosen="a", rejected="b"),
        PreferenceExample(prompt=" repeated ", chosen="c", rejected="d"),
        PreferenceExample(prompt="Other", chosen="e", rejected="f"),
    ]
    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    assert not (
        {example.prompt.strip().casefold() for example in train}
        & {example.prompt.strip().casefold() for example in validation}
    )
