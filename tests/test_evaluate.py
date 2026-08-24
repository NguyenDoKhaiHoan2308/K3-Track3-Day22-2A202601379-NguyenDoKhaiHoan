import pytest

from preference_lab.evaluate import deterministic_response_score, pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_counts_ties_as_half() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [1.0], [1.0]) == 0.5


def test_pairwise_accuracy_rejects_length_mismatch() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="equal lengths"):
        pairwise_accuracy(examples, [], [])


def test_deterministic_response_score_is_reproducible() -> None:
    response = "A concise response with varied and informative words."
    assert deterministic_response_score(response) == deterministic_response_score(response)
