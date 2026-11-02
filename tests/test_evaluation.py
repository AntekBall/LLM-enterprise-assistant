import pytest

from src.evaluation.evaluate_answers import has_real_citation, is_refusal


@pytest.mark.parametrize(
    "text",
    [
        "I don't have enough information in the provided documents to answer that.",
        "I don[2] do not have enough information in the provided documents.",
        "This question cannot be answered based on the provided passages.",
        "There is not enough information to say.",
    ],
)
def test_is_refusal_true(text):
    assert is_refusal(text)


def test_is_refusal_false_for_real_answer():
    assert not is_refusal("Dutch employees get at least 20 vacation days per year. [1]")


def test_has_real_citation():
    assert has_real_citation("You get 20 days of leave per year. [1]")
    assert not has_real_citation("[1]")
    assert not has_real_citation("20 days of leave per year")
