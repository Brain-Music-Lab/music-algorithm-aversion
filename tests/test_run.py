import pytest
from algorithmaversion.run import main as run


@pytest.mark.parametrize(
    "option",
    [
        "arm_cleaning",
        "context_source_comparison",
        "count_vectorizer",
        "decision_tree",
        "tfidf_vectorizer",
        "sentiment_analysis",
    ]
)
def test_run(option):
    run([option])
