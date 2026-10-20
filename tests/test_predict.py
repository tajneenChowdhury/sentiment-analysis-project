import pytest
from src.predict import load_model, predict_texts


MODEL_PATH = "models/sentiment.joblib"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this movie, it was fantastic and inspiring!", 1),
        ("The service was terrible and the food was awful.", 0),
    ],
)
def test_sentiment_predictions(text: str, expected: int) -> None:
    clf = load_model(MODEL_PATH)
    preds, _ = predict_texts(clf, [text])
    assert preds[0] == expected