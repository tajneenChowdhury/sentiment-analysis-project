import argparse
from typing import Any
import numpy as np
from numpy.typing import NDArray
from joblib import load


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


def main(model_path: str, input_texts: list[str]) -> None:
    model = load_model(model_path)
    predictions = model.predict(input_texts)
    for text, prediction in zip(input_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted sentiment: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)