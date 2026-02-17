# src/tsc/models/tfidf/predict_tfidf.py

import argparse
import csv
import pickle

from tsc.utils.paths import RESULTS, TFIDF
from tsc.models.tfidf.data import load_tfidf_test_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict sentiment labels on test data using a trained GloVe model."
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "svm"],
        default="logreg",
        help="Which trained model to use (default: logreg).",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    model_path = TFIDF / "saved_models" / f"{args.model}_tfidf_model.pkl"
    out_path = RESULTS / "predictions.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it before running prediction."
        )

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    vectorizer = obj["vectorizer"]
    model = obj["model"]

    X_test = load_tfidf_test_data()
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    out_path = RESULTS / "predictions.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Prediction"])
        for i, pred in enumerate(y_pred, start=1):
            w.writerow([i, int(pred)])

    print("Saved submission to:", out_path)


if __name__ == "__main__":
    main()
