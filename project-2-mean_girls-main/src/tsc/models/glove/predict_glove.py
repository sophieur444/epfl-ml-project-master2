# src/tsc/models/glove/predict_glove.py

import argparse
import pickle
import numpy as np

from tsc.utils.paths import GLOVE, RESULTS
from tsc.models.glove.data import load_glove_test_data
from tsc.utils.helpers import create_csv_submission


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

    model_path = GLOVE / "saved_models" / f"{args.model}_glove_model.pkl"
    out_path = RESULTS / "predictions.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it before running prediction."
        )

    # Load test data
    X_test = load_glove_test_data()

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Loaded model:", model_path)

    # Make predictions
    preds = model.predict(X_test)
    preds = np.asarray(preds).reshape(-1)

    # Create submission file
    ids = np.arange(1, len(preds) + 1)
    create_csv_submission(ids, preds, out_path)

    print("Submission saved to:", out_path)


if __name__ == "__main__":
    main()
