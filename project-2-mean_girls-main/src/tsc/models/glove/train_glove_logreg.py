# src/tsc/models/glove/train_glove_logreg.py

import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression

from tsc.utils.paths import GLOVE
from tsc.preprocess.split_dataset import stratified_val_split
from tsc.models.glove.data import load_glove_train_data
from tsc.utils.metrics import accuracy, f1_score


def main():
    # Load data and GloVe embeddings
    X, y = load_glove_train_data()
    print(f"Data shape: {X.shape} | Labels shape: {y.shape}")

    X_train, y_train, X_val, y_val = stratified_val_split(X, y, val_ratio=0.2, seed=0)
    print(f"Train shape: {X_train.shape} | Validation shape: {X_val.shape}")

    # Hyperparameter search (C) selection by accuracy
    C_values = np.logspace(-5, -2, 7)
    results = {}

    best_acc = -1.0
    best_f1 = -1.0
    best_C = None

    for C in C_values:
        print(f"\n=== Logistic Regression (C={C}) ===")

        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
            random_state=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        acc = accuracy(y_pred, y_val)
        f1 = f1_score(y_pred, y_val)
        print(
            "Val label distribution:",
            np.mean(y_val == 1),
            np.mean(y_val == -1),
        )
        print(
            "Val pred distribution:",
            np.mean(y_pred == 1),
            np.mean(y_pred == -1),
        )

        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation F1-score: {f1:.4f}")

        results[C] = (acc, f1)

        # Select by accuracy. If tie, prefer smaller C (stronger regularization).
        if (acc > best_acc) or (acc == best_acc and (best_C is None or C < best_C)):
            best_acc = acc
            best_f1 = f1
            best_C = C

    print("\n=== Logistic Regression summary (validation) ===")
    for C in C_values:
        acc, f1 = results[C]
        print(f"C={C:5} | acc={acc:.4f} | F1={f1:.4f}")

    print(
        f"\nBest Logistic Regression (by accuracy): C={best_C} | acc={best_acc:.4f} | F1={best_f1:.4f}"
    )

    # Final training on full dataset (best accuracy hyperparams)
    print("\nTraining final Logistic Regression on the full dataset...")

    final_logreg = LogisticRegression(
        C=best_C,
        max_iter=1000,
        solver="liblinear",
        random_state=0,
    )
    final_logreg.fit(X, y)

    model_path = GLOVE / "saved_models" / "logreg_glove_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_logreg, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Final model saved to:", model_path)


if __name__ == "__main__":
    main()
