# src/tsc/models/tfidf/train_tfidf_logreg.py

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tsc.utils.paths import TFIDF
from tsc.preprocess.split_dataset import stratified_val_split
from tsc.models.tfidf.data import load_tfidf_train_data
from tsc.utils.metrics import accuracy, f1_score


def main():
    seed = 0
    val_ratio = 0.2

    X, y = load_tfidf_train_data()

    X_train, y_train, X_val, y_val = stratified_val_split(
        X, y, val_ratio=val_ratio, seed=seed
    )

    # TF-IDF preprocess
    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=300_000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Grid over C
    C_list = np.logspace(-5, -2, 7)

    results = {}
    best_acc = -1.0
    best_cfg = None

    print(
        f"Val label distribution: {np.mean(y_val == 1):.4f} {np.mean(y_val == -1):.4f}"
    )

    for C in C_list:
        print(f"\n=== Logistic Regression (C={C}) ===")

        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            n_jobs=-1,
        )

        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_val_tfidf)

        print(
            "Val pred distribution:",
            f"{np.mean(y_pred == 1):.6f}",
            f"{np.mean(y_pred == -1):.6f}",
        )

        acc = accuracy(y_pred, y_val)
        f1 = f1_score(y_pred, y_val)

        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation F1-score: {f1:.4f}")

        results[C] = (acc, f1)

        if acc > best_acc:
            best_acc = acc
            best_cfg = C

    print("\n=== Logistic Regression summary (validation) ===")
    for C, (acc, f1) in results.items():
        print(f"C={C:<10} | acc={acc:.4f} | F1={f1:.4f}")

    print(f"\nBest Logistic Regression: C={best_cfg} | acc={results[best_cfg][0]:.4f}")

    # Train final model on full dataset
    print("\nTraining final TF-IDF + Logistic Regression on full dataset...")
    X_all_tfidf = vectorizer.fit_transform(X)

    final_clf = LogisticRegression(
        C=best_cfg,
        max_iter=2000,
        n_jobs=-1,
    )
    final_clf.fit(X_all_tfidf, y)

    out_path = TFIDF / "saved_models" / "logreg_tfidf_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "model": final_clf},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Final model saved to:", out_path)


if __name__ == "__main__":
    main()
