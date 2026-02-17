# src/tsc/models/tfidf/train_tfidf_rf.py

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

    max_train = 200_000
    max_val = 50_000
    X_train, y_train = X_train[:max_train], y_train[:max_train]
    X_val, y_val = X_val[:max_val], y_val[:max_val]

    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50_000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(
        f"Val label distribution: {np.mean(y_val == 1):.4f} {np.mean(y_val == -1):.4f}"
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )

    print("\n=== RandomForest (TF-IDF) ===")
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

    # Train final model on full dataset
    print("\nTraining final TF-IDF + RF on full dataset...")
    X_all_tfidf = vectorizer.fit_transform(X)
    final_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    final_rf.fit(X_all_tfidf, y)

    out_path = TFIDF / "saved_models" / "rf_tfidf_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "model": final_rf},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("Final model saved to:", out_path)


if __name__ == "__main__":
    main()
