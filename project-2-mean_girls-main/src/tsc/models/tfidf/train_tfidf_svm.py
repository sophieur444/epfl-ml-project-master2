# src/tsc/models/tfidf/train_tfidf_svm.py

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from tsc.utils.paths import RESULTS, TFIDF
from tsc.preprocess.split_dataset import stratified_val_split
from tsc.models.tfidf.data import load_tfidf_train_data
from tsc.utils.metrics import accuracy, f1_score


def run_once(X_tr, y_tr, X_val, y_val, vectorizer, C):
    X_tr_tfidf = vectorizer.fit_transform(X_tr)
    X_val_tfidf = vectorizer.transform(X_val)

    clf = LinearSVC(C=C, class_weight=None)
    clf.fit(X_tr_tfidf, y_tr)
    y_pred = clf.predict(X_val_tfidf)

    acc = accuracy(y_pred, y_val)
    f1 = f1_score(y_pred, y_val)
    return acc, f1


def main():
    seed = 0
    val_ratio = 0.2

    X, y = load_tfidf_train_data()
    X_train, y_train, X_val, y_val = stratified_val_split(
        X, y, val_ratio=val_ratio, seed=seed
    )

    # Baseline TF-IDF config (used to tune C)
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

    # LinearSVC tune: C (regularization)
    C_list = np.logspace(-3, 1, 7)  # 0.001 -> 10

    results = {}
    best_acc = -1.0
    best_C = None

    print(
        f"Val label distribution: {np.mean(y_val == 1):.4f} {np.mean(y_val == -1):.4f}"
    )

    for C in C_list:
        print(f"\n=== LinearSVC (C={C}) ===")

        clf = LinearSVC(C=C, class_weight=None)
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_val_tfidf)

        acc = accuracy(y_pred, y_val)
        f1 = f1_score(y_pred, y_val)

        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation F1-score: {f1:.4f}")

        results[C] = (acc, f1)
        if acc > best_acc:
            best_acc = acc
            best_C = C

    print("\n=== LinearSVC summary (validation) ===")
    for C, (acc, f1) in results.items():
        print(f"C={C:<10} | acc={acc:.4f} | F1={f1:.4f}")

    print(
        f"\nBest LinearSVC: C={best_C} | acc={results[best_C][0]:.4f} | F1={results[best_C][1]:.4f}"
    )

    # Plot: C tuning curve
    Cs = np.array(sorted(results.keys()))
    accs = np.array([results[c][0] for c in Cs])
    f1s = np.array([results[c][1] for c in Cs])

    plt.figure()
    plt.plot(np.log10(Cs), accs, marker="o", label="Accuracy")
    plt.plot(np.log10(Cs), f1s, marker="o", label="F1")
    plt.xlabel("log10(C)")
    plt.ylabel("Validation score")
    plt.title("TF-IDF + LinearSVC: effect of regularization")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS / "tfidf_svm_C_tuning.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Saved plot:", plot_path)

    # Ablation study (C fixed)
    print("\n=== TF-IDF Ablation Study (LinearSVC with fixed C) ===")

    ablation_results = []

    # Ablation 1: n-gram range
    for ngr in [(1, 1), (1, 2), (1, 3)]:
        vec = TfidfVectorizer(
            lowercase=False,
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            ngram_range=ngr,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=300_000,
        )
        acc, f1 = run_once(X_train, y_train, X_val, y_val, vec, best_C)
        ablation_results.append(("ngram_range", str(ngr), acc, f1))

    # Ablation 2: vocabulary size (max_features)
    for k in [50_000, 100_000, 300_000]:
        vec = TfidfVectorizer(
            lowercase=False,
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=k,
        )
        acc, f1 = run_once(X_train, y_train, X_val, y_val, vec, best_C)
        ablation_results.append(("max_features", str(k), acc, f1))

    print("\n--- Ablation results (validation) ---")
    for factor, setting, acc, f1 in ablation_results:
        print(f"{factor:12} | {setting:10} | acc={acc:.4f} | f1={f1:.4f}")

    # Train final model on full dataset
    print("\nTraining final TF-IDF + LinearSVC on full dataset...")
    final_vectorizer = TfidfVectorizer(
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

    X_all_tfidf = final_vectorizer.fit_transform(X)
    final_clf = LinearSVC(C=best_C)
    final_clf.fit(X_all_tfidf, y)

    out_path = TFIDF / "saved_models" / "svm_tfidf_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": final_vectorizer, "model": final_clf},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Final model saved to:", out_path)


if __name__ == "__main__":
    main()
