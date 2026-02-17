# src/tsc/models/glove/train_glove_svm.py

import pickle
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


from tsc.utils.paths import GLOVE, RESULTS
from tsc.preprocess.split_dataset import stratified_val_split
from tsc.models.glove.data import load_glove_train_data
from tsc.utils.metrics import accuracy, f1_score


def main():
    seed = 0
    val_ratio = 0.2

    search_train_size = 500_000
    search_val_size = 200_000

    # Load data and GloVe embeddings
    X, y = load_glove_train_data()
    print(f"Data shape: {X.shape} | Labels shape: {y.shape}")

    X_train, y_train, X_val, y_val = stratified_val_split(
        X, y, val_ratio=val_ratio, seed=seed
    )
    print(f"Train shape: {X_train.shape} | Validation shape: {X_val.shape}")

    if search_train_size is not None:
        rng = np.random.default_rng(seed)

        train_idx = rng.choice(len(X_train), size=search_train_size, replace=False)
        val_idx = rng.choice(len(X_val), size=search_val_size, replace=False)

        X_train_s = X_train[train_idx]
        y_train_s = y_train[train_idx]
        X_val_s = X_val[val_idx]
        y_val_s = y_val[val_idx]

        print(
            f"Search-train shape: {X_train_s.shape} | Search-val shape: {X_val_s.shape}"
        )
        print(
            "Search-val label distribution:",
            np.mean(y_val_s == 1),
            np.mean(y_val_s == -1),
        )
    else:
        X_train_s, y_train_s, X_val_s, y_val_s = X_train, y_train, X_val, y_val

    C_values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    losses = ["hinge", "squared_hinge"]

    results = {}
    best_acc = -1.0
    best_f1 = -1.0
    best_loss = None
    best_C = None

    # Hyperparameter search (loss, C) - selection by ACCURACY
    for loss in losses:
        for C in C_values:
            print(f"\n=== LinearSVC (loss={loss}, C={C:g}) ===")

            clf = LinearSVC(
                C=C,
                loss=loss,
                random_state=seed,
                max_iter=5000,
            )
            clf.fit(X_train_s, y_train_s)

            y_pred = clf.predict(X_val_s)

            print(
                "Val label distribution:",
                np.mean(y_val_s == 1),
                np.mean(y_val_s == -1),
            )
            print(
                "Val pred distribution:",
                np.mean(y_pred == 1),
                np.mean(y_pred == -1),
            )

            acc = accuracy(y_pred, y_val_s)
            f1 = f1_score(y_pred, y_val_s)

            print(f"Validation accuracy: {acc:.4f}")
            print(f"Validation F1-score: {f1:.4f}")

            results[(loss, C)] = (acc, f1)

            # Select by accuracy. If tie, prefer smaller C (stronger regularization).
            if (acc > best_acc) or (acc == best_acc and (best_C is None or C < best_C)):
                best_acc = acc
                best_f1 = f1
                best_loss = loss
                best_C = C

    print("\n=== LinearSVC summary (validation) ===")
    for loss in losses:
        for C in C_values:
            acc, f1 = results[(loss, C)]
            print(f"loss={loss:13s} | C={C:9.3g} | acc={acc:.4f} | F1={f1:.4f}")

    #  Plot: C tuning curves for each loss
    for loss in losses:
        Cs = np.array(C_values, dtype=float)
        accs = np.array([results[(loss, C)][0] for C in C_values])
        f1s = np.array([results[(loss, C)][1] for C in C_values])

        plt.figure()
        plt.plot(np.log10(Cs), accs, marker="o", label="Accuracy")
        plt.plot(np.log10(Cs), f1s, marker="o", label="F1")

        # Mark best for this loss (by accuracy)
        i_best = int(np.argmax(accs))
        plt.scatter(
            np.log10(Cs[i_best]), accs[i_best], s=80, marker="x", label="Best (acc)"
        )

        plt.xlabel("log10(C)")
        plt.ylabel("Validation score")
        plt.title(f"GloVe + LinearSVC: effect of regularization ({loss})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = RESULTS / f"glove_svm_C_tuning_{loss}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print("Saved plot:", plot_path)

    best_pairs = [
        (loss, C) for (loss, C), (acc, _) in results.items() if acc == best_acc
    ]
    best_pairs = sorted(best_pairs, key=lambda x: x[1])

    if len(best_pairs) > 1:
        print(
            f"\nBest LinearSVC (by accuracy): tie at acc={best_acc:.4f} for {best_pairs}. "
            f"Using loss={best_loss}, C={best_C:g} (smallest C)."
        )

    print(
        f"\nBest LinearSVC (by accuracy): loss={best_loss} | C={best_C:g} | "
        f"acc={best_acc:.4f} | F1={best_f1:.4f}"
    )

    # Final training on full dataset (best accuracy hyperparams)
    print("\nTraining final LinearSVC on the full dataset...")

    final_svm = LinearSVC(
        C=best_C,
        loss=best_loss,
        random_state=seed,
        max_iter=5000,
    )
    final_svm.fit(X, y)

    model_path = GLOVE / "saved_models" / "svm_glove_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_svm, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Final model saved to:", model_path)


if __name__ == "__main__":
    main()
