# src/tsc/models/glove/train_glove_rf.py

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from tsc.utils.paths import GLOVE
from tsc.preprocess.split_dataset import stratified_val_split
from tsc.models.glove.data import load_glove_train_data
from tsc.utils.metrics import accuracy, f1_score


def _tie_break_key(cfg):
    n_estimators, max_depth, min_samples_leaf, max_samples = cfg
    depth_key = max_depth if max_depth is not None else 10**9
    ms_key = max_samples if max_samples is not None else 1.0
    return (depth_key, -min_samples_leaf, ms_key, n_estimators)


def main():
    seed = 0
    rng = np.random.default_rng(seed)

    # Load data and GloVe embeddings
    X, y = load_glove_train_data()
    print("Data shape:", X.shape, "Labels shape:", y.shape)

    X_train, y_train, X_val, y_val = stratified_val_split(
        X, y, val_ratio=0.2, seed=seed
    )
    print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)

    search_train_size = 150_000
    search_val_size = 75_000

    train_idx = rng.choice(
        len(X_train), size=min(search_train_size, len(X_train)), replace=False
    )
    val_idx = rng.choice(
        len(X_val), size=min(search_val_size, len(X_val)), replace=False
    )

    X_train_s, y_train_s = X_train[train_idx], y_train[train_idx]
    X_val_s, y_val_s = X_val[val_idx], y_val[val_idx]

    print("Search-train shape:", X_train_s.shape, "Search-val shape:", X_val_s.shape)
    print(
        "Search-val label distribution:", np.mean(y_val_s == 1), np.mean(y_val_s == -1)
    )

    n_estimators_list = [300, 500]
    max_depth_list = [20, None]
    min_samples_leaf_list = [1, 5]
    max_samples_list = [0.7, 1.0]

    results = {}
    best_acc = -1.0
    best_cfg = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for max_samples in max_samples_list:
                    cfg = (n_estimators, max_depth, min_samples_leaf, max_samples)

                    print(
                        f"\n=== Random Forest "
                        f"(n_estimators={n_estimators}, max_depth={max_depth}, "
                        f"min_samples_leaf={min_samples_leaf}, max_samples={max_samples}) ==="
                    )

                    clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features="sqrt",
                        max_samples=max_samples,
                        n_jobs=-1,
                        random_state=seed,
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

                    results[cfg] = (acc, f1)

                    if (acc > best_acc) or (
                        acc == best_acc
                        and (
                            best_cfg is None
                            or _tie_break_key(cfg) < _tie_break_key(best_cfg)
                        )
                    ):
                        best_acc = acc
                        best_cfg = cfg

    print("\n=== Random Forest summary (validation) ===")
    for (n_estimators, max_depth, min_leaf, max_samples), (acc, f1) in results.items():
        print(
            f"n_estimators={n_estimators:3d}, "
            f"max_depth={str(max_depth):>4}, "
            f"min_leaf={min_leaf:2d}, "
            f"max_samples={max_samples} | "
            f"acc={acc:.4f} | F1={f1:.4f}"
        )

    best_acc, best_f1 = results[best_cfg]
    print(
        f"\nBest Random Forest (by accuracy): "
        f"n_estimators={best_cfg[0]}, max_depth={best_cfg[1]}, "
        f"min_leaf={best_cfg[2]}, max_samples={best_cfg[3]} "
        f"(acc={best_acc:.4f}, F1={best_f1:.4f})"
    )

    print("\nTraining final Random Forest on the full dataset...")

    final_rf = RandomForestClassifier(
        n_estimators=best_cfg[0],
        max_depth=best_cfg[1],
        min_samples_leaf=best_cfg[2],
        max_features="sqrt",
        max_samples=best_cfg[3],
        n_jobs=-1,
        random_state=seed,
    )
    final_rf.fit(X, y)

    model_path = GLOVE / "saved_models" / "rf_glove_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_rf, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Final model saved to:", model_path)


if __name__ == "__main__":
    main()
