# src/tsc/models/tfidf/data.py

import numpy as np

from tsc.utils.paths import DATA_TWITTER


def load_tfidf_train_data():
    """Return data adapted to tfidf from train_pos_full / train_neg_full."""
    X, y = [], []

    with open(DATA_TWITTER / "train_neg_full.txt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                X.append(s)
                y.append(-1)

    with open(DATA_TWITTER / "train_pos_full.txt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                X.append(s)
                y.append(1)

    return X, np.array(y)


def load_tfidf_test_data():
    X_test = []
    with open(DATA_TWITTER / "test_data.txt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                X_test.append(s)
            else:
                X_test.append("")
    return X_test
