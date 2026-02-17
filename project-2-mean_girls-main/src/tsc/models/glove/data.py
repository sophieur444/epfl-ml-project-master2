# src/tsc/models/glove/data.py

import pickle
import numpy as np

from tsc.utils.paths import DATA_PROCESSED, DATA_TWITTER
from tsc.preprocess.embedding.awe import AverageWordEmbedding


def load_glove_encoder():
    """Load vocab + embeddings + return the AverageWordEmbedding encoder."""
    with open(DATA_PROCESSED / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(DATA_PROCESSED / "embeddings.npy")
    encoder = AverageWordEmbedding(embeddings, vocab)
    return encoder


def load_glove_train_data():
    """Return X, y (tweets encoded with GloVe) for train_pos_full / train_neg_full."""
    encoder = load_glove_encoder()

    X_list = []
    y_list = []

    # negatives
    with open(DATA_TWITTER / "train_neg_full.txt", encoding="utf-8") as fneg:
        for line in fneg:
            tokens = [t for t in line.rstrip("\n").split(" ") if t]
            X_list.append(encoder.encode(tokens))
            y_list.append(-1)

    # positives
    with open(DATA_TWITTER / "train_pos_full.txt", encoding="utf-8") as fpos:
        for line in fpos:
            tokens = [t for t in line.rstrip("\n").split(" ") if t]
            X_list.append(encoder.encode(tokens))
            y_list.append(1)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def load_glove_test_data():
    """Returns a list of encoded vectors from test_data.txt."""
    encoder = load_glove_encoder()
    X_test = []

    with open(DATA_TWITTER / "test_data.txt", encoding="utf-8") as f:
        for line in f:
            tokens = [t for t in line.strip().split(" ") if t]
            X_test.append(encoder.encode(tokens))

    return np.vstack(X_test)
