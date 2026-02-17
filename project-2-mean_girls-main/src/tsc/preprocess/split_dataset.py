# src/tsc/preprocess/split_dataset.py

import argparse
import numpy as np
from tsc.utils.paths import DATA_TWITTER, DATA_INTERMEDIATE
from tsc.utils.helpers import write_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split positive and negative tweet files into "
            "train/val with labels and concatenate them."
        )
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.02,
        help="Fraction of data to use for splitting (default: 0.02).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling data.",
    )
    return parser.parse_args()


def stratified_val_split(X, y, val_ratio=0.2, seed=0):
    """Split the training dataset into a training and validation set."""

    assert 0 < val_ratio < 1, "The validation ratio must be between 0 and 1"

    X = np.asarray(X)
    y = np.asarray(y)

    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # Separate classes
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    # Shuffle indices
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_set_size = int(len(pos_idx) * val_ratio)
    neg_set_size = int(len(neg_idx) * val_ratio)

    #  Concatenate random indices of each class for validation set
    val_idx = np.concatenate([pos_idx[:pos_set_size], neg_idx[:neg_set_size]])
    # Concatenate the rest for training set
    train_idx = np.concatenate([pos_idx[pos_set_size:], neg_idx[neg_set_size:]])

    # Re-shuffle to equally distribute classes in sets
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # Split data
    X_val, y_val = X[val_idx], y[val_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    return X_train, y_train, X_val, y_val


def val_split_tweets(pos_tweets, neg_tweets, val_ratio=0.01, seed=42):
    """Split the training dataset into a training and validation set."""

    assert 0 < val_ratio < 1, "The validation ratio must be between 0 and 1"

    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # Shuffle dataset
    rng.shuffle(pos_tweets)
    rng.shuffle(neg_tweets)

    ### Split each set into train and val

    # Positive set
    pos_val_size = int(len(pos_tweets) * val_ratio)
    pos_train_size = len(pos_tweets) - pos_val_size
    pos_train_set = pos_tweets[:pos_train_size]
    pos_val_set = pos_tweets[pos_train_size:]

    # Negative set
    neg_val_size = int(len(neg_tweets) * val_ratio)
    neg_train_size = len(neg_tweets) - neg_val_size
    neg_train_set = neg_tweets[:neg_train_size]
    neg_val_set = neg_tweets[neg_train_size:]

    ### Assign labels and combine sets

    # Train
    train_set = []
    train_set.extend((pos_train_line, 1) for pos_train_line in pos_train_set)
    train_set.extend((neg_train_line, 0) for neg_train_line in neg_train_set)

    # Val
    val_set = []
    val_set.extend((pos_val_line, 1) for pos_val_line in pos_val_set)
    val_set.extend((neg_val_line, 0) for neg_val_line in neg_val_set)

    rng.shuffle(train_set)
    rng.shuffle(val_set)

    return train_set, val_set


def main():
    args = parse_args()

    with (DATA_TWITTER / "train_pos_full.txt").open("r", encoding="utf-8") as f:
        pos_lines = [line.rstrip("\n") for line in f if line.strip() != ""]

    with (DATA_TWITTER / "train_neg_full.txt").open("r", encoding="utf-8") as f:
        neg_lines = [line.rstrip("\n") for line in f if line.strip() != ""]

    train_set, val_set = val_split_tweets(
        pos_lines, neg_lines, val_ratio=args.split_ratio, seed=args.seed
    )

    write_csv(DATA_INTERMEDIATE / "tweets_train.csv", train_set)
    write_csv(DATA_INTERMEDIATE / "tweets_val.csv", val_set)


if __name__ == "__main__":
    main()
