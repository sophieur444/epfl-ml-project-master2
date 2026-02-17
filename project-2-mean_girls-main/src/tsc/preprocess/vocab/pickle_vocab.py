# src/tsc/preprocess/vocab/pickle_vocab.py

"""
Builds an indexed vocabulary from vocab_cut.txt and saves it as vocab.pkl.

This script reads each token from vocab_cut.txt, assigns it a unique integer
index based on its order of appearance, and serializes the resulting dictionary
to vocab.pkl using the highest pickle protocol.
"""

import pickle
from tsc.utils.paths import DATA_INTERMEDIATE, DATA_PROCESSED


def main():
    vocab = dict()
    with open(DATA_INTERMEDIATE / "vocab_cut.txt") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(DATA_PROCESSED / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
