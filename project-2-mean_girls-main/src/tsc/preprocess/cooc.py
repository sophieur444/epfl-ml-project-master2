# src/tsc/preprocess/cooc.py¨

"""
Builds a word co-occurrence matrix from the Twitter training dataset.

This script loads the vocabulary index from `vocab.pkl`, then scans the positive
and negative tweet files. For each tweet, it maps tokens to their integer
indices and records pairwise word co-occurrences within the same tweet. The
resulting sparse COO matrix is deduplicated and saved to `cooc.pkl` for use in
count-based NLP methods (e.g., PMI, SVD embeddings).
"""

from scipy.sparse import coo_matrix
import numpy as np
import pickle
from tsc.utils.paths import DATA_RAW, DATA_PROCESSED


def main():
    with open(DATA_PROCESSED / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    for fn in [
        DATA_RAW / "twitter-datasets/train_pos.txt",
        DATA_RAW / "twitter-datasets/train_neg.txt",
    ]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(DATA_PROCESSED / "cooc.pkl", "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
