# src/tsc/preprocess/natural_language/awe.py

import numpy as np


class AverageWordEmbedding:
    """
    Average the word vectors over all words of the tweet.

    Parameters
    ----------
    embeddings : np.ndarray
        Matrix of shape (vocab_size, embedding_dim) containing word vectors.
    vocab : dict
        Mapping word -> index.
    """

    def __init__(self, embeddings: np.ndarray, vocab: dict):
        self.embeddings = embeddings
        self.vocab = vocab
        self.dim = embeddings.shape[1]

    def encode(self, tokens):
        """
        Average the word vectors of a list of tokens.
        If no token matches the vocabulary, returns a zero vector.
        """
        vecs = []
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                vecs.append(self.embeddings[idx])

        if not vecs:
            return np.zeros(self.dim, dtype=float)

        return np.mean(vecs, axis=0)

    def encode_batch(self, list_of_token_lists):
        """
        Average the word vectors of a batch of token lists.
        Returns a matrix of shape (batch_size, embedding_dim).
        """
        return np.vstack([self.encode(tokens) for tokens in list_of_token_lists])
