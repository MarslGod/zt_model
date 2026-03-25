"""
Word2Vec training module for user-behaviour semantic learning.

Implements the Negative-Sampling Skip-Gram Word2Vec model described in §3.3
of the paper.  It takes the user behaviour token corpus produced by
``data_preprocessing.py`` and trains an embedding model so that semantically
related behaviour tokens (e.g. 'logon at 08:00' and 'logon at 09:00') are
mapped to nearby vectors in the embedding space.

The trained model is used later by the Transformer encoder to initialise its
embedding layer with pre-trained behaviour semantics.

Usage (command-line)
--------------------
    python word2vec_training.py \
        --corpus data/processed/corpus.txt \
        --output models/word2vec.model \
        --vector_size 128 \
        --window 5 \
        --min_count 1 \
        --negative 5 \
        --epochs 10

Key hyper-parameters
--------------------
vector_size  : dimensionality of the embedding vectors  (d_model)
window       : context window size  (2c in the paper)
negative     : number of negative samples per positive example
sg           : 1 = Skip-Gram (as used in the paper), 0 = CBOW
hs           : 0 = Negative Sampling (as used in the paper),
               1 = Hierarchical Softmax
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from gensim.models import Word2Vec

from data_preprocessing import load_corpus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_word2vec(
    corpus: List[List[str]],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 1,
    negative: int = 5,
    epochs: int = 10,
    workers: int = 4,
    seed: int = 42,
) -> Word2Vec:
    """Train a Skip-Gram Word2Vec model with Negative Sampling.

    Parameters
    ----------
    corpus:
        List of tokenised sentences (each sentence is a list of string tokens).
    vector_size:
        Dimensionality of the learned embedding vectors.
    window:
        Maximum distance between the current and predicted word within a
        sentence (2c in the paper).
    min_count:
        Minimum frequency required for a token to be included in the
        vocabulary.
    negative:
        Number of negative samples drawn per positive training example.
    epochs:
        Number of passes over the corpus.
    workers:
        Number of worker threads for training.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    model:
        Trained ``gensim.models.Word2Vec`` instance.
    """
    logger.info(
        "Training Word2Vec: vector_size=%d, window=%d, negative=%d, epochs=%d",
        vector_size,
        window,
        negative,
        epochs,
    )
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,        # Skip-Gram
        hs=0,        # Negative Sampling (not Hierarchical Softmax)
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    logger.info(
        "Vocabulary size: %d  |  Embedding dimension: %d",
        len(model.wv),
        vector_size,
    )
    return model


def save_model(model: Word2Vec, path: str | Path) -> None:
    """Persist the trained Word2Vec model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("Word2Vec model saved to %s", path)


def load_model(path: str | Path) -> Word2Vec:
    """Load a previously saved Word2Vec model."""
    return Word2Vec.load(str(path))


# ---------------------------------------------------------------------------
# Embedding matrix extraction
# ---------------------------------------------------------------------------


def build_embedding_matrix(
    model: Word2Vec,
    vocab_size: int,
    vector_size: int,
    pad_token_id: int = 0,
) -> np.ndarray:
    """Build a fixed-size embedding matrix indexed by integer token id.

    Token ids are the integer behaviour tokens produced by
    ``data_preprocessing.encode_token`` (range 0 – MAX_TOKEN = 167).
    Tokens not present in the Word2Vec vocabulary receive a zero vector.

    Parameters
    ----------
    model:
        Trained Word2Vec model.
    vocab_size:
        Number of rows in the output matrix (should be MAX_TOKEN + 2 to
        account for the padding token and all valid behaviour tokens).
    vector_size:
        Embedding dimension; must match the model's ``vector_size``.
    pad_token_id:
        Index reserved for the padding token; its row is kept as zeros.

    Returns
    -------
    matrix:
        ``numpy.ndarray`` of shape ``(vocab_size, vector_size)``.
    """
    matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)
    for token_id in range(vocab_size):
        if token_id == pad_token_id:
            continue
        key = str(token_id)
        if key in model.wv:
            matrix[token_id] = model.wv[key]
        else:
            # Randomly initialise unseen tokens (uniform in [-0.1, 0.1])
            matrix[token_id] = np.random.uniform(-0.1, 0.1, vector_size).astype(
                np.float32
            )
    return matrix


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model on user behaviour sequences."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to the corpus text file (one sequence per line).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/word2vec.model",
        help="Path to save the trained model.",
    )
    parser.add_argument("--vector_size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parse_args()

    logger.info("Loading corpus from %s …", args.corpus)
    corpus = load_corpus(args.corpus)
    logger.info("Corpus: %d sequences loaded.", len(corpus))

    model = train_word2vec(
        corpus=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
    save_model(model, args.output)


if __name__ == "__main__":
    main()
