"""
End-to-end training and evaluation script for the Zero-Trust behaviour
anomaly detection system.

Workflow
--------
1. (Optional) Preprocess raw CERT log CSVs  →  corpus.txt + sequences.
2. Train Word2Vec on the behaviour corpus to obtain behaviour embeddings.
3. Prepare labelled windows of user behaviour sequences.
4. Train the Transformer encoder classifier.
5. Evaluate on the held-out test split and report metrics.

Quick start (assuming preprocessed data already exists)
-------------------------------------------------------
    python train.py \
        --corpus   data/processed/corpus.txt \
        --w2v      models/word2vec.model \
        --data_dir data/processed \
        --output   models/transformer.pt

Full pipeline (starting from raw CSVs)
---------------------------------------
    python train.py \
        --raw_dir  data/raw \
        --corpus   data/processed/corpus.txt \
        --w2v      models/word2vec.model \
        --data_dir data/processed \
        --output   models/transformer.pt \
        --run_preprocessing
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split

from data_preprocessing import (
    MAX_TOKEN,
    load_corpus,
    preprocess,
)
from transformer_model import build_model
from word2vec_training import build_embedding_matrix, load_model, train_word2vec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

# Maximum token value: 6 * 24 + 23 = 167  (from data_preprocessing.MAX_TOKEN)
# PAD_TOKEN_ID is one beyond the largest valid token so it never collides.
PAD_TOKEN_ID = MAX_TOKEN + 1   # index reserved for padding


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BehaviourSequenceDataset(Dataset):
    """Windowed user behaviour sequence dataset.

    Each sample is a fixed-length window of behaviour tokens drawn from a
    user's chronological sequence together with a binary label (0 = normal,
    1 = anomalous).

    Parameters
    ----------
    sequences:
        Mapping of ``"<user>@<pc>"`` → list of integer tokens.
    labels:
        Mapping of ``"<user>@<pc>"`` → 0 or 1.  Keys missing from this dict
        default to 0 (normal).
    window_size:
        Number of tokens per sample.
    stride:
        Step size between consecutive windows.
    pad_token_id:
        Token id used for sequences shorter than *window_size*.
    """

    def __init__(
        self,
        sequences: Dict[str, List[int]],
        labels: Dict[str, int],
        window_size: int = 128,
        stride: int = 64,
        pad_token_id: int = PAD_TOKEN_ID,
    ) -> None:
        self.window_size = window_size
        self.pad_token_id = pad_token_id
        self.samples: List[Tuple[List[int], int]] = []

        for key, seq in sequences.items():
            label = labels.get(key, 0)
            if len(seq) == 0:
                continue
            # Slide a window over the sequence.
            for start in range(0, max(1, len(seq) - window_size + 1), stride):
                window = seq[start : start + window_size]
                # Pad if necessary.
                if len(window) < window_size:
                    window = window + [pad_token_id] * (window_size - len(window))
                self.samples.append((window, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, label = self.samples[idx]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * tokens.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, str]:
    """Evaluate on *loader* and return (loss, accuracy, report_string)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        total_loss += loss.item() * tokens.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)

    unique_labels = set(all_labels)
    if len(unique_labels) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        report += f"\nROC-AUC: {auc:.4f}"

    return avg_loss, acc, report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Zero-Trust behaviour anomaly detection system."
    )
    # Data sources
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Directory with raw CERT CSV files.  Required if --run_preprocessing.",
    )
    parser.add_argument(
        "--malicious_path",
        type=str,
        default=None,
        help="Path to the insider-threat answer CSV (for malicious-record removal).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.txt",
        help="Path to the preprocessed behaviour corpus.",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help=(
            "CSV with columns 'user_pc' and 'label' (0/1) for supervised training. "
            "When omitted, all sequences are treated as normal (label=0)."
        ),
    )
    # Preprocessing flag
    parser.add_argument(
        "--run_preprocessing",
        action="store_true",
        help="Run the full preprocessing pipeline before training.",
    )
    # Word2Vec
    parser.add_argument(
        "--w2v",
        type=str,
        default="models/word2vec.model",
        help="Path to a pre-trained Word2Vec model.  Trained from scratch if absent.",
    )
    parser.add_argument("--w2v_vector_size", type=int, default=128)
    parser.add_argument("--w2v_window", type=int, default=5)
    parser.add_argument("--w2v_negative", type=int, default=5)
    parser.add_argument("--w2v_epochs", type=int, default=10)
    # Transformer
    parser.add_argument("--output", type=str, default="models/transformer.pt")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. (Optional) Preprocessing
    # ------------------------------------------------------------------
    if args.run_preprocessing:
        if args.raw_dir is None:
            raise ValueError("--raw_dir must be provided when --run_preprocessing is set.")
        data_dir = Path(args.raw_dir)
        output_dir = Path(args.corpus).parent
        logger.info("Running preprocessing pipeline …")
        sequences = preprocess(
            data_dir=data_dir,
            output_dir=output_dir,
            malicious_path=args.malicious_path,
        )
    else:
        # Load from pre-built corpus.
        logger.info("Loading corpus from %s …", args.corpus)
        raw_corpus = load_corpus(args.corpus)
        # Reconstruct sequences dict from corpus (keys become str indices).
        sequences: Dict[str, List[int]] = {
            str(i): [int(t) for t in seq] for i, seq in enumerate(raw_corpus)
        }

    # ------------------------------------------------------------------
    # 2. Labels  (default: all normal)
    # ------------------------------------------------------------------
    labels: Dict[str, int] = {}
    if args.labels_csv is not None:
        import pandas as pd

        ldf = pd.read_csv(args.labels_csv)
        labels = dict(zip(ldf["user_pc"].astype(str), ldf["label"].astype(int)))
        logger.info(
            "Loaded labels for %d user-pc pairs  (anomalous: %d).",
            len(labels),
            sum(v for v in labels.values()),
        )

    # ------------------------------------------------------------------
    # 3. Word2Vec embeddings
    # ------------------------------------------------------------------
    w2v_path = Path(args.w2v)
    if w2v_path.exists():
        logger.info("Loading pre-trained Word2Vec model from %s …", w2v_path)
        w2v_model = load_model(w2v_path)
    else:
        logger.info("Training Word2Vec model …")
        corpus = load_corpus(args.corpus)
        w2v_model = train_word2vec(
            corpus=corpus,
            vector_size=args.w2v_vector_size,
            window=args.w2v_window,
            negative=args.w2v_negative,
            epochs=args.w2v_epochs,
        )
        from word2vec_training import save_model as save_w2v

        save_w2v(w2v_model, w2v_path)

    vocab_size = PAD_TOKEN_ID + 1  # tokens 0 … MAX_TOKEN + 1 (pad)
    embedding_matrix = build_embedding_matrix(
        model=w2v_model,
        vocab_size=vocab_size,
        vector_size=args.d_model,
    )

    # ------------------------------------------------------------------
    # 4. Dataset & DataLoaders
    # ------------------------------------------------------------------
    dataset = BehaviourSequenceDataset(
        sequences=sequences,
        labels=labels,
        window_size=args.window_size,
        stride=args.stride,
        pad_token_id=PAD_TOKEN_ID,
    )
    logger.info("Total samples in dataset: %d", len(dataset))

    n_test = max(1, int(len(dataset) * args.test_split))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info("Train samples: %d  |  Test samples: %d", n_train, n_test)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    model = build_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_embeddings=embedding_matrix,
    ).to(device)

    # Handle class imbalance with weighted cross-entropy loss.
    if labels:
        n_pos = sum(v for v in labels.values())
        n_neg = len(labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0
        class_weights = torch.tensor([1.0, pos_weight], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, report = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  test_loss=%.4f  test_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            test_loss,
            test_acc,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), output_path)
            logger.info("  → Saved best model to %s", output_path)

    # Final evaluation
    model.load_state_dict(torch.load(output_path, map_location=device))
    _, final_acc, final_report = evaluate(model, test_loader, criterion, device)
    logger.info("=== Final Evaluation ===\n%s", final_report)
    logger.info("Final accuracy: %.4f", final_acc)


if __name__ == "__main__":
    main()
