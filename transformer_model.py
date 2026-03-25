"""
Transformer-based user behaviour anomaly detection model (§4 of the paper).

Architecture
------------
Input  : user behaviour sequence of integer tokens
         X = [x1, x2, …, xT]  where xᵢ = activity_id * 24 + hour

1. Token Embedding
   Pre-trained Word2Vec embeddings are loaded into a fixed or fine-tuneable
   nn.Embedding layer (dimension d_model).

2. Positional Encoding  (eq. 4-9)
   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

3. Transformer Encoder
   N layers, each containing:
     a) Multi-Head Self-Attention  (eq. 4-10, 4-11)
     b) Position-wise Feed-Forward Network  (eq. 4-12)
   Both sub-layers use residual connections and layer normalisation.

4. Classification Head
   Mean-pool the encoder output over the sequence dimension and apply
   a linear layer to produce a binary (normal / anomalous) logit.

The class labels follow the convention:  0 = normal,  1 = anomalous.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding  (eq. 4-9)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Parameters
    ----------
    d_model:
        Embedding / model dimension.
    max_len:
        Maximum sequence length supported.
    dropout:
        Dropout probability applied after adding positional encodings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding table once.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # (d_model // 2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, the cosine slice has one fewer column than the sine slice.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as *x* with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Full Transformer Encoder Anomaly Detector
# ---------------------------------------------------------------------------


class BehaviourTransformer(nn.Module):
    """Transformer encoder for user behaviour anomaly detection.

    Parameters
    ----------
    vocab_size:
        Number of distinct behaviour tokens (``MAX_TOKEN + 2`` to leave room
        for a padding token at index 0).
    d_model:
        Model / embedding dimension.
    nhead:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers (6 in the paper).
    dim_feedforward:
        Hidden dimension of the position-wise FFN sub-layer.
    dropout:
        Dropout probability.
    num_classes:
        Output classes (2 for binary normal / anomalous classification).
    pretrained_embeddings:
        Optional ``numpy.ndarray`` of shape ``(vocab_size, d_model)`` used to
        initialise the embedding layer.  Rows corresponding to the padding
        token should be zero.
    freeze_embeddings:
        When ``True``, the pre-trained embedding weights are not updated
        during fine-tuning.
    pad_token_id:
        Token id used for padding; its embedding is kept at zero and it is
        masked out in the attention computation.
    max_seq_len:
        Maximum supported sequence length for positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        pad_token_id: int = 0,
        max_seq_len: int = 5000,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id
        self.d_model = d_model

        # --- Token embedding ---
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        if pretrained_embeddings is not None:
            weight = torch.from_numpy(pretrained_embeddings).float()
            assert weight.shape == (vocab_size, d_model), (
                f"Expected pretrained_embeddings shape ({vocab_size}, {d_model}), "
                f"got {weight.shape}."
            )
            self.embedding.weight = nn.Parameter(weight, requires_grad=not freeze_embeddings)

        # --- Positional encoding ---
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # --- Transformer encoder (N=6 layers as in the paper) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (batch, seq, feature)
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

        # --- Classification head ---
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise classifier weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _make_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create a boolean padding mask for :class:`nn.TransformerEncoder`.

        Returns a ``(batch, seq_len)`` bool tensor where ``True`` indicates
        positions that should be *ignored* (padding tokens).
        """
        return src == self.pad_token_id  # (batch, seq_len)

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        src : torch.Tensor
            Integer token sequence, shape ``(batch, seq_len)``.

        Returns
        -------
        logits : torch.Tensor
            Raw class logits, shape ``(batch, num_classes)``.
        """
        # Padding mask: True where tokens are padding.
        src_key_padding_mask = self._make_padding_mask(src)  # (B, T)

        # 1. Token embedding + scale by sqrt(d_model) (standard practice)
        x = self.embedding(src) * math.sqrt(self.d_model)  # (B, T, d_model)

        # 2. Add positional encoding
        x = self.pos_encoding(x)  # (B, T, d_model)

        # 3. Transformer encoder
        #    src_key_padding_mask uses True = ignore (padding) convention.
        memory = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d_model)

        # 4. Mean-pool over the sequence dimension (ignoring padding tokens)
        lengths = (~src_key_padding_mask).float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        # Zero out padding positions before summing.
        mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(memory)  # (B, T, d_model)
        memory = memory.masked_fill(mask_expanded, 0.0)
        pooled = memory.sum(dim=1) / lengths  # (B, d_model)

        # 5. Classification head
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_model(
    vocab_size: int,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    num_classes: int = 2,
    pretrained_embeddings: Optional[np.ndarray] = None,
    freeze_embeddings: bool = False,
    pad_token_id: int = 0,
    max_seq_len: int = 5000,
) -> BehaviourTransformer:
    """Instantiate and return a :class:`BehaviourTransformer`."""
    return BehaviourTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    )
