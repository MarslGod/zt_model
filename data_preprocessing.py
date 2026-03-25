"""
Data preprocessing module for the Zero-Trust anomaly detection system.

Processes the CERT r4.2 multi-source heterogeneous log dataset into
user behaviour sequences ready for Word2Vec training and Transformer
classification.

Pipeline
--------
Step 1 – Data cleaning      : keep only relevant columns per log type.
Step 2 – Standardisation    : normalise the 'date' column format and
                              parse timestamps.
Step 3 – Malicious filtering: remove rows that match known malicious
                              activity records supplied by the dataset.
Step 4 – Behaviour encoding : map (activity_type, hour_of_day) → token
                              using  token = activity_id * 24 + hour.
Step 5 – Sort & merge       : group by (user, pc, date) and sort by
                              timestamp to build per-user behaviour
                              sequences stored as plain-text corpus files.

Activity dictionary
-------------------
logon            → 0
logoff           → 1
device connect   → 2
device disconnect→ 3
http             → 4
email            → 5
file             → 6

Token formula (eq. 3-1)
-----------------------
    token = activity_id * 24 + hour_of_day
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVITY_MAP: Dict[str, int] = {
    "logon": 0,
    "logoff": 1,
    "device connect": 2,
    "device disconnect": 3,
    "http": 4,
    "email": 5,
    "file": 6,
}

# Maximum token value: 6 * 24 + 23 = 167
MAX_TOKEN: int = max(ACTIVITY_MAP.values()) * 24 + 23

CHUNKSIZE: int = 100_000  # rows per chunk when reading large CSVs


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_date(series: pd.Series) -> pd.Series:
    """Convert various date string formats to a uniform datetime series.

    The CERT dataset stores dates as ``MM/DD/YYYY HH:MM:SS``.  We
    normalise to ``YYYY/MM/DD HH:MM:SS`` and parse as datetime.
    """
    # Try the known CERT format first, then fall back to pandas inference.
    try:
        return pd.to_datetime(series, format="%m/%d/%Y %H:%M:%S")
    except (ValueError, TypeError):
        return pd.to_datetime(series, format="mixed")


def _load_csv_chunks(
    filepath: str | Path,
    usecols: Optional[List[str]] = None,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    """Read a potentially large CSV file in chunks and return a single DataFrame."""
    chunks = []
    for chunk in pd.read_csv(
        filepath,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 1 & 2: Per-log-type cleaning and standardisation
# ---------------------------------------------------------------------------


def load_logon(path: str | Path) -> pd.DataFrame:
    """Load and clean the logon log (logon.csv / device.csv share the same schema).

    Keeps: date, user, pc, activity.
    Parses the date column.
    """
    df = _load_csv_chunks(path, usecols=["date", "user", "pc", "activity"])
    df["date"] = _parse_date(df["date"])
    df["activity"] = df["activity"].str.lower().str.strip()
    return df


def load_device(path: str | Path) -> pd.DataFrame:
    """Load and clean the device log (device.csv).

    Keeps: date, user, pc, activity.
    Activity values are normalised to 'device connect' / 'device disconnect'.
    """
    df = _load_csv_chunks(path, usecols=["date", "user", "pc", "activity"])
    df["date"] = _parse_date(df["date"])
    df["activity"] = (
        df["activity"]
        .str.lower()
        .str.strip()
        .map({"connect": "device connect", "disconnect": "device disconnect"})
        .fillna(df["activity"].str.lower().str.strip())
    )
    return df


def load_http(path: str | Path) -> pd.DataFrame:
    """Load and clean the HTTP log (http.csv).

    Drops: id, content, url (replaced by constant 'http').
    Keeps: date, user, pc, activity='http'.
    """
    df = _load_csv_chunks(path, usecols=["date", "user", "pc"])
    df["date"] = _parse_date(df["date"])
    df["activity"] = "http"
    return df


def load_email(path: str | Path) -> pd.DataFrame:
    """Load and clean the email log (email.csv).

    Drops: id, to, from, size, attachment, content (replaced by 'email').
    Keeps: date, user, pc, activity='email'.
    """
    df = _load_csv_chunks(path, usecols=["date", "user", "pc"])
    df["date"] = _parse_date(df["date"])
    df["activity"] = "email"
    return df


def load_file(path: str | Path) -> pd.DataFrame:
    """Load and clean the file-operation log (file.csv).

    Drops: id, filename (replaced by 'file'), content.
    Keeps: date, user, pc, activity='file'.
    """
    df = _load_csv_chunks(path, usecols=["date", "user", "pc", "activity"])
    df["date"] = _parse_date(df["date"])
    # Map granular file activities to the single 'file' activity label used
    # for token encoding.  The original activity string is preserved so callers
    # can filter malicious records against the raw log.
    df["activity"] = "file"
    return df


# ---------------------------------------------------------------------------
# Step 3: Malicious behaviour filtering
# ---------------------------------------------------------------------------


def remove_malicious_records(
    df: pd.DataFrame,
    malicious_df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows from *df* that match records in *malicious_df*.

    Matching is performed on (user, pc, date) — the same columns present in
    all log types after preprocessing.  This corresponds to the similarity-
    detection approach described in §3.2 of the paper.

    Parameters
    ----------
    df:
        Preprocessed log dataframe.
    malicious_df:
        Dataframe loaded from the insider-threat answer file (e.g.
        ``answers/insiders.csv``) with at least ``user``, ``pc``, and
        ``date`` columns whose date has been parsed to datetime.
    """
    if malicious_df is None or malicious_df.empty:
        return df

    # Round both sides to the same precision to avoid float mismatches.
    key_cols = ["user", "pc", "date"]
    mal_keys = set(
        zip(
            malicious_df["user"],
            malicious_df["pc"],
            malicious_df["date"].dt.floor("s"),
        )
    )
    mask = ~df.apply(
        lambda r: (r["user"], r["pc"], r["date"].floor("s")) in mal_keys,
        axis=1,
    )
    n_removed = (~mask).sum()
    logger.info("Removed %d malicious records.", n_removed)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Temporal behaviour encoding  (eq. 3-1)
# ---------------------------------------------------------------------------


def encode_token(activity: str, hour: int) -> int:
    """Map an (activity, hour) pair to an integer token.

    Uses the formula from eq. (3-1):
        token = activity_id * 24 + hour

    Unknown activity strings are silently mapped to -1.
    """
    activity_id = ACTIVITY_MAP.get(activity, -1)
    if activity_id == -1:
        return -1
    return activity_id * 24 + hour


def add_token_column(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised application of :func:`encode_token` to a dataframe.

    Adds a ``token`` column; rows with unknown activities receive token = -1
    and are subsequently dropped.
    """
    activity_ids = df["activity"].map(ACTIVITY_MAP)
    hours = df["date"].dt.hour
    df = df.copy()
    df["token"] = activity_ids * 24 + hours
    df = df[df["token"] >= 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 5: Sort & merge – build user behaviour sequences
# ---------------------------------------------------------------------------


def build_user_sequences(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Group tokens by (user, pc) and sort by timestamp.

    Returns a dict mapping ``"<user>@<pc>"`` to a list of integer tokens
    ordered chronologically.  This constitutes the training corpus for
    Word2Vec (§3.3) and the input sequences for the Transformer (§4).
    """
    sequences: Dict[str, List[int]] = {}
    groups = df.sort_values("date").groupby(["user", "pc"])
    for (user, pc), grp in tqdm(groups, desc="Building sequences"):
        key = f"{user}@{pc}"
        sequences[key] = grp["token"].tolist()
    return sequences


def sequences_to_corpus(
    sequences: Dict[str, List[int]],
) -> List[List[str]]:
    """Convert integer-token sequences to lists of string tokens.

    Word2Vec in gensim expects an iterable of tokenised sentences; each
    token must be a string.
    """
    return [[str(t) for t in seq] for seq in sequences.values()]


def save_corpus(corpus: List[List[str]], filepath: str | Path) -> None:
    """Persist the corpus to a plain-text file (one sequence per line)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as fh:
        for seq in corpus:
            fh.write(" ".join(seq) + "\n")
    logger.info("Corpus saved to %s (%d sequences).", filepath, len(corpus))


def load_corpus(filepath: str | Path) -> List[List[str]]:
    """Load a corpus previously saved by :func:`save_corpus`."""
    filepath = Path(filepath)
    corpus: List[List[str]] = []
    with filepath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                corpus.append(line.split())
    return corpus


# ---------------------------------------------------------------------------
# High-level preprocessing entry point
# ---------------------------------------------------------------------------


def preprocess(
    data_dir: str | Path,
    output_dir: str | Path,
    malicious_path: Optional[str | Path] = None,
) -> Dict[str, List[int]]:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``logon.csv``, ``device.csv``, ``http.csv``,
        ``email.csv``, and ``file.csv``.
    output_dir:
        Directory where the processed corpus and sequence files are saved.
    malicious_path:
        Optional path to a CSV file listing insider-threat events.  When
        supplied, matching rows are excluded from the corpus (Step 3).

    Returns
    -------
    sequences:
        Mapping of ``"<user>@<pc>"`` → list of integer behaviour tokens.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1 & 2: Loading and cleaning log files ===")
    dfs: List[pd.DataFrame] = []

    log_files = {
        "logon.csv": load_logon,
        "device.csv": load_device,
        "http.csv": load_http,
        "email.csv": load_email,
        "file.csv": load_file,
    }

    for filename, loader in log_files.items():
        fpath = data_dir / filename
        if not fpath.exists():
            logger.warning("Log file not found, skipping: %s", fpath)
            continue
        logger.info("Loading %s …", fpath)
        dfs.append(loader(fpath))

    if not dfs:
        raise FileNotFoundError(
            f"No recognised log files found in '{data_dir}'."
        )

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total records after concatenation: %d", len(df))

    # Step 3: Remove malicious records (if answer file is available)
    if malicious_path is not None:
        logger.info("=== Step 3: Removing malicious records ===")
        mal_df = pd.read_csv(malicious_path, low_memory=False)
        mal_df["date"] = _parse_date(mal_df["date"])
        df = remove_malicious_records(df, mal_df)

    # Step 4: Encode tokens
    logger.info("=== Step 4: Encoding behaviour tokens ===")
    df = add_token_column(df)
    logger.info("Records after encoding: %d", len(df))

    # Step 5: Sort & merge into per-user sequences
    logger.info("=== Step 5: Building user behaviour sequences ===")
    sequences = build_user_sequences(df)
    logger.info("Total unique (user, pc) pairs: %d", len(sequences))

    # Persist corpus
    corpus = sequences_to_corpus(sequences)
    save_corpus(corpus, output_dir / "corpus.txt")

    # Save sequence lengths for diagnostics
    lengths = {k: len(v) for k, v in sequences.items()}
    pd.DataFrame(
        {"user_pc": list(lengths.keys()), "seq_len": list(lengths.values())}
    ).to_csv(output_dir / "sequence_lengths.csv", index=False)

    return sequences
