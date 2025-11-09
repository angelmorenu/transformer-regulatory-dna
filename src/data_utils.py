"""
data_utils.py
Utilities for loading sequences, extracting windows, and building datasets.
"""
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd

try:
    from pyfaidx import Fasta
except Exception:
    Fasta = None

def _extract_seq_naive(fasta_path: str, chrom: str, start_1based: int, end_1based_exclusive: int) -> str:
    """
    Naive FASTA reader that extracts [start, end) (1-based, end-exclusive) for the given chrom.
    This is a simple fallback for small FASTA files when pyfaidx isn't available.
    """
    seq = []
    current = None
    start0 = start_1based - 1
    end0 = end_1based_exclusive - 1
    acc_len = 0
    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                acc_len = 0
                continue
            if current == chrom:
                # consume this sequence line
                chunk = line.upper()
                chunk_len = len(chunk)
                # Determine overlap between this chunk and desired [start0, end0)
                chunk_start = acc_len
                chunk_end = acc_len + chunk_len
                # Overlap in 0-based coordinates
                ov_start = max(start0, chunk_start)
                ov_end = min(end0, chunk_end)
                if ov_start < ov_end:
                    # indices within chunk
                    i0 = ov_start - chunk_start
                    i1 = ov_end - chunk_start
                    seq.append(chunk[i0:i1])
                acc_len = chunk_end
                # Early exit if we've reached or passed end
                if acc_len >= end0:
                    break
    return "".join(seq)

def extract_sequence_window(fasta_path: str, chrom: str, center: int, window: int = 1000) -> str:
    """
    Extract +/- window around center (1-based half-open) and return uppercase sequence.
    Requires a FASTA index (.fai). If pyfaidx not available, returns 'N' string as fallback.
    """
    start = max(1, int(center) - window)
    end = int(center) + window
    target_len = end - start
    # Fallback: if pyfaidx isn't available, attempt naive FASTA parsing
    if Fasta is None:
        seq = _extract_seq_naive(fasta_path, chrom, start, end)
        if len(seq) == 0:
            seq = "N" * target_len
        # Normalize length exactly
        if len(seq) > target_len:
            seq = seq[:target_len]
        elif len(seq) < target_len:
            seq = seq + ("N" * (target_len - len(seq)))
        return seq
    fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    # Retrieve sequence region. pyfaidx slicing is end-exclusive when using 0-based
    # coordinates; here we convert to 0-based start and use an end-exclusive slice.
    # This aims to produce exactly (end - start) bases = 2*window.
    seq = fa[chrom][start-1:end-1]
    seq = str(seq).upper()
    # Normalize length exactly to target_len by trimming or padding with Ns
    if len(seq) > target_len:
        seq = seq[:target_len]
    elif len(seq) < target_len:
        seq = seq + ("N" * (target_len - len(seq)))
    return seq

def load_coordinates(path: str) -> pd.DataFrame:
    """
    Load coordinates/labels from CSV/TSV/BED-like file.
    Expected columns (minimum): chrom, start, end, label  (extra cols ignored).
    """
    p = Path(path)
    if p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
    else:
        df = pd.read_csv(p, sep=None, engine="python")
    required = {"chrom", "start", "end", "label"}
    missing = required - set(df.columns.str.lower())
    # Try lower-casing col names
    df.columns = [c.lower() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode DNA sequence into shape (len(seq), 4) with A,C,G,T (N -> zeros).
    """
    map_idx = {"A":0,"C":1,"G":2,"T":3}
    arr = np.zeros((len(seq),4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = map_idx.get(ch.upper(), None)
        if j is not None:
            arr[i, j] = 1.0
    return arr

def train_val_test_split(n: int, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]
    return train_idx, val_idx, test_idx