"""
Picklable collate functions for use with PyTorch DataLoader workers.
Keep these at module top-level so multiprocessing can pickle the callable.
"""
from typing import List
import torch
import numpy as np


def collate_for_workers(batch: List[dict]):
    """Top-level collate function suitable for DataLoader with num_workers>0.

    Handles common cases:
    - inputs are 2D one-hot arrays (seq_len x channels) -> stacked into embeddings tensor
    - inputs are 1D token id arrays -> padded into input_ids + attention_mask
    - otherwise attempts to stack into embeddings
    """
    labels = torch.tensor([int(b['label']) for b in batch], dtype=torch.float)
    inputs = [b['input'] for b in batch]

    # Convert to tensors where possible
    arrs = []
    for x in inputs:
        if isinstance(x, np.ndarray):
            arrs.append(torch.as_tensor(x))
        else:
            try:
                arrs.append(torch.as_tensor(np.asarray(x)))
            except Exception:
                arrs.append(x)

    # If first element is a 2D tensor -> treat as embeddings (batch, seq_len, channels)
    first = arrs[0]
    if isinstance(first, torch.Tensor) and first.ndim == 2:
        stacked = torch.stack(arrs)
        return {'embeddings': stacked, 'label': labels}

    # If 1D tensors -> assume token ids and pad
    if isinstance(first, torch.Tensor) and first.ndim == 1:
        lengths = [a.shape[0] for a in arrs]
        maxlen = max(lengths)
        batch_ids = torch.zeros((len(arrs), maxlen), dtype=torch.long)
        for i, a in enumerate(arrs):
            batch_ids[i, :a.shape[0]] = a.long()
        attention_mask = (batch_ids != 0).long()
        return {'input_ids': batch_ids, 'attention_mask': attention_mask, 'label': labels}

    # Fallback: try stacking with numpy then convert
    try:
        stacked = torch.as_tensor(np.stack(inputs))
        return {'embeddings': stacked, 'label': labels}
    except Exception:
        # Last resort: return raw Python objects (will be slow)
        return {'input': inputs, 'label': labels}
