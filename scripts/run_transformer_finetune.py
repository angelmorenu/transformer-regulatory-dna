"""Small runner to execute a quick 1-epoch training run for the transformer fine-tuning notebook.
This script mirrors the notebook logic but runs as a script so DataLoader can use num_workers>0 safely.
"""
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys

# Ensure project root is on sys.path so `import src` works when running this script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SequenceDataset(Dataset):
    def __init__(self, path: str):
        npz = np.load(path, allow_pickle=True)
        if 'X' in npz and 'y' in npz:
            self.inputs = npz['X']
            self.labels = npz['y']
        elif 'inputs' in npz and 'labels' in npz:
            self.inputs = npz['inputs']
            self.labels = npz['labels']
        elif 'arr_0' in npz and 'arr_1' in npz:
            self.inputs = npz['arr_0']
            self.labels = npz['arr_1']
        else:
            keys = list(npz.keys())
            if len(keys) >= 2:
                self.inputs = npz[keys[0]]
                self.labels = npz[keys[1]]
            else:
                raise ValueError(f'Unrecognized .npz structure: {keys} in {path}')
        assert len(self.inputs) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input': self.inputs[idx], 'label': int(self.labels[idx])}


def main():
    # Try to import top-level collate
    collate_for_workers = None
    try:
        from src.collate import collate_for_workers
        collate_for_workers = collate_for_workers
    except Exception:
        try:
            from src.collate import collate_for_workers as collate_for_workers
        except Exception:
            collate_for_workers = None

    # locate data
    base_dir = Path('data') / 'processed'
    if not base_dir.exists():
        candidates = list(Path('.').rglob('data/processed'))
        if candidates:
            base_dir = candidates[0]

    train_path = base_dir / 'train.npz'
    val_path = base_dir / 'val.npz'
    test_path = base_dir / 'test.npz'

    if not train_path.exists():
        # search for any train.npz
        candidates = list(Path('.').rglob('train.npz'))
        if candidates:
            train_path = candidates[0]
            val_path = train_path.parent / 'val.npz'
            test_path = train_path.parent / 'test.npz'

    print('Using train path:', train_path)
    if not train_path.exists():
        raise FileNotFoundError('train.npz not found; run preprocessing notebook first')

    train_ds = SequenceDataset(str(train_path))
    val_ds = SequenceDataset(str(val_path)) if val_path.exists() else None
    test_ds = SequenceDataset(str(test_path)) if test_path.exists() else None
    print('Loaded datasets: train=%d, val=%s, test=%s' % (len(train_ds), len(val_ds) if val_ds is not None else 'None', len(test_ds) if test_ds is not None else 'None'))

    # small model: LinearProbe with placeholder encoder
    class _TinyEncoder(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.dim = dim
        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            if input_ids is not None:
                b = input_ids.shape[0]
                l = input_ids.shape[1]
            else:
                b = 1
                l = 64
            return type('X', (), {'last_hidden_state': torch.randn(b, l, self.dim, device=DEVICE)})

    class LinearProbe(nn.Module):
        def __init__(self, encoder, encoder_dim: int, num_labels: int = 1, freeze_encoder: bool = True):
            super().__init__()
            self.encoder = encoder
            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            self.pool = lambda x: x.mean(dim=1)
            self.classifier = nn.Linear(encoder_dim, num_labels)
        def forward(self, input_ids=None, attention_mask=None, embeddings=None):
            if embeddings is None:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = outputs.last_hidden_state
            else:
                last_hidden = embeddings
            pooled = self.pool(last_hidden)
            logits = self.classifier(pooled)
            return logits.squeeze(-1)

    encoder = _TinyEncoder(128)
    probe = LinearProbe(encoder, encoder_dim=128, num_labels=1, freeze_encoder=True).to(DEVICE)
    print('Probe initialized. Trainable params:', sum(p.numel() for p in probe.parameters() if p.requires_grad))

    # collate selection and DataLoader creation
    batch_size = 8
    num_workers = 2
    # simple_collate is defined up-front so it can be used as a fallback in any branch
    def simple_collate(batch):
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float)
        inputs = [b['input'] for b in batch]
        arrs = [torch.as_tensor(x) for x in inputs]
        if arrs[0].ndim == 2:
            return {'embeddings': torch.stack(arrs), 'label': labels}
        else:
            lengths = [a.shape[0] for a in arrs]
            maxlen = max(lengths)
            batch_ids = torch.zeros((len(arrs), maxlen), dtype=torch.long)
            for i, a in enumerate(arrs):
                batch_ids[i, :a.shape[0]] = a
            return {'input_ids': batch_ids, 'attention_mask': (batch_ids!=0).long(), 'label': labels}

    use_collate = collate_for_workers if collate_for_workers is not None else None
    if use_collate is None:
        collate_fn = simple_collate
        num_workers_try = 0
    else:
        collate_fn = use_collate
        num_workers_try = num_workers

    # create dataloaders with fallback
    try:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers_try)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=num_workers_try) if val_ds is not None else None
        test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=num_workers_try) if test_ds is not None else None
        print('Created DataLoaders with num_workers=', num_workers_try)
    except Exception as e:
        print('Failed to create multi-worker dataloaders:', e)
        collate_fn = simple_collate
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=0) if val_ds is not None else None
        test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=0) if test_ds is not None else None
        print('Created DataLoaders with num_workers=0 fallback')

    # Use top-level training utilities
    try:
        from src.train import run_training, set_seed as train_set_seed
    except Exception as e:
        raise ImportError('Could not import training helpers from src.train: ' + str(e))

    # ensure train module seed behavior is consistent
    train_set_seed(42)

    # Build a projection layer automatically if dataset uses one-hot embeddings
    proj_emb = None
    try:
        encoder_dim = probe.classifier.in_features
        sample = train_loader.dataset[0]
        x = sample['input']
        if hasattr(x, 'ndim') and x.ndim == 2 and x.shape[-1] != encoder_dim:
            proj_emb = nn.Linear(x.shape[-1], encoder_dim).to(DEVICE)
            print(f'Auto-created proj_emb: {x.shape[-1]} -> {encoder_dim}')
    except Exception:
        # safe fallback: let src.train attempt to detect via batch inspection
        proj_emb = None

    print('Starting 1-epoch training (via src.train.run_training)...')
    trained = run_training(probe, train_loader, val_loader=val_loader, epochs=1, head_lr=1e-3, encoder_lr=1e-5, full_finetune=False, writer=None, use_wandb=False, device=DEVICE, proj_emb=proj_emb)
    print('Finished quick training run')


if __name__ == '__main__':
    main()
