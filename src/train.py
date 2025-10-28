from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Checkpoint directory
ckpt_dir = Path("results") / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Optional tensorboard writer can be passed into run_training; keep wandb optional via parameter


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], epoch: int, path: Path):
    state = {"epoch": epoch, "model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, str(path))


def train_epoch(model: nn.Module, dataloader, optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, proj_emb: Optional[nn.Module] = None, encoder_dim: int = 128):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    pbar = tqdm(dataloader)
    for step, batch in enumerate(pbar):
        # move tensors to device
        if "input_ids" in batch:
            batch["input_ids"] = batch["input_ids"].to(device)
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            batch["attention_mask"] = batch["attention_mask"].to(device)
        if "embeddings" in batch:
            batch["embeddings"] = batch["embeddings"].to(device).float()
            # project if needed
            if proj_emb is not None and batch["embeddings"].shape[-1] != encoder_dim:
                batch["embeddings"] = proj_emb(batch["embeddings"])
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = None
        if "input_ids" in batch:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
        elif "embeddings" in batch:
            logits = model(embeddings=batch["embeddings"])
        else:
            logits = model(input_ids=None)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds.append(logits.detach().cpu().numpy())
        trues.append(labels.detach().cpu().numpy())
        pbar.set_description(f"loss={loss.item():.4f}")

    preds = np.concatenate(preds) if len(preds) else np.array([])
    trues = np.concatenate(trues) if len(trues) else np.array([])
    return total_loss / len(dataloader.dataset), preds, trues


def eval_epoch(model: nn.Module, dataloader, loss_fn, device: torch.device, proj_emb: Optional[nn.Module] = None, encoder_dim: int = 128):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if "input_ids" in batch:
                batch["input_ids"] = batch["input_ids"].to(device)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                batch["attention_mask"] = batch["attention_mask"].to(device)
            if "embeddings" in batch:
                batch["embeddings"] = batch["embeddings"].to(device).float()
                if proj_emb is not None and batch["embeddings"].shape[-1] != encoder_dim:
                    batch["embeddings"] = proj_emb(batch["embeddings"])
            labels = batch["label"].to(device)

            if "input_ids" in batch:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
            elif "embeddings" in batch:
                logits = model(embeddings=batch["embeddings"])
            else:
                logits = model(input_ids=None)

            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds.append(logits.detach().cpu().numpy())
            trues.append(labels.detach().cpu().numpy())

    preds = np.concatenate(preds) if len(preds) else np.array([])
    trues = np.concatenate(trues) if len(trues) else np.array([])
    return total_loss / len(dataloader.dataset), preds, trues


def run_training(model: nn.Module,
                 train_loader,
                 val_loader=None,
                 epochs: int = 3,
                 head_lr: float = 1e-3,
                 encoder_lr: float = 1e-5,
                 full_finetune: bool = False,
                 writer=None,
                 use_wandb: bool = False,
                 device: Optional[torch.device] = None,
                 proj_emb: Optional[nn.Module] = None):
    """Orchestrate training with optional projection for one-hot channels.

    The function will inspect train_loader.dataset[0]['input'] to detect if raw inputs are
    one-hot arrays (ndim==2). If so and channel dim != encoder_dim it will create a
    projection layer and include it in the optimizer.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # determine encoder dim from model if possible
    encoder_dim = getattr(model, 'classifier').in_features if hasattr(model, 'classifier') else 128

    # decide param groups
    params = []
    if full_finetune:
        try:
            encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
            params.append({'params': encoder_params, 'lr': encoder_lr})
        except Exception:
            pass
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    params.append({'params': head_params, 'lr': head_lr})

    # check dataset first example to detect raw embedding channels if proj_emb not provided
    if proj_emb is None:
        try:
            sample = train_loader.dataset[0]['input']
            if hasattr(sample, 'ndim') and sample.ndim == 2:
                channels = sample.shape[-1]
                if channels != encoder_dim:
                    proj_emb = nn.Linear(channels, encoder_dim)
                    params.append({'params': proj_emb.parameters(), 'lr': head_lr})
        except Exception:
            # can't inspect dataset, continue without projection
            proj_emb = None
    else:
        # If projection provided by caller (e.g., notebook PROJ), include its params in optimizer
        try:
            params.append({'params': proj_emb.parameters(), 'lr': head_lr})
        except Exception:
            pass
    # Fallback: if still no proj_emb, try inspecting one *batch* from the dataloader
    if proj_emb is None:
        try:
            for batch in train_loader:
                if isinstance(batch, dict) and 'embeddings' in batch:
                    last_dim = batch['embeddings'].shape[-1]
                    if last_dim != encoder_dim:
                        proj_emb = nn.Linear(last_dim, encoder_dim)
                        params.append({'params': proj_emb.parameters(), 'lr': head_lr})
                    break
                # if collate returns tensors under other keys, try to coerce
                if isinstance(batch, dict) and 'input' in batch:
                    x = batch['input']
                    if hasattr(x, 'ndim') and x.ndim == 2:
                        last_dim = x.shape[-1]
                        if last_dim != encoder_dim:
                            proj_emb = nn.Linear(last_dim, encoder_dim)
                            params.append({'params': proj_emb.parameters(), 'lr': head_lr})
                        break
        except Exception:
            # dataloader iteration may be expensive or not allowed here; ignore silently
            pass

    optimizer = torch.optim.Adam(params)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_preds, train_trues = train_epoch(model, train_loader, optimizer, loss_fn, device, proj_emb=proj_emb, encoder_dim=encoder_dim)
        if writer is not None:
            try:
                writer.add_scalar('loss/train', train_loss, epoch)
            except Exception:
                pass
        if use_wandb:
            try:
                import wandb
                wandb.log({'loss/train': train_loss, 'epoch': epoch})
            except Exception:
                pass

        print(f'Epoch {epoch} train_loss={train_loss:.4f} time={(time.time()-t0):.1f}s')
        if val_loader is not None:
            val_loss, val_preds, val_trues = eval_epoch(model, val_loader, loss_fn, device, proj_emb=proj_emb, encoder_dim=encoder_dim)
            if writer is not None:
                try:
                    writer.add_scalar('loss/val', val_loss, epoch)
                except Exception:
                    pass
            if use_wandb:
                try:
                    import wandb
                    wandb.log({'loss/val': val_loss, 'epoch': epoch})
                except Exception:
                    pass

            print(f'Epoch {epoch} val_loss={val_loss:.4f}')

            # compute AUC if possible
            try:
                from sklearn.metrics import roc_auc_score
                if len(val_preds) and len(np.unique(val_trues)) > 1:
                    auc = roc_auc_score(val_trues, val_preds)
                    print('Val AUC:', auc)
                    if writer is not None:
                        try:
                            writer.add_scalar('val/auc', auc, epoch)
                        except Exception:
                            pass
                    if use_wandb:
                        try:
                            import wandb
                            wandb.log({'val/auc': auc, 'epoch': epoch})
                        except Exception:
                            pass
            except Exception:
                pass

            # checkpoint
            ckpt_path = ckpt_dir / f'probe_epoch{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print('Saved checkpoint to', ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, ckpt_dir / 'probe_best.pt')
                print('Saved best checkpoint')

    return model
