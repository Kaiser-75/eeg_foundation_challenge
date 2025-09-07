from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from braindecode.models import EEGNetv4
from dataloader import SuSWindowDataset, PreprocessConfig, MAX_CH, OUT_T

# =========================
# HYPERPARAMETERS (edit here)
# =========================
BASE_DIR              = Path(os.environ.get("EEG_BASE_DIR", "."))  # root folder (contains competition_data/*)
RELEASES              = None          # e.g. ["R1","R2"] or None for all
MAX_FILES             = None          # e.g. 500 to cap number of recordings, or None
STRIDE_SEC            = 1.0           # SSL window stride (sec)

EPOCHS                = 10
BATCH_SIZE            = 64
LR                    = 1e-3
WEIGHT_DECAY          = 1e-4
EMB_DIM               = 128           # encoder embedding size (also EEGNetv4 n_outputs)
PROJ_DIM              = 128           # projection head size
TEMP                  = 0.2           # NT-Xent temperature
LAMBDA_T              = 1.0           # weight for temporal contrastive loss
LAMBDA_S              = 1.0           # weight for spatial contrastive loss
MAX_STEPS_PER_EPOCH   = None       # 500/2000 or None to use full epoch
LOG_EVERY             = 100
SAVE_DIR              = Path("checkpoints")
SEED                  = 42

NUM_WORKERS           = 8
PERSISTENT_WORKERS    = True
PREFETCH_FACTOR       = 4

# =========================
# Utilities
# =========================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cosine_lr(optimizer, base_lr: float, epochs: int, steps_per_epoch: int, warmup_epochs: int = 1):
    def lr_lambda(step):
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================
# EEGNetv4-backed encoder
# =========================
class EEGV4Encoder(nn.Module):
    """
    Wrap EEGNetv4 so that it outputs an embedding vector of size EMB_DIM.
    We set n_outputs=EMB_DIM and use logits as features.
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGNetv4(
            n_chans=in_ch,
            n_outputs=emb,             # logits size = embedding size
            n_times=T,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EEGNetv4 expects [B, C, T]; returns [B, emb]
        return self.backbone(x)

# =========================
# SimCLR
# =========================
class SimCLR(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM, proj: int = PROJ_DIM):
        super().__init__()
        self.encoder = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.projector = nn.Sequential(
            nn.Linear(emb, emb, bias=False),
            nn.BatchNorm1d(emb),
            nn.GELU(),
            nn.Linear(emb, proj, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)            # [B, emb]
        z = self.projector(h)          # [B, proj]
        z = F.normalize(z, dim=-1)
        return h, z

# =========================
# Augmentations
# =========================
def rand_time_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
    if max_shift <= 0:
        return x
    B, C, T = x.shape
    shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(B,), device=x.device)
    out = torch.empty_like(x)
    for i, s in enumerate(shifts):
        out[i] = torch.roll(x[i], shifts=int(s.item()), dims=-1)
    return out

def rand_time_mask(x: torch.Tensor, max_frac: float = 0.2, num_masks: int = 2) -> torch.Tensor:
    if max_frac <= 0 or num_masks <= 0:
        return x
    B, C, T = x.shape
    out = x.clone()
    max_len = max(1, int(max_frac * T))
    for i in range(B):
        for _ in range(num_masks):
            w = torch.randint(1, max_len + 1, (1,), device=x.device).item()
            s = torch.randint(0, max(1, T - w + 1), (1,), device=x.device).item()
            out[i, :, s:s + w] = 0.0
    return out

def add_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)

def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0:
        return x
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask

def channel_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0:
        return x
    B, C, T = x.shape
    scale = (1.0 + sigma * torch.randn(B, C, 1, device=x.device))
    return x * scale

def make_temporal_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    v2 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    return v1, v2

def make_spatial_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    v2 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    return v1, v2

# =========================
# NT-Xent loss (compute in fp32, mask with -inf)
# =========================
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMP) -> torch.Tensor:
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    z = torch.cat([z1, z2], dim=0)                       # [2B, D], fp32
    sim = (z @ z.T) / float(temperature)                 # [2B, 2B], fp32
    sim.fill_diagonal_(-float("inf"))                    # avoid fp16 overflow
    B = z1.shape[0]
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)   # [2B]
    return F.cross_entropy(sim, targets)

# =========================
# DataLoader (custom collate)
# =========================
def collate_unlabeled(batch):
    xs = [b[0] if isinstance(b, (list, tuple)) else b for b in batch]
    return torch.stack(xs, dim=0)

def build_loader(base_dir: Path, batch_size: int, releases=None, max_files=None, stride: float = 1.0):
    cfg = PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True, avg_ref=True,
        resample_hz=100.0,
        amp_clip_uv=200.0, window_standardize=True,
    )
    ds = SuSWindowDataset(
        base_dir=base_dir,
        releases=releases,
        max_files=max_files,
        preprocess=cfg,
        preload=False,
        stride_sec=stride,
        verbose="INFO",
    )
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        collate_fn=collate_unlabeled,
    )
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    loader = DataLoader(ds, **dl_kwargs)
    return ds, loader

# =========================
# Train
# =========================
def train():
    set_seed(SEED)
    device = get_device()

    base_dir = BASE_DIR.expanduser().resolve()
    assert base_dir.exists(), f"Base dir not found: {base_dir}"

    ds, loader = build_loader(
        base_dir=base_dir,
        batch_size=BATCH_SIZE,
        releases=RELEASES,
        max_files=MAX_FILES,
        stride=STRIDE_SEC,
    )

    print(f"SimCLR SSL on SuS | windows={len(ds)} | device={device.type}")

    model = SimCLR(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM, proj=PROJ_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(loader) if MAX_STEPS_PER_EPOCH is None else min(len(loader), MAX_STEPS_PER_EPOCH)
    scheduler = cosine_lr(optimizer, base_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, warmup_epochs=1)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model.train()
    for ep in range(1, EPOCHS + 1):
        running = {"temp": 0.0, "spat": 0.0, "total": 0.0}
        for it, X in enumerate(loader, start=1):   # X: [B, 129, 200]
            if MAX_STEPS_PER_EPOCH is not None and it > MAX_STEPS_PER_EPOCH:
                break

            X = X.to(device, non_blocking=True)

            # two kinds of positive pairs
            Xt1, Xt2 = make_temporal_views(X)
            Xs1, Xs2 = make_spatial_views(X)

            with autocast(enabled=(device.type == "cuda")):
                _, zt1 = model(Xt1)
                _, zt2 = model(Xt2)
                _, zs1 = model(Xs1)
                _, zs2 = model(Xs2)

                loss_t = nt_xent_loss(zt1, zt2, temperature=TEMP)
                loss_s = nt_xent_loss(zs1, zs2, temperature=TEMP)
                loss   = LAMBDA_T * loss_t + LAMBDA_S * loss_s

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["temp"]  += loss_t.item()
            running["spat"]  += loss_s.item()
            running["total"] += loss.item()

            if it % LOG_EVERY == 0 or it == 1:
                lr = scheduler.get_last_lr()[0]
                avg_t = running["temp"] / it
                avg_s = running["spat"] / it
                avg_total = running["total"] / it
                print(f"epoch {ep:02d} | step {it:05d}/{steps_per_epoch:05d} | "
                      f"lr {lr:.3e} | temp {avg_t:.4f} | spat {avg_s:.4f} | total {avg_total:.4f}")

        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hparams": {
                "emb": EMB_DIM, "proj": PROJ_DIM, "temp": TEMP,
                "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
                "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "backbone": "EEGNetv4",
            },
        }
        torch.save(ckpt, SAVE_DIR / f"simclr_sus_epoch{ep:03d}.pt")
        torch.save(ckpt, SAVE_DIR / "simclr_sus_latest.pt")
        print(f"[ckpt] saved → {SAVE_DIR / f'simclr_sus_epoch{ep:03d}.pt'}")

    print("Training complete.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    train()
