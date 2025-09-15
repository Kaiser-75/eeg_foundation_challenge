from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler

from braindecode.models import EEGNetv4
from dataloader_c2 import (
    make_ssl_dataset,
    PreprocessConfig,
    MAX_CH,
    OUT_T,
)

# =========================
# HYPERPARAMETERS
# =========================
BASE_DIR              = Path(os.environ.get("EEG_BASE_DIR", "competition_data"))
RELEASES              = None            # e.g., ["R1","R2"] or None for all
MAX_STEPS_PER_EPOCH   = None            # e.g., 2000 to cap per-epoch steps for debugging; None uses full epoch
EPOCHS                = 10
BATCH_SIZE            = 256
LR                    = 1e-3
WEIGHT_DECAY          = 1e-4

# Embedding/projection
EMB_DIM               = 128             # EEGNetv4 n_outputs (feature dim)
PROJ_DIM              = 128             # projection head output

# Loss & views
TEMP                  = 0.2             # NT-Xent temperature
LAMBDA_TEMPORAL       = 1.0             # weight for temporal view loss
LAMBDA_SPATIAL        = 1.0             # weight for spatial view loss
LAMBDA_FFT            = 0.3            # 0 = disable freq-domain view loss; 0.3= mild

# Data & loader
STRIDE_SEC            = 1.0             # SSL window stride for all tasks
NUM_WORKERS           = 8
PERSISTENT_WORKERS    = True
PREFETCH_FACTOR       = 4
PIN_MEMORY            = torch.cuda.is_available()
PRELOAD               = False           # set True for faster reuse (uses host RAM); may explode memory use

# Logging / saving
LOG_EVERY             = 100
SAVE_DIR              = Path("checkpoints_foundation")
SEED                  = 42

# =========================
# Utils
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
    EEGNetv4 with n_outputs=EMB_DIM; we treat logits as embeddings.
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGNetv4(
            n_chans=in_ch,
            n_outputs=emb,
            n_times=T,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, emb]
        return self.backbone(x)

# =========================
# SimCLR model
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

# Optional frequency-domain perturb (disabled by default via LAMBDA_FFT=0)
def fft_perturb(x: torch.Tensor, mag_sigma: float = 0.02) -> torch.Tensor:
    """
    Simple magnitude jitter in frequency domain; keeps phase, mild spectral augmentation.
    """
    B, C, T = x.shape
    Xf = torch.fft.rfft(x, dim=-1)                          # [B, C, F]
    mag = Xf.abs()
    pha = torch.angle(Xf)
    mag = mag * (1.0 + mag_sigma * torch.randn_like(mag))
    Xf_new = mag * torch.exp(1j * pha)
    x_new = torch.fft.irfft(Xf_new, n=T, dim=-1)
    return x_new.real

def make_temporal_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    v2 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    return v1, v2

def make_spatial_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    v2 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    return v1, v2

# =========================
# NT-Xent loss (fp32 logits; diagonal masked)
# =========================
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMP) -> torch.Tensor:
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    z = torch.cat([z1, z2], dim=0)                       # [2B, D]
    sim = (z @ z.T) / float(temperature)                 # [2B, 2B]
    sim.fill_diagonal_(-float("inf"))
    B = z1.shape[0]
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)   # [2B]
    return F.cross_entropy(sim, targets)

# =========================
# DataLoader (simple collate)
# =========================
def collate_unlabeled(batch):
    # batch: list[Tensor [C,T]]
    return torch.stack(batch, dim=0)

def build_loader():
    cfg = PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True,
        avg_ref=True, resample_hz=100.0,
        amp_clip_uv=800.0, window_standardize=True,
    )
    ds = make_ssl_dataset(
        base_dir=BASE_DIR,
        releases=RELEASES,
        preprocess=cfg,
        stride_sec=STRIDE_SEC,
        preload=PRELOAD,
        verbose="INFO",
    )
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_unlabeled,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    loader = DataLoader(ds, **kwargs)
    return ds, loader

# =========================
# Train
# =========================
def train():
    set_seed(SEED)
    device = get_device()
    print(f"[foundation] SSL on all tasks (CCD pretrial only) | device={device.type}")
    print(f"BASE_DIR = {BASE_DIR.resolve()}")

    ds, loader = build_loader()
    steps_per_epoch = len(loader) if MAX_STEPS_PER_EPOCH is None else min(len(loader), MAX_STEPS_PER_EPOCH)
    print(f"steps/epoch = {steps_per_epoch} (of {len(loader)})")

    model = SimCLR(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM, proj=PROJ_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = cosine_lr(optimizer, base_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, warmup_epochs=1)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model.train()
    gstep = 0
    for ep in range(1, EPOCHS + 1):
        running = {"temp": 0.0, "spat": 0.0, "fft": 0.0, "total": 0.0}
        for it, X in enumerate(loader, start=1):
            if MAX_STEPS_PER_EPOCH is not None and it > MAX_STEPS_PER_EPOCH:
                break

            X = X.to(device, non_blocking=True)  # [B, C, T]

            # make positive pairs
            Xt1, Xt2 = make_temporal_views(X)
            Xs1, Xs2 = make_spatial_views(X)

            with autocast(enabled=(device.type == "cuda")):
                # temporal
                _, zt1 = model(Xt1)
                _, zt2 = model(Xt2)
                loss_t = nt_xent_loss(zt1, zt2, temperature=TEMP)

                # spatial
                _, zs1 = model(Xs1)
                _, zs2 = model(Xs2)
                loss_s = nt_xent_loss(zs1, zs2, temperature=TEMP)

                # frequency-domain view
                loss_f = torch.tensor(0.0, device=device)
                if LAMBDA_FFT > 0.0:
                    Xf1 = fft_perturb(X)
                    Xf2 = fft_perturb(X)
                    _, zf1 = model(Xf1)
                    _, zf2 = model(Xf2)
                    loss_f = nt_xent_loss(zf1, zf2, temperature=TEMP)

                loss = LAMBDA_TEMPORAL * loss_t + LAMBDA_SPATIAL * loss_s + LAMBDA_FFT * loss_f

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            gstep += 1

            running["temp"]  += loss_t.item()
            running["spat"]  += loss_s.item()
            running["fft"]   += (loss_f.item() if isinstance(loss_f, torch.Tensor) else 0.0)
            running["total"] += loss.item()

            if it % LOG_EVERY == 0 or it == 1:
                lr = scheduler.get_last_lr()[0]
                avg_t = running["temp"] / it
                avg_s = running["spat"] / it
                avg_f = running["fft"]  / max(1, it)
                avg_total = running["total"] / it
                print(f"epoch {ep:02d} | step {it:05d}/{steps_per_epoch:05d} | "
                      f"lr {lr:.3e} | temp {avg_t:.4f} | spat {avg_s:.4f} | fft {avg_f:.4f} | total {avg_total:.4f}")

        # save checkpoint each epoch
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hparams": {
                "emb": EMB_DIM, "proj": PROJ_DIM, "temp": TEMP,
                "lambda_t": LAMBDA_TEMPORAL, "lambda_s": LAMBDA_SPATIAL, "lambda_fft": LAMBDA_FFT,
                "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "backbone": "EEGNetv4",
            },
        }
        torch.save(ckpt, SAVE_DIR / f"foundation_simclr_epoch{ep:03d}.pt")
        torch.save(ckpt, SAVE_DIR / "foundation_simclr_latest.pt")
        print(f"[ckpt] saved → {SAVE_DIR / f'foundation_simclr_epoch{ep:03d}.pt'}")

    print("Training complete.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # keep thread usage predictable on CPU ops (IO/preproc)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    train()
