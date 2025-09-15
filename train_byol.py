from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Tuple, Dict

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
BASE_DIR              = Path(os.environ.get("EEG_BASE_DIR", "."))  # root (contains competition_data/*)
RELEASES              = None          # e.g. ["R1","R2"] or None for all
MAX_FILES             = None          # cap number of files or None
STRIDE_SEC            = 1.0           # window stride for unlabeled SuS

EPOCHS                = 10
BATCH_SIZE            = 64
LR                    = 1e-3
WEIGHT_DECAY          = 1e-4

EMB_DIM               = 128           # EEGNetv4 n_outputs (logits-as-embedding)
PROJ_DIM              = 128           # projector/predictor dimension

MOMENTUM_BASE         = 0.996         # EMA for target network (can schedule upward)
MAX_STEPS_PER_EPOCH   = None          # 500/2000 or None to use full epoch
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

def momentum_schedule(base_m: float, step: int, total_steps: int) -> float:
    # smoothly increase m toward 1.0 during training
    if total_steps <= 0:
        return base_m
    cos_term = (1 + math.cos(math.pi * step / total_steps)) / 2.0  # 1 -> 0
    return 1.0 - (1.0 - base_m) * cos_term

# =========================
# Augmentations (BYOL uses two random views)
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

def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0:
        return x
    mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > p).float()
    return x * mask

def channel_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0:
        return x
    scale = (1.0 + sigma * torch.randn(x.size(0), x.size(1), 1, device=x.device))
    return x * scale

def add_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)

def make_view(x: torch.Tensor) -> torch.Tensor:
    # a single augmentation pipeline for BYOL
    v = rand_time_shift(x, max_shift=10)
    v = rand_time_mask(v, max_frac=0.15, num_masks=2)
    v = channel_dropout(v, p=0.10)
    v = channel_jitter(v, sigma=0.02)
    v = add_noise(v, sigma=0.01)
    return v

# =========================
# Small MLPs for projector/predictor
# =========================
def mlp(in_dim: int, hid: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False),
        nn.BatchNorm1d(hid),
        nn.GELU(),
        nn.Linear(hid, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False),  # like BYOL: BN w/o affine on z
    )

def predictor_mlp(in_dim: int, hid: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False),
        nn.BatchNorm1d(hid),
        nn.GELU(),
        nn.Linear(hid, out_dim)  # predictor keeps affine
    )

# =========================
# EEGNetv4-backed encoders
# =========================
class EEGV4Encoder(nn.Module):
    """EEGNetv4 outputs an embedding vector of size EMB_DIM (logits-as-embedding)."""
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGNetv4(n_chans=in_ch, n_outputs=emb, n_times=T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, EMB_DIM]

# =========================
# BYOL model
# =========================
class BYOL(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM, proj: int = PROJ_DIM):
        super().__init__()
        # online network
        self.online_encoder = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.online_projector = mlp(emb, emb, proj)
        self.online_predictor = predictor_mlp(proj, emb, proj)

        # target network (EMA copy; no predictor)
        self.target_encoder = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.target_projector = mlp(emb, emb, proj)

        # initialize target with online weights
        self._copy_params(self.target_encoder, self.online_encoder, copy_bn_buffers=True)
        self._copy_params(self.target_projector, self.online_projector, copy_bn_buffers=True)

    @torch.no_grad()
    def _copy_params(self, tgt: nn.Module, src: nn.Module, copy_bn_buffers: bool = True):
        for (name_t, p_t), (_, p_s) in zip(tgt.named_parameters(), src.named_parameters()):
            p_t.data.copy_(p_s.data)
        if copy_bn_buffers:
            for (name_t, b_t), (_, b_s) in zip(tgt.named_buffers(), src.named_buffers()):
                b_t.data.copy_(b_s.data)

    @torch.no_grad()
    def update_momentum(self, m: float):
        # EMA update: theta_t = m*theta_t + (1-m)*theta_o
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))
        for p_t, p_o in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))

    def forward_online(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.online_encoder(x)          # [B, EMB_DIM]
        z = self.online_projector(h)        # [B, PROJ_DIM]
        q = self.online_predictor(z)        # [B, PROJ_DIM]
        return z, q

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        h = self.target_encoder(x)
        z = self.target_projector(h)
        return z

# =========================
# Loss: BYOL uses cosine similarity between predictor(p) and target(z)
# =========================
def byol_cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # both p and z: [B, D]; stop-grad is applied to z by caller
    p = F.normalize(p.float(), dim=-1)
    z = F.normalize(z.float(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()

# =========================
# DataLoader (unlabeled)
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

    print(f"BYOL SSL on SuS | windows={len(ds)} | device={device.type}")

    model = BYOL(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM, proj=PROJ_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(loader) if MAX_STEPS_PER_EPOCH is None else min(len(loader), MAX_STEPS_PER_EPOCH)
    total_steps = EPOCHS * steps_per_epoch
    scheduler = cosine_lr(optimizer, base_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, warmup_epochs=1)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()
    for ep in range(1, EPOCHS + 1):
        running = 0.0
        for it, X in enumerate(loader, start=1):
            if MAX_STEPS_PER_EPOCH is not None and it > MAX_STEPS_PER_EPOCH:
                break

            X = X.to(device, non_blocking=True)
            # two random views for BYOL
            v1 = make_view(X)
            v2 = make_view(X)

            with autocast(enabled=(device.type == "cuda")):
                # online
                z1_o, q1 = model.forward_online(v1)
                z2_o, q2 = model.forward_online(v2)
                # target (stop-grad)
                with torch.no_grad():
                    z1_t = model.forward_target(v1)
                    z2_t = model.forward_target(v2)

                loss = 0.5 * (byol_cosine_loss(q1, z2_t) + byol_cosine_loss(q2, z1_t))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update for target network
            m = momentum_schedule(MOMENTUM_BASE, global_step, total_steps)
            model.update_momentum(m)

            scheduler.step()
            running += loss.item()
            global_step += 1

            if it % LOG_EVERY == 0 or it == 1:
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {ep:02d} | step {it:05d}/{steps_per_epoch:05d} | lr {lr:.3e} | loss {running/it:.4f} | m {m:.5f}")

        # ---- Save checkpoint ----
        # Native state dict (for resuming BYOL)
        model_sd = model.state_dict()

        # Add compatibility keys so train_supervised can load with "encoder.backbone.*"
        compat = {}
        for k, v in model.online_encoder.backbone.state_dict().items():
            compat["encoder.backbone." + k] = v

        merged = dict(model_sd)
        merged.update(compat)

        ckpt = {
            "epoch": ep,
            "model": merged,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hparams": {
                "emb": EMB_DIM, "proj": PROJ_DIM,
                "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "backbone": "EEGNetv4", "ssl": "BYOL", "momentum_base": MOMENTUM_BASE,
            },
        }
        torch.save(ckpt, SAVE_DIR / f"byol_sus_epoch{ep:03d}.pt")
        torch.save(ckpt, SAVE_DIR / "byol_sus_latest.pt")
        print(f"[ckpt] saved → {SAVE_DIR / f'byol_sus_epoch{ep:03d}.pt'}")

    print("Training complete.")

if __name__ == "__main__":
    train()
