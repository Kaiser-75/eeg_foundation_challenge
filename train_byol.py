from __future__ import annotations
import os, math
from pathlib import Path
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from braindecode.models import EEGNetv4
from dataloader import SuSWindowDataset, PreprocessConfig, MAX_CH, OUT_HZ

# =========================
# HYPERPARAMETERS (edit here)
# =========================
BASE_DIR              = Path(os.environ.get("EEG_BASE_DIR", "competition_data"))
RELEASES              = None          # e.g. ["R1","R2"] or None for all
MAX_FILES             = None          # cap files for quick runs (e.g., 500) or None

# SSL windowing (SuS)
WIN_SEC_SSL           = 6.0           # 6-second windows for SSL (OK for pretraining)
STRIDE_SEC            = 3.0           # stride between SSL windows

# Train
EPOCHS                = 10
BATCH_SIZE            = 128
LR                    = 1e-3
WEIGHT_DECAY          = 1e-4
MAX_STEPS_PER_EPOCH   = None          # e.g., 2000 for speed, or None for full
LOG_EVERY             = 100
SAVE_DIR              = Path("checkpoints_c1")
SEED                  = 42

# Model dims
EMB_DIM               = 128           # EEGNetv4 n_outputs (embedding dim)
PROJ_DIM              = 128           # projector/predictor hidden/output

# BYOL momentum (EMA) schedule
MOMENTUM_BASE         = 0.996         # start EMA; will smoothly → 1.0

# Loader perf
NUM_WORKERS           = 8
PERSISTENT_WORKERS    = True
PREFETCH_FACTOR       = 4
PIN_MEMORY            = torch.cuda.is_available()
PRELOAD               = False         # set True to cache preprocessed raws (big RAM)

# =========================
# Utilities
# =========================
def set_seed(seed: int = SEED):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # smoothly increases m toward 1.0
    if total_steps <= 0:
        return base_m
    cos_term = (1 + math.cos(math.pi * step / total_steps)) / 2.0  # 1 → 0
    return 1.0 - (1.0 - base_m) * cos_term

# =========================
# Augmentations (BYOL: two independent views)
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
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask

def channel_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0:
        return x
    B, C, T = x.shape
    scale = (1.0 + sigma * torch.randn(B, C, 1, device=x.device))
    return x * scale

def add_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)

def make_view(x: torch.Tensor) -> torch.Tensor:
    v = rand_time_shift(x, max_shift=10)
    v = rand_time_mask(v, max_frac=0.15, num_masks=2)
    v = channel_dropout(v, p=0.10)
    v = channel_jitter(v, sigma=0.02)
    v = add_noise(v, sigma=0.01)
    return v

# =========================
# Small MLPs for projector/predictor
# =========================
def projector_mlp(in_dim: int, hid: int, out_dim: int) -> nn.Sequential:
    # BYOL-style: BN without affine on the output
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False),
        nn.BatchNorm1d(hid),
        nn.GELU(),
        nn.Linear(hid, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False),
    )

def predictor_mlp(in_dim: int, hid: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False),
        nn.BatchNorm1d(hid),
        nn.GELU(),
        nn.Linear(hid, out_dim)  # keep affine
    )

# =========================
# EEGNetv4-backed encoder (safe pooling)
# =========================
class EEGV4Encoder(nn.Module):
    def __init__(self, in_ch: int, T: int, emb: int):
        super().__init__()
        self.backbone = EEGNetv4(n_chans=in_ch, n_outputs=emb, n_times=T)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)         # [B, emb] or [B, emb, t']
        if h.ndim == 3:
            h = h.mean(-1)           # global avg pool over time if present
        return h                      # [B, emb]

# =========================
# BYOL model
# =========================
class BYOL(nn.Module):
    def __init__(self, in_ch: int, T: int, emb: int, proj: int):
        super().__init__()
        # online
        self.online_encoder   = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.online_projector = projector_mlp(emb, emb, proj)
        self.online_predictor = predictor_mlp(proj, emb, proj)
        # target (EMA)
        self.target_encoder   = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.target_projector = projector_mlp(emb, emb, proj)
        # init target = online
        self._copy_params(self.target_encoder,   self.online_encoder)
        self._copy_params(self.target_projector, self.online_projector, copy_bn_buffers=True)

    @torch.no_grad()
    def _copy_params(self, tgt: nn.Module, src: nn.Module, copy_bn_buffers: bool = True):
        for p_t, p_s in zip(tgt.parameters(), src.parameters()):
            p_t.data.copy_(p_s.data)
        if copy_bn_buffers:
            for b_t, b_s in zip(tgt.buffers(), src.buffers()):
                b_t.data.copy_(b_s.data)

    @torch.no_grad()
    def update_momentum(self, m: float):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))
        for p_t, p_o in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))

    def forward_online(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.online_encoder(x)
        z = self.online_projector(h)
        q = self.online_predictor(z)
        return z, q

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        h = self.target_encoder(x)
        z = self.target_projector(h)
        return z

# =========================
# Loss (BYOL cosine)
# =========================
def byol_cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p.float(), dim=-1)
    z = F.normalize(z.float(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()

# =========================
# DataLoader
# =========================
def collate_unlabeled(batch):
    xs = [b[0] if isinstance(b, (list, tuple)) else b for b in batch]
    return torch.stack(xs, dim=0)

def build_loader():
    cfg = PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True, avg_ref=True,
        resample_hz=100.0,
        amp_clip_uv=600.0, window_standardize=True,
    )
    ds = SuSWindowDataset(
        base_dir=BASE_DIR,
        releases=RELEASES,
        max_files=MAX_FILES,
        preprocess=cfg,
        preload=PRELOAD,
        stride_sec=STRIDE_SEC,
        win_sec=WIN_SEC_SSL,
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

    ds, loader = build_loader()
    steps_per_epoch = len(loader) if MAX_STEPS_PER_EPOCH is None else min(len(loader), MAX_STEPS_PER_EPOCH)
    T_LOCAL = int(round(WIN_SEC_SSL * OUT_HZ))

    print(f"[C1][BYOL] SSL on SuS | windows={len})

