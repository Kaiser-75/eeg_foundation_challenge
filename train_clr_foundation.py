from __future__ import annotations
import os, math, time
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from braindecode.models import EEGNetv4
from dataloader_c2 import (
    PreprocessConfig,
    make_ssl_dataset,
    MAX_CH,
    OUT_T,
)

# ============================================================
# Paths / environment 
# ============================================================
BASE_DIR   = Path(os.environ.get("EEG_BASE_DIR", "competition_data")).resolve()
SAVE_DIR   = Path("checkpoints_foundation").resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================
SEED                 = 42
EPOCHS               = 20
BATCH_SIZE           = 256                 # adjust for GPU memory (256 for A100, 128 for 2080Ti)
NUM_WORKERS          = 8
PERSISTENT_WORKERS   = True
PREFETCH_FACTOR      = 4
PIN_MEMORY           = torch.cuda.is_available()

LR                   = 1e-3
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 1

EMB_DIM              = 128                 # EEGNetv4 n_outputs
PROJ_DIM             = 128                 # projector output (unit-norm)
TEMP                 = 0.2                 # NT-Xent temperature

LAMBDA_TEMPORAL      = 1.0
LAMBDA_SPATIAL       = 1.0
LAMBDA_SPECTRAL      = 0.0                 # set >0 to enable spectral (FFT) contrast

STRIDE_SEC           = 1.0                 # window stride in SSL dataset
MAX_STEPS_PER_EPOCH  = None                # NOne fpr full epoch
LOG_EVERY            = 100

AMP_CLIP_UV          = 800.0               # match dataloader_c2 

# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = SEED) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
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

# ============================================================
# Model: EEGNetv4 encoder + projector (with LayerNorm)
# ============================================================
class EEGV4Encoder(nn.Module):
    """
    EEGNetv4 producing an embedding vector (logits) of size EMB_DIM.
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGNetv4(
            n_chans=in_ch,
            n_outputs=emb,
            n_times=T,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, EMB_DIM]
        return self.backbone(x)

class Projector(nn.Module):
    """
    2-layer projector with LayerNorm → GELU → Linear, then L2 normalize.
    LayerNorm stabilizes across-subject amplitude/band variance.
    """
    def __init__(self, emb: int = EMB_DIM, proj: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb, emb, bias=True),
            nn.LayerNorm(emb, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(emb, proj, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=-1)

class SimCLR(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM, proj: int = PROJ_DIM):
        super().__init__()
        self.encoder   = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.projector = Projector(emb=emb, proj=proj)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)            # [B, EMB_DIM]
        z = self.projector(h)          # [B, PROJ_DIM] (unit norm)
        return h, z

# ============================================================
# Augmentations
# ============================================================
def rand_time_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
    # roll along time (samples), max_shift at 100 Hz -> up to 100 ms default
    if max_shift <= 0:
        return x
    B, C, T = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    out = torch.empty_like(x)
    for i, s in enumerate(shifts):
        out[i] = torch.roll(x[i], shifts=int(s.item()), dims=-1)
    return out

def rand_time_mask(x: torch.Tensor, max_frac: float = 0.15, num_masks: int = 2) -> torch.Tensor:
    if max_frac <= 0.0 or num_masks <= 0:
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
    if sigma <= 0.0:
        return x
    return x + sigma * torch.randn_like(x)

def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0.0:
        return x
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask

def channel_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    B, C, T = x.shape
    scale = (1.0 + sigma * torch.randn(B, C, 1, device=x.device))
    return x * scale

def spectral_bandstop(x: torch.Tensor, p_apply: float = 0.3, width_hz: float = 2.0, fs: float = 100.0) -> torch.Tensor:
    """
    Optional spectral augmentation: randomly zero a narrow symmetric frequency band.
    Implemented via real FFT / iFFT. SSL only
    """
    if p_apply <= 0.0:
        return x
    if torch.rand(()) > p_apply:
        return x
    B, C, T = x.shape
    # rfft over time
    X = torch.fft.rfft(x.float(), dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0/fs).to(x.device)  # [F]
    # choose center band away from DC/Nyquist
    fmin, fmax = 1.0, fs/2 - 1.0
    f0 = (fmin + (fmax - fmin) * torch.rand(())).item()
    band = (freqs >= (f0 - width_hz/2.0)) & (freqs <= (f0 + width_hz/2.0))
    X[..., band] = 0.0
    x_ = torch.fft.irfft(X, n=T, dim=-1)
    return x_.type_as(x)

def make_temporal_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    v2 = add_noise(rand_time_mask(rand_time_shift(x, max_shift=10), max_frac=0.15, num_masks=2), sigma=0.02)
    return v1, v2

def make_spatial_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    v2 = add_noise(channel_jitter(channel_dropout(x, p=0.1), sigma=0.02), sigma=0.01)
    return v1, v2

def make_spectral_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = spectral_bandstop(x, p_apply=1.0, width_hz=2.0, fs=100.0)
    v2 = spectral_bandstop(x, p_apply=1.0, width_hz=2.0, fs=100.0)
    return v1, v2

# ============================================================
# NT-Xent loss (fp32 sim with -inf diagonal)
# ============================================================
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMP) -> torch.Tensor:
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    z  = torch.cat([z1, z2], dim=0)             # [2B, D]
    sim = (z @ z.T) / float(temperature)        # [2B, 2B]
    sim.fill_diagonal_(-float("inf"))
    B = z1.shape[0]
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)  # positives across the block
    return F.cross_entropy(sim, targets)

# ============================================================
# Data
# ============================================================
def build_loader() -> DataLoader:
    cfg = PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True, avg_ref=True,
        resample_hz=100.0,
        amp_clip_uv=AMP_CLIP_UV, window_standardize=True,  # keep consistent with supervised/inference
    )
    ds = make_ssl_dataset(
        base_dir=BASE_DIR,
        preprocess=cfg,
        stride_sec=STRIDE_SEC,
        preload=False,
    )
    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(ds, **dl_kwargs)

# ============================================================
# Train loop
# ============================================================
def train():
    set_seed(SEED)
    device = get_device()

    print(f"[foundation] SSL on all tasks (CCD pretrial only) | device={device.type}")
    print(f"BASE_DIR = {BASE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    loader = build_loader()
    steps_per_epoch = len(loader) if MAX_STEPS_PER_EPOCH is None else min(len(loader), MAX_STEPS_PER_EPOCH)

    model = SimCLR(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM, proj=PROJ_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = cosine_lr(optimizer, base_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, warmup_epochs=WARMUP_EPOCHS)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    global_step = 0
    t0_all = time.time()
    model.train()
    for ep in range(1, EPOCHS + 1):
        running = {"temp": 0.0, "spat": 0.0, "spec": 0.0, "total": 0.0}
        t0 = time.time()
        for it, X in enumerate(loader, start=1):
            if MAX_STEPS_PER_EPOCH is not None and it > MAX_STEPS_PER_EPOCH:
                break
            X = X.to(device, non_blocking=True)  # [B, C, T]

            # Build positive pairs
            Xt1, Xt2 = make_temporal_views(X)
            Xs1, Xs2 = make_spatial_views(X)
            if LAMBDA_SPECTRAL > 0.0:
                Xf1, Xf2 = make_spectral_views(X)

            with autocast(enabled=(device.type == "cuda")):
                _, zt1 = model(Xt1)
                _, zt2 = model(Xt2)
                _, zs1 = model(Xs1)
                _, zs2 = model(Xs2)

                loss_t = nt_xent_loss(zt1, zt2, temperature=TEMP)
                loss_s = nt_xent_loss(zs1, zs2, temperature=TEMP)
                loss = LAMBDA_TEMPORAL * loss_t + LAMBDA_SPATIAL * loss_s

                if LAMBDA_SPECTRAL > 0.0:
                    _, zf1 = model(Xf1)
                    _, zf2 = model(Xf2)
                    loss_f = nt_xent_loss(zf1, zf2, temperature=TEMP)
                    loss = loss + LAMBDA_SPECTRAL * loss_f
                else:
                    loss_f = torch.tensor(0.0, device=device)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["temp"]  += float(loss_t.item())
            running["spat"]  += float(loss_s.item())
            running["spec"]  += float(loss_f.item())
            running["total"] += float(loss.item())

            global_step += 1
            if it % LOG_EVERY == 0 or it == 1:
                lr = scheduler.get_last_lr()[0]
                avg_t = running["temp"] / it
                avg_s = running["spat"] / it
                avg_f = running["spec"] / it
                avg_tot = running["total"] / it
                elapsed = time.time() - t0
                print(f"ep {ep:02d} | step {it:05d}/{steps_per_epoch:05d} | "
                      f"lr {lr:.2e} | temp {avg_t:.4f} | spat {avg_s:.4f} | spec {avg_f:.4f} | total {avg_tot:.4f} | "
                      f"{elapsed:.1f}s")

        # ---- save checkpoint each epoch (encoder-only + metadata)
        enc_state = {}
        for k, v in model.state_dict().items():
            if k.startswith("encoder.backbone."):
                enc_state[k] = v
        # If keys are "backbone.*", prefix them to "encoder.backbone.*"
        if not enc_state:
            for k, v in model.encoder.backbone.state_dict().items():
                enc_state["encoder.backbone." + k] = v

        ckpt = {
            "epoch": ep,
            "hparams": {
                "emb_dim": EMB_DIM,
                "proj_dim": PROJ_DIM,
                "temp": TEMP,
                "lambda_temporal": LAMBDA_TEMPORAL,
                "lambda_spatial": LAMBDA_SPATIAL,
                "lambda_spectral": LAMBDA_SPECTRAL,
                "amp_clip_uv": AMP_CLIP_UV,
                "stride_sec": STRIDE_SEC,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
            "encoder_only": enc_state,
        }
        torch.save(ckpt, SAVE_DIR / f"simclr_foundation_epoch{ep:03d}.pt")
        torch.save(ckpt, SAVE_DIR / "simclr_foundation_latest.pt")
        print(f"[ckpt] saved → {SAVE_DIR / f'simclr_foundation_epoch{ep:03d}.pt'} | epoch time {time.time()-t0:.1f}s")

    print(f"Done. Total time: {time.time()-t0_all:.1f}s | latest: {SAVE_DIR / 'simclr_foundation_latest.pt'}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    train()
