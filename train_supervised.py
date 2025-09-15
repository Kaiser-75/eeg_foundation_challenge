from __future__ import annotations
import os, math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error

from braindecode.models import EEGNetv4
from dataloader import CCDWindowDataset, PreprocessConfig, MAX_CH, OUT_T

# =========================
# Hyperparameters (edit here)
# =========================
BASE_DIR            = Path(os.environ.get("EEG_BASE_DIR", "competition_data"))
PRETRAINED_CKPT     = Path("checkpoints_c1/simclr_sus_latest.pt")
SAVE_DIR            = Path("checkpoints_c1_supervised")

TRAIN_RELEASES      = ["R1","R2","R3","R4","R6","R7","R8","R9","R10","R11"]
VAL_RELEASES        = ["R5"]
TEST_RELEASES       = ["R12"]

CCD_MODE            = "poststim"   

BATCH_SIZE          = 128
NUM_WORKERS         = 8
PIN_MEMORY          = torch.cuda.is_available()
PERSISTENT_WORKERS  = True
PREFETCH_FACTOR     = 4
PRELOAD             = False

# model / head
EMB_DIM             = 128
HID_FC              = 256
DROPOUT             = 0.10

# training
SEED                = 42
EPOCHS_LINEAR       = 3     # freeze encoder (linear probe)
EPOCHS_FT           = 7     # unfreeze encoder (finetune)
LR_LINEAR           = 1e-3
LR_FT_BACKBONE      = 5e-5
LR_FT_HEADS         = 1e-3
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 1
LOG_EVERY           = 100

# =========================
# Utils
# =========================
def set_seed(seed: int = SEED):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
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

# =========================
# Collate (RT only)
# =========================
def collate_supervised(batch):
    xs, rts = [], []
    for x, y in batch:
        xs.append(x)
        rts.append(y["rt"])
    X  = torch.stack(xs, dim=0)           # [B, C, T]
    rt = torch.stack(rts, dim=0).float()  # [B]
    return X, {"rt": rt}

# =========================
# Data / splits
# =========================
def make_cfg() -> PreprocessConfig:
    return PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True,
        avg_ref=True, resample_hz=100.0,
        amp_clip_uv=600.0, window_standardize=True,  
    )

def build_loaders() -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    cfg = make_cfg()
    ds_train = CCDWindowDataset(base_dir=BASE_DIR, releases=TRAIN_RELEASES, mode=CCD_MODE,
                                preprocess=cfg, preload=PRELOAD, verbose="INFO")
    ds_val   = CCDWindowDataset(base_dir=BASE_DIR, releases=VAL_RELEASES, mode=CCD_MODE,
                                preprocess=cfg, preload=PRELOAD, verbose="INFO")
    ds_test  = CCDWindowDataset(base_dir=BASE_DIR, releases=TEST_RELEASES, mode=CCD_MODE,
                                preprocess=cfg, preload=PRELOAD, verbose="INFO")

    def _make_loader(d, shuffle: bool):
        kwargs = dict(
            batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            drop_last=True, collate_fn=collate_supervised
        )
        if NUM_WORKERS > 0:
            kwargs["persistent_workers"] = PERSISTENT_WORKERS
            kwargs["prefetch_factor"] = PREFETCH_FACTOR
        return DataLoader(d, **kwargs)

    return _make_loader(ds_train, True), _make_loader(ds_val, False), _make_loader(ds_test, False)

# =========================
# Encoder (EEGNetv4) + RT head
# =========================
class EEGV4Encoder(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGNetv4(n_chans=in_ch, n_outputs=emb, n_times=T)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, EMB_DIM]

class CCDHeadRT(nn.Module):
    def __init__(self, emb: int = EMB_DIM, hid: int = HID_FC, dropout: float = DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(emb, hid), nn.GELU(), nn.Dropout(dropout),
        )
        self.out_rt = nn.Linear(hid, 1)   # regression (seconds)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.trunk(h)
        return self.out_rt(x).squeeze(-1)

class CCDModel(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.head = CCDHeadRT(emb=emb, hid=HID_FC, dropout=DROPOUT)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)  # rt

def load_simclr_encoder_weights(model: CCDModel, ckpt_path: Path):
    if not ckpt_path.exists():
        print(f"[ckpt] WARNING: {ckpt_path} not found → training from scratch.")
        return
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("model", sd)
    sub = {k.replace("encoder.backbone.", ""): v
           for k, v in state.items() if k.startswith("encoder.backbone.")}
    missing, unexpected = model.backbone.backbone.load_state_dict(sub, strict=False)
    print(f"[ckpt] loaded backbone | missing={len(missing)} unexpected={len(unexpected)}")

# =========================
# Train / Eval
# =========================
@dataclass
class BatchOut:
    loss: torch.Tensor
    y_rt: torch.Tensor
    p_rt: torch.Tensor

def step_supervised(model: CCDModel, batch, device: torch.device) -> BatchOut:
    X, y = batch
    X = X.to(device, non_blocking=True)
    y_rt = y["rt"].to(device)
    p_rt = model(X)
    loss = F.mse_loss(p_rt, y_rt)
    return BatchOut(loss, y_rt.detach(), p_rt.detach())

@torch.no_grad()
def evaluate(model: CCDModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    rts, rts_pred = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        out = model(X)
        rts_pred.append(out.cpu())
        rts.append(y["rt"])
    y_rt = torch.cat(rts).numpy()
    p_rt = torch.cat(rts_pred).numpy()

    mae = float(mean_absolute_error(y_rt, p_rt))
    rmse = float(np.sqrt(np.mean((y_rt - p_rt) ** 2)))
    nrmse = float(rmse / (np.std(y_rt) + 1e-12))
    r2  = float(r2_score(y_rt, p_rt)) if len(np.unique(y_rt)) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": r2}

def train():
    set_seed(SEED)
    device = get_device()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders()
    print(f"[C1] Supervised on CCD ({CCD_MODE}) | device={device.type}")

    model = CCDModel(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM).to(device)
    load_simclr_encoder_weights(model, PRETRAINED_CKPT)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------- Stage 1: Linear probe (freeze encoder) ----------
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = AdamW(model.head.parameters(), lr=LR_LINEAR, weight_decay=WEIGHT_DECAY)
    sch = cosine_lr(opt, LR_LINEAR, EPOCHS_LINEAR, len(train_loader), WARMUP_EPOCHS)

    best_val = float("inf")
    for ep in range(1, EPOCHS_LINEAR + 1):
        model.train()
        for it, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                bout = step_supervised(model, batch, device)
            scaler.scale(bout.loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
            if it % LOG_EVERY == 0 or it == 1:
                print(f"[stage1] ep {ep:02d} it {it:05d} | loss {bout.loss.item():.4f}")

        val = evaluate(model, val_loader, device)
        print(f"[stage1][val] ep {ep:02d} | MAE {val['mae']:.4f} | RMSE {val['rmse']:.4f} | "
              f"NRMSE {val['nrmse']:.4f} | R2 {val['r2']:.3f}")
        if val["nrmse"] < best_val:
            best_val = val["nrmse"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val}, SAVE_DIR / "ccd_best_linear.pt")

    # ---------- Stage 2: Finetune (unfreeze encoder) ----------
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = AdamW([
        {"params": model.backbone.parameters(), "lr": LR_FT_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_FT_HEADS},
    ], weight_decay=WEIGHT_DECAY)
    sch = cosine_lr(opt, LR_FT_HEADS, EPOCHS_FT, len(train_loader), WARMUP_EPOCHS)

    best_val = float("inf")
    for ep in range(1, EPOCHS_FT + 1):
        model.train()
        for it, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                bout = step_supervised(model, batch, device)
            scaler.scale(bout.loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
            if it % LOG_EVERY == 0 or it == 1:
                print(f"[stage2] ep {ep:02d} it {it:05d} | loss {bout.loss.item():.4f}")

        val = evaluate(model, val_loader, device)
        print(f"[stage2][val] ep {ep:02d} | MAE {val['mae']:.4f} | RMSE {val['rmse']:.4f} | "
              f"NRMSE {val['nrmse']:.4f} | R2 {val['r2']:.3f}")
        if val["nrmse"] < best_val:
            best_val = val["nrmse"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val}, SAVE_DIR / "ccd_best_finetune.pt")

    # ---------- Test ----------
    best_path = SAVE_DIR / "ccd_best_finetune.pt"
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        test = evaluate(model, test_loader, device)
        print(f"[test] MAE {test['mae']:.4f} | RMSE {test['rmse']:.4f} | NRMSE {test['nrmse']:.4f} | R2 {test['r2']:.3f}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    train()


