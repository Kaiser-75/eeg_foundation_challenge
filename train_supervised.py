from __future__ import annotations
import os, math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score, mean_absolute_error

from braindecode.models import EEGNetv4  
from dataloader import CCDWindowDataset, PreprocessConfig, MAX_CH, OUT_T

# =========================
# Hyperparameters (edit here)
# =========================
BASE_DIR            = Path(os.environ.get("EEG_BASE_DIR", "competition_data"))
PRETRAINED_CKPT     = Path("checkpoints/simclr_sus_latest.pt")
SAVE_DIR            = Path("checkpoints_ccd")

TRAIN_RELEASES      = ["R1","R2","R3","R4","R6","R7","R8","R9","R10","R11"]
VAL_RELEASES        = ["R5"]
TEST_RELEASES       = ["R12"]   # optional; ignored if not present

CCD_MODE            = "poststim"    # "pretrial" | "poststim"
PRELOAD             = False

BATCH_SIZE          = 128
NUM_WORKERS         = 8
PIN_MEMORY          = torch.cuda.is_available()
PERSISTENT_WORKERS  = True
PREFETCH_FACTOR     = 4

# model / heads
EMB_DIM             = 128           # must match SimCLR n_outputs
HID_FC              = 256
DROPOUT             = 0.10

# losses
ALPHA_RT            = 1.0           # MAE weight
BETA_HIT            = 1.0           # CE weight

# training
SEED                = 42
EPOCHS_LINEAR       = 3             # freeze encoder
EPOCHS_FT           = 7             # unfreeze encoder
LR_LINEAR           = 1e-3          # heads only
LR_FT_BACKBONE      = 5e-5          # smaller LR for encoder
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
# Collate
# =========================
def collate_supervised(batch):
    xs, rts, hits = [], [], []
    for x, y in batch:
        xs.append(x)
        rts.append(y["rt"])
        hits.append(y["hit"])
    X = torch.stack(xs, dim=0)                 # [B, C, T]
    rt = torch.stack(rts, dim=0).float()       # [B]
    hit = torch.stack(hits, dim=0).long()      # [B]
    return X, {"rt": rt, "hit": hit}

# =========================
# Data / splits
# =========================
def make_cfg() -> PreprocessConfig:
    return PreprocessConfig(
        l_freq=0.5, h_freq=40.0, line_freq=60, notch=True,
        avg_ref=True, resample_hz=100.0,
        amp_clip_uv=200.0, window_standardize=True,
    )

@dataclass
class SubjectSplit:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]

class IndexSubset(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        return self.base[self.indices[i]]

def build_subject_split(ds: CCDWindowDataset, train_subjects: set[str], val_subjects: set[str], test_subjects: set[str]) -> SubjectSplit:
    train_idx, val_idx, test_idx = [], [], []
    for k in range(len(ds)):
        sj = ds.get_subject(k)
        if sj in train_subjects:
            train_idx.append(k)
        elif sj in val_subjects:
            val_idx.append(k)
        elif sj in test_subjects:
            test_idx.append(k)
    return SubjectSplit(train_idx, val_idx, test_idx)

def build_loaders() -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    cfg = make_cfg()

    ds_train = CCDWindowDataset(base_dir=BASE_DIR, releases=TRAIN_RELEASES, mode=CCD_MODE,
                                preprocess=cfg, preload=PRELOAD, verbose="INFO")
    ds_val   = CCDWindowDataset(base_dir=BASE_DIR, releases=VAL_RELEASES, mode=CCD_MODE,
                                preprocess=cfg, preload=PRELOAD, verbose="INFO")

    try:
        ds_test = CCDWindowDataset(base_dir=BASE_DIR, releases=TEST_RELEASES, mode=CCD_MODE,
                                   preprocess=cfg, preload=PRELOAD, verbose="INFO")
        if len(ds_test) == 0:
            ds_test = None
    except Exception:
        ds_test = None

    train_subjects = {ds_train.get_subject(k) for k in range(len(ds_train))}
    val_subjects   = {ds_val.get_subject(k)   for k in range(len(ds_val))}
    test_subjects  = {ds_test.get_subject(k) for k in range(len(ds_test))} if ds_test is not None else set()

    ds_all = CCDWindowDataset(base_dir=BASE_DIR, releases=TRAIN_RELEASES + VAL_RELEASES + (TEST_RELEASES if ds_test else []),
                              mode=CCD_MODE, preprocess=cfg, preload=PRELOAD, verbose="ERROR")

    split = build_subject_split(ds_all, train_subjects, val_subjects, test_subjects)
    tr = IndexSubset(ds_all, split.train_idx)
    va = IndexSubset(ds_all, split.val_idx)
    te = IndexSubset(ds_all, split.test_idx) if ds_test is not None else None

    def _make_loader(d):
        return DataLoader(
            d, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            drop_last=True, collate_fn=collate_supervised
        )

    train_loader = _make_loader(tr)
    val_loader   = _make_loader(va)
    test_loader  = _make_loader(te) if te is not None and len(te) > 0 else None
    return train_loader, val_loader, test_loader

# =========================
# Encoder (EEGNetv4) + heads
# =========================
class EEGV4Encoder(nn.Module):
    """EEGNetv4 that outputs a feature vector of size EMB_DIM (we use logits as features)."""
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        # Newer braindecode prefers n_chans / n_outputs / n_times
        self.backbone = EEGNetv4(n_chans=in_ch, n_outputs=emb, n_times=T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, EMB_DIM]

class CCDHead(nn.Module):
    def __init__(self, emb: int = EMB_DIM, hid: int = HID_FC, dropout: float = DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb, hid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_rt  = nn.Linear(hid, 1)
        self.out_hit = nn.Linear(hid, 2)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.mlp(h)
        rt  = self.out_rt(x).squeeze(-1)
        logit = self.out_hit(x)
        return {"rt": rt, "hit_logit": logit}

class CCDModel(nn.Module):
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = EEGV4Encoder(in_ch=in_ch, T=T, emb=emb)
        self.heads = CCDHead(emb=emb, hid=HID_FC, dropout=DROPOUT)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return self.heads(h)

def load_simclr_encoder_weights(model: CCDModel, ckpt_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load EEGNetv4 weights from a SimCLR checkpoint saved with keys like:
      'encoder.backbone.<EEGNetv4_param_name>'
    into:
      model.backbone.backbone.<EEGNetv4_param_name>
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("model", sd)

    sub = {k.replace("encoder.backbone.", ""): v
           for k, v in state.items() if k.startswith("encoder.backbone.")}

    # Load into the *inner* EEGNetv4 module
    missing = model.backbone.backbone.load_state_dict(sub, strict=False)
    # normalize return to lists for printing
    if isinstance(missing, tuple) and len(missing) == 2:
        miss, unexp = list(missing[0]), list(missing[1])
    else:
        miss, unexp = [], []
    return miss, unexp

# =========================
# Train / Eval
# =========================
@dataclass
class BatchOut:
    loss: torch.Tensor
    loss_rt: torch.Tensor
    loss_hit: torch.Tensor
    y_rt: torch.Tensor
    y_hit: torch.Tensor
    p_rt: torch.Tensor
    p_hit: torch.Tensor

def estimate_class_weights(loader: DataLoader, max_batches: int = 10) -> torch.Tensor:
    pos = 0
    total = 0
    with torch.no_grad():
        for i, (_, y) in enumerate(loader, start=1):
            pos += int(y["hit"].sum().item())
            total += int(y["hit"].numel())
            if i >= max_batches:
                break
    if total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    neg = total - pos
    w_pos = neg / total
    w_neg = pos / total
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)

def step_supervised(model: CCDModel, batch, class_weights: torch.Tensor, device: torch.device) -> BatchOut:
    X, y = batch
    X = X.to(device, non_blocking=True)
    y_rt  = y["rt"].to(device)
    y_hit = y["hit"].to(device)

    out = model(X)
    p_rt = out["rt"]
    logit = out["hit_logit"]

    loss_rt = F.l1_loss(p_rt, y_rt)
    cw = class_weights.to(device)
    loss_hit = F.cross_entropy(logit, y_hit, weight=cw)

    loss = ALPHA_RT * loss_rt + BETA_HIT * loss_hit
    p_hit = torch.softmax(logit, dim=-1)[:, 1]
    return BatchOut(loss, loss_rt, loss_hit, y_rt.detach(), y_hit.detach(), p_rt.detach(), p_hit.detach())

@torch.no_grad()
def evaluate(model: CCDModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    rts, rts_pred, hits, hits_prob = [], [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        out = model(X)
        rts_pred.append(out["rt"].cpu())
        hits_prob.append(torch.softmax(out["hit_logit"], dim=-1)[:, 1].cpu())
        rts.append(y["rt"])
        hits.append(y["hit"])
    y_rt  = torch.cat(rts).numpy()
    p_rt  = torch.cat(rts_pred).numpy()
    y_hit = torch.cat(hits).numpy()
    p_hit = torch.cat(hits_prob).numpy()

    mae = float(mean_absolute_error(y_rt, p_rt))
    r2  = float(r2_score(y_rt, p_rt)) if len(np.unique(y_rt)) > 1 else 0.0
    try:
        auc = float(roc_auc_score(y_hit, p_hit))
    except Exception:
        auc = 0.5
    y_pred = (p_hit >= 0.5).astype(np.int64)
    bacc = float(balanced_accuracy_score(y_hit, y_pred))
    return {"mae": mae, "r2": r2, "auc": auc, "bacc": bacc}

def train():
    set_seed(SEED)
    device = get_device()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders()
    print(f"[data] train steps/epoch: {len(train_loader)} | val steps: {len(val_loader)}" +
          (f" | test steps: {len(test_loader)}" if test_loader is not None else ""))

    class_weights = estimate_class_weights(train_loader, max_batches=10)

    # Model
    model = CCDModel(in_ch=MAX_CH, T=OUT_T, emb=EMB_DIM).to(device)
    miss, unexp = load_simclr_encoder_weights(model, PRETRAINED_CKPT)
    print(f"[ckpt] loaded SimCLR(EEGNetv4) → backbone | missing={len(miss)} unexpected={len(unexp)}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- Stage 1: Linear probe
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = AdamW(model.heads.parameters(), lr=LR_LINEAR, weight_decay=WEIGHT_DECAY)
    sch = cosine_lr(opt, base_lr=LR_LINEAR, epochs=EPOCHS_LINEAR, steps_per_epoch=len(train_loader), warmup_epochs=WARMUP_EPOCHS)

    print(f"[stage1] linear probe for {EPOCHS_LINEAR} epochs")
    best_mae = float("inf")
    for ep in range(1, EPOCHS_LINEAR + 1):
        model.train()
        run = {"loss":0.0, "rt":0.0, "hit":0.0}
        for it, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                bout = step_supervised(model, batch, class_weights=class_weights, device=device)
            scaler.scale(bout.loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()

            run["loss"] += bout.loss.item()
            run["rt"]   += bout.loss_rt.item()
            run["hit"]  += bout.loss_hit.item()
            if it % LOG_EVERY == 0 or it == 1:
                lr = sch.get_last_lr()[0]
                print(f"[stage1] ep {ep:02d} | it {it:05d}/{len(train_loader):05d} | "
                      f"lr {lr:.2e} | loss {run['loss']/it:.4f} | rt {run['rt']/it:.4f} | hit {run['hit']/it:.4f}")

        val = evaluate(model, val_loader, device)
        print(f"[stage1][val] ep {ep:02d} | MAE {val['mae']:.4f} | R2 {val['r2']:.3f} | AUC {val['auc']:.3f} | BAcc {val['bacc']:.3f}")
        if val["mae"] < best_mae:
            best_mae = val["mae"]
            torch.save({"model": model.state_dict(), "epoch": ep, "stage": 1, "val": val}, SAVE_DIR / "ccd_best_linear.pt")

    # ---- Stage 2: Finetune
    for p in model.backbone.parameters():
        p.requires_grad = True

    opt = AdamW([
        {"params": model.backbone.parameters(), "lr": LR_FT_BACKBONE},
        {"params": model.heads.parameters(),    "lr": LR_FT_HEADS},
    ], weight_decay=WEIGHT_DECAY)

    sch = cosine_lr(opt, base_lr=LR_FT_HEADS, epochs=EPOCHS_FT, steps_per_epoch=len(train_loader), warmup_epochs=WARMUP_EPOCHS)

    print(f"[stage2] finetune for {EPOCHS_FT} epochs")
    best_score = float("inf")
    for ep in range(1, EPOCHS_FT + 1):
        model.train()
        run = {"loss":0.0, "rt":0.0, "hit":0.0}
        for it, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                bout = step_supervised(model, batch, class_weights=class_weights, device=device)
            scaler.scale(bout.loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()

            run["loss"] += bout.loss.item()
            run["rt"]   += bout.loss_rt.item()
            run["hit"]  += bout.loss_hit.item()
            if it % LOG_EVERY == 0 or it == 1:
                lr = sch.get_last_lr()[0]
                print(f"[stage2] ep {ep:02d} | it {it:05d}/{len(train_loader):05d} | "
                      f"lr {lr:.2e} | loss {run['loss']/it:.4f} | rt {run['rt']/it:.4f} | hit {run['hit']/it:.4f}")

        val = evaluate(model, val_loader, device)
        print(f"[stage2][val] ep {ep:02d} | MAE {val['mae']:.4f} | R2 {val['r2']:.3f} | AUC {val['auc']:.3f} | BAcc {val['bacc']:.3f}")

        # simple composite for checkpointing
        score = 0.4*val["mae"] + 0.2*(1.0 - max(0.0, min(1.0, val["r2"]))) + 0.3*(1.0 - val["auc"]) + 0.1*(1.0 - val["bacc"])
        if score < best_score:
            best_score = score
            state = {"model": model.state_dict(), "epoch": ep, "stage": 2, "val": val}
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(state, SAVE_DIR / "ccd_best_finetune.pt")
            torch.save(state, SAVE_DIR / "ccd_latest.pt")
            print(f"[ckpt] saved best at ep {ep:02d} → {SAVE_DIR/'ccd_best_finetune.pt'}")

    if test_loader is not None:
        best_state = torch.load(SAVE_DIR / "ccd_best_finetune.pt", map_location=device)
        model.load_state_dict(best_state["model"])
        test = evaluate(model, test_loader, device)
        print(f"[test] MAE {test['mae']:.4f} | R2 {test['r2']:.3f} | AUC {test['auc']:.3f} | BAcc {test['bacc']:.3f}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    train()
