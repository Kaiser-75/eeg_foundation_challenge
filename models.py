from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# braindecode backbones
from braindecode.models import EEGNetv4, AttentionBaseNet

# Challenge I/O sizes
MAX_CH: int = 129
OUT_T:  int = 200   # 2.0 s @ 100 Hz

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _kaiming_(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

# ------------------------------------------------------------
# Backbones (encoders): produce an embedding vector of size `emb`
# ------------------------------------------------------------
class EEGV4Encoder(nn.Module):
    """
    EEGNetv4 as an encoder: we set n_classes=emb so its final linear
    produces the representation directly.
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = 128):
        super().__init__()
        self.emb = int(emb)
        self.backbone = EEGNetv4(
            in_chans=in_ch,
            n_classes=self.emb,               # treat as feature dim
            input_window_samples=T,
        )

    @property
    def out_dim(self) -> int:
        return self.emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]; EEGNetv4 internally handles shape via Ensure4d
        return self.backbone(x)  # [B, emb]


class AttentionBaseEncoder(nn.Module):
    """
    AttentionBaseNet encoder: set n_outputs=emb so the classifier is the feature map.
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = 128, sfreq: float = 100.0):
        super().__init__()
        self.emb = int(emb)
        self.backbone = AttentionBaseNet(
            n_times=T,
            n_chans=in_ch,
            n_outputs=self.emb,              # treat as feature dim
            sfreq=sfreq,
            # keep the rest at defaults; you can tune if needed
        )

    @property
    def out_dim(self) -> int:
        return self.emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, emb]


class SimpleCNNEncoder(nn.Module):
    """
    Lightweight 1D CNN over time with channel mixing (works directly on [B, C, T]).
    """
    def __init__(self, in_ch: int = MAX_CH, T: int = OUT_T, emb: int = 128, drop: float = 0.1):
        super().__init__()
        self.emb = int(emb)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),                      # 200 -> 100
            nn.Dropout(drop),

            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(2),                      # 100 -> 50
            nn.Dropout(drop),

            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),              # -> [B, 256, 1]
        )
        self.proj = nn.Linear(256, self.emb)
        self.apply(_kaiming_)

    @property
    def out_dim(self) -> int:
        return self.emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.net(x).squeeze(-1)  # [B, 256]
        z = self.proj(h)             # [B, emb]
        return z


# ------------------------------------------------------------
# Heads for supervised transfer (RT regression + HIT classification)
# ------------------------------------------------------------
class RTHead(nn.Module):
    """Small MLP → single scalar (regression)."""
    def __init__(self, in_dim: int, hid: int = 128, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, 1),
        )
        self.apply(_kaiming_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # [B]


class HitHead(nn.Module):
    """Binary logit (for BCEWithLogitsLoss)."""
    def __init__(self, in_dim: int, hid: int = 128, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, 1),
        )
        self.apply(_kaiming_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # [B] logit


# ------------------------------------------------------------
# Unified backbone
# ------------------------------------------------------------
BackboneName = Literal["eegnetv4", "attention", "simplecnn"]

@dataclass
class BackboneConfig:
    name: BackboneName = "eegnetv4"
    in_ch: int = MAX_CH
    T: int = OUT_T
    emb: int = 128
    sfreq: float = 100.0  # only used by AttentionBase

def make_backbone(cfg: BackboneConfig) -> nn.Module:
    if cfg.name == "eegnetv4":
        return EEGV4Encoder(in_ch=cfg.in_ch, T=cfg.T, emb=cfg.emb)
    if cfg.name == "attention":
        return AttentionBaseEncoder(in_ch=cfg.in_ch, T=cfg.T, emb=cfg.emb, sfreq=cfg.sfreq)
    if cfg.name == "simplecnn":
        return SimpleCNNEncoder(in_ch=cfg.in_ch, T=cfg.T, emb=cfg.emb)
    raise ValueError(f"Unknown backbone: {cfg.name}")

class EEGTransferModel(nn.Module):
    """
    Backbone (feature extractor) + two heads:
      - RT regression (scalar)
      - HIT classification (binary logit)
    """
    def __init__(self, backbone: nn.Module, head_hid: int = 128, drop: float = 0.1):
        super().__init__()
        self.backbone = backbone
        emb = getattr(backbone, "out_dim", None)
        if emb is None:
            raise ValueError("Backbone must define `out_dim` property.")
        self.rt_head = RTHead(emb, hid=head_hid, drop=drop)
        self.hit_head = HitHead(emb, hid=head_hid, drop=drop)

    @property
    def feature_dim(self) -> int:
        return self.backbone.out_dim

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)      # [B, emb]
        rt   = self.rt_head(feat)    # [B]
        hit  = self.hit_head(feat)   # [B] logit
        return {"feat": feat, "rt": rt, "hit": hit}
