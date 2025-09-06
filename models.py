# models.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F



def kaiming_init_(m: nn.Module) -> None:
    # Use ReLU gain for conv/linear so it works even if 'gelu' is unsupported
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        if getattr(m, "weight", None) is not None and m.weight is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if getattr(m, "weight", None) is not None and m.weight is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DWSeparableBlock(nn.Module):
    """
    Depthwise separable 1D conv block with residual.
    x -> DWConv(k, s) -> BN -> GELU -> PWConv(1x1) -> BN -> GELU -> +res
    """
    def __init__(self, ch: int, out_ch: int, k: int = 7, s: int = 1):
        super().__init__()
        self.dw = ConvBNAct(ch, ch, k=k, s=s, g=ch)        # depthwise
        self.pw = ConvBNAct(ch, out_ch, k=1, s=1, p=0)     # pointwise
        self.res = None
        if s != 1 or ch != out_ch:
            self.res = nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        r = x if self.res is None else self.res(x)
        return self.act(y + r)

# ---------------------------
# Encoder
# ---------------------------

@dataclass
class EEGEncoderCfg:
    in_ch: int = 129
    T: int = 200
    stem_ch: int = 128
    width_mult: float = 1.0
    dropout: float = 0.1
    mlp_hidden: int = 256
    emb: int = 128

class EEGEncoder(nn.Module):
    """
    Compact EEG encoder for 2s windows at 100 Hz (C=129, T=200).
    Pure Conv1d, depthwise separable blocks, global pool → MLP.
    """
    def __init__(self, in_ch: int = 129, T: int = 200, hid: int = 256, emb: int = 128):
        super().__init__()
        width = 1.0
        c1 = int(128 * width)
        c2 = int(192 * width)
        c3 = int(256 * width)
        c4 = int(256 * width)

        self.stem = ConvBNAct(in_ch, c1, k=7, s=1)
        self.block1 = DWSeparableBlock(c1, c1, k=7, s=1)   # T -> T
        self.block2 = DWSeparableBlock(c1, c2, k=5, s=2)   # T -> T/2
        self.block3 = DWSeparableBlock(c2, c3, k=5, s=2)   # T/2 -> T/4
        self.block4 = DWSeparableBlock(c3, c4, k=5, s=1)   # T/4 -> T/4

        self.dropout = nn.Dropout(p=0.1)

        self.head = nn.Sequential(
            nn.Linear(c4, hid, bias=False),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, emb, bias=True),
            nn.LayerNorm(emb),
        )

        self.apply(kaiming_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 129, 200] -> z: [B, emb]
        """
        assert x.dim() == 3, f"EEGEncoder expects [B, C, T], got {tuple(x.shape)}"
        y = self.stem(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.dropout(y)
        y = y.mean(dim=-1)      # global avg pool over time
        z = self.head(y)        # [B, emb]
        return z

# ---------------------------
# Utilities
# ---------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    enc = EEGEncoder(in_ch=129, T=200, hid=256, emb=128)
    x = torch.randn(8, 129, 200)
    z = enc(x)
    print("Output shape:", z.shape)  # [8, 128]
    print("Params (M):", count_parameters(enc) / 1e6)
