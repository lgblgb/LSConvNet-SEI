from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if k % 2 == 1 else k + 1

def _nearest_divisor(n: int, target: int) -> int:
    target = max(1, min(n, int(target)))
    divs = []
    r = int(math.sqrt(n))
    for d in range(1, r + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort()
    best = min(divs, key=lambda d: (abs(d - target), -d))
    return int(best)

def _adapt_share_groups(dim: int, groups: int) -> int:
    g = int(groups)
    if g <= 0:
        g = 1
    if g > dim:
        g = dim
    if dim % g == 0:
        return g
    return _nearest_divisor(dim, g)

class ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int=7, s: int=1, p: Optional[int]=None, dropout: float=0.0):
        super().__init__()
        k = _make_odd(k)
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.do = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x

class LKP1D(nn.Module):

    def __init__(self, dim: int, lks: int, sks: int, groups: int):
        super().__init__()
        self.dim = int(dim)
        self.lks = _make_odd(lks)
        self.sks = _make_odd(sks)
        self.groups = _adapt_share_groups(self.dim, groups)
        self.wc = self.dim // self.groups
        hidden = max(8, self.dim // 2)
        self.cv1 = nn.Sequential(nn.Conv1d(self.dim, hidden, kernel_size=1, bias=False), nn.BatchNorm1d(hidden))
        self.act = nn.ReLU(inplace=True)
        self.cv2 = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=self.lks, padding=self.lks // 2, groups=hidden, bias=False), nn.BatchNorm1d(hidden))
        self.cv3 = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=1, bias=False), nn.BatchNorm1d(hidden))
        self.cv4 = nn.Conv1d(hidden, self.sks * self.wc, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=self.wc, num_channels=self.sks * self.wc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.cv1(x))
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))
        w = self.cv4(x)
        w = self.norm(w)
        (B, _, L) = w.shape
        w = w.view(B, self.wc, self.sks, L)
        return w

class SKA1D(nn.Module):

    def __init__(self, dim: int, sks: int, wc: int):
        super().__init__()
        self.dim = int(dim)
        self.sks = _make_odd(sks)
        self.pad = self.sks // 2
        self.wc = int(wc)
        pat = torch.arange(self.dim) % self.wc
        self.register_buffer('pattern_idx', pat, persistent=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        (B, C, L) = x.shape
        assert C == self.dim, 'Channel mismatch.'
        assert w.shape[1] == self.wc and w.shape[2] == self.sks and (w.shape[3] == L), 'Weight shape mismatch.'
        x2d = x.unsqueeze(2)
        patches = F.unfold(x2d, kernel_size=(1, self.sks), padding=(0, self.pad), stride=(1, 1))
        patches = patches.view(B, C, self.sks, L)
        w_ch = w[:, self.pattern_idx]
        y = (patches * w_ch).sum(dim=2)
        return y

class LSConv1D(nn.Module):

    def __init__(self, dim: int, lks: int, sks: int, groups: int):
        super().__init__()
        self.dim = int(dim)
        self.lks = _make_odd(lks)
        self.sks = _make_odd(sks)
        self.groups = _adapt_share_groups(self.dim, groups)
        self.wc = self.dim // self.groups
        self.lkp = LKP1D(self.dim, lks=self.lks, sks=self.sks, groups=self.groups)
        self.ska = SKA1D(self.dim, sks=self.sks, wc=self.wc)
        self.bn = nn.BatchNorm1d(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lkp(x)
        y = self.ska(x, w)
        y = self.bn(y)
        return y + x

class LSConvStage(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k_down: int, s_down: int, dropout: float, ls_large_k: int, ls_small_k: int, ls_groups: int):
        super().__init__()
        self.down = nn.AvgPool1d(kernel_size=s_down, stride=s_down) if s_down and s_down > 1 else nn.Identity()
        self.pw = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.mixer = LSConv1D(out_ch, lks=ls_large_k, sks=ls_small_k, groups=ls_groups)
        self.do = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.pw(x)
        x = self.mixer(x)
        x = self.do(x)
        return x

class Branch_LSConv(nn.Module):

    def __init__(self, in_ch: int, ch: int, dropout: float, ls_large_k: int, ls_small_k: int, ls_groups: int):
        super().__init__()
        self.s1 = ConvBNAct(in_ch, ch, k=9, s=2, dropout=dropout)
        self.s2 = LSConvStage(ch, ch, k_down=7, s_down=2, dropout=dropout, ls_large_k=ls_large_k, ls_small_k=ls_small_k, ls_groups=ls_groups)
        self.s3 = LSConvStage(ch, ch * 2, k_down=5, s_down=2, dropout=dropout, ls_large_k=ls_large_k, ls_small_k=ls_small_k, ls_groups=ls_groups)
        self.s4 = LSConvStage(ch * 2, ch * 2, k_down=3, s_down=1, dropout=dropout, ls_large_k=ls_large_k, ls_small_k=ls_small_k, ls_groups=ls_groups)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.pool(x).squeeze(-1)
        return x

class MSCNN_LSConv(nn.Module):

    def __init__(self, num_classes: int, base_ch: int=64, dropout: float=0.2, ls_large_k: int=15, ls_small_k: int=3, ls_groups: int=8):
        super().__init__()
        self.branch0 = Branch_LSConv(2, base_ch, dropout, ls_large_k, ls_small_k, ls_groups)
        self.branch1 = Branch_LSConv(2, base_ch, dropout, ls_large_k, ls_small_k, ls_groups)
        self.branch2 = Branch_LSConv(2, base_ch, dropout, ls_large_k, ls_small_k, ls_groups)
        self.down2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.down4 = nn.AvgPool1d(kernel_size=4, stride=4)
        feat_dim = base_ch * 2 * 3
        self.fc = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.branch0(x)
        f1 = self.branch1(self.down2(x))
        f2 = self.branch2(self.down4(x))
        return torch.cat([f0, f1, f2], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.fc(feat)

def build_model(num_classes: int, base_ch: int=64, dropout: float=0.2, ls_large_k: int=15, ls_small_k: int=3, ls_groups: int=8) -> nn.Module:
    return MSCNN_LSConv(num_classes=num_classes, base_ch=base_ch, dropout=dropout, ls_large_k=ls_large_k, ls_small_k=ls_small_k, ls_groups=ls_groups)

# Public name; parameter names remain compatible with the original checkpoint.
LSConvNet = MSCNN_LSConv
