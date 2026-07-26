"""Stacked Hourglass Network for heatmap-based landmark detection."""

import torch
from torch import nn

from .registry import register_model

Pool = nn.MaxPool2d


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super().__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class HeatmapLoss(nn.Module):
    def forward(self, pred, gt):
        l = ((pred - gt) ** 2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l


@register_model("stacked_hourglass")
class StackedHourGlass(nn.Module):
    """Stacked Hourglass network for heatmap-based landmark detection.

    Args:
        num_landmarks: Number of output landmarks (required).
        nstack: Number of hourglass stacks.
        inp_dim: Internal feature dimension.
        bn: Use batch normalization in hourglass.
        increase: Channel increase per hourglass level.
    """

    def __init__(self, num_landmarks: int, nstack: int = 2, inp_dim: int = 256,
                 bn: bool = False, increase: int = 0, **kwargs):
        super().__init__()
        self.nstack = nstack
        self.num_landmarks = num_landmarks

        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim),
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(Hourglass(4, inp_dim, bn, increase))
            for _ in range(nstack)
        ])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True),
            )
            for _ in range(nstack)
        ])

        self.outs = nn.ModuleList([
            Conv(inp_dim, num_landmarks, 1, relu=False, bn=False)
            for _ in range(nstack)
        ])
        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for _ in range(nstack - 1)
        ])
        self.merge_preds = nn.ModuleList([
            Merge(num_landmarks, inp_dim) for _ in range(nstack - 1)
        ])
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for pred in combined_hm_preds:
            combined_loss.append(self.heatmapLoss(pred, heatmaps))
        return torch.stack(combined_loss, dim=1)
