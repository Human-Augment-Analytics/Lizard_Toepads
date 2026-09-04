"""PIPNet (Pixel-in-Pixel Net) — framework-native, reference-faithful.

Reimplements PIPNet (Jin, Kong & Deng, IJCV 2021) aligned to the official
reference at ``../PIPNet/PIPNet`` (``lib/networks.py``, ``lib/functions.py``,
``lib/data_utils.py``). A torchvision ResNet produces a single low-resolution
feature map; five 1x1 conv heads predict per landmark and per grid cell:

  - ``cls``  : a score selecting the coarse cell,
  - ``x``,``y``      : the raw within-cell offset (cell units, target [0,1)),
  - ``nb_x``,``nb_y``: raw offsets to each of ``num_nb`` mean-shape neighbors
                       (the Neighbor Regression Module, a training-time
                       structural self-constraint).

Coordinates decode as "argmax cell + offset". The only deliberate deviation
from the reference is that neighbors are derived from the framework's mean-shape
tensor (``mean_shape_path``) instead of a ``meanface.txt`` file; the
neighbor-selection logic reproduces the reference ``get_meanface`` exactly.

This module imports only torch, torchvision, and numpy — never torch_geometric
or timm — so ``pipnet`` remains constructible in minimal environments.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvm

from .registry import register_model


def get_meanface_indices(
    meanface: torch.Tensor, num_nb: int
) -> Tuple[torch.Tensor, List[int], List[int], int]:
    """Derive per-landmark neighbor indices from a mean shape.

    Reproduces the reference ``get_meanface`` (PIPNet ``lib/functions.py``): for
    each landmark, rank all landmarks by squared Euclidean distance to it and
    take the nearest ``num_nb`` excluding itself. Also builds the reverse index
    used at inference to average each landmark's own prediction with the
    neighbor-predicted estimates of it.

    The only difference from the reference is the input: a tensor of shape
    (N, 2) rather than a parsed ``meanface.txt``. The selection logic is
    identical.

    Args:
        meanface: (N, 2) mean-shape coordinates (any consistent units; only
            relative distances matter).
        num_nb: Number of neighbors per landmark. Must be <= N - 1.

    Returns:
        Tuple of:
          - meanface_indices: (N, num_nb) long tensor of neighbor indices.
          - reverse_index1: flat list mapping (see reference).
          - reverse_index2: flat list mapping (see reference).
          - max_len: max reverse-neighbor count across landmarks.

    Raises:
        ValueError: If num_nb <= 0 or num_nb > N - 1.
    """
    mf = meanface.detach().cpu().numpy().reshape(-1, 2)
    n = mf.shape[0]
    if num_nb <= 0:
        raise ValueError(f"num_nb must be positive, got {num_nb}")
    if num_nb > n - 1:
        raise ValueError(
            f"num_nb={num_nb} exceeds N-1={n - 1}; a landmark cannot have more "
            f"than {n - 1} distinct neighbors for a mean shape of {n} points."
        )

    # Each landmark predicts its num_nb nearest neighbors (excluding self).
    meanface_indices: List[np.ndarray] = []
    for i in range(n):
        dists = np.sum((mf[i] - mf) ** 2, axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1 : 1 + num_nb])

    # Reverse mapping: for each landmark, which (landmark, slot) pairs predict it.
    reverse: dict = {i: [[], []] for i in range(n)}
    for i in range(n):
        for j in range(num_nb):
            reverse[meanface_indices[i][j]][0].append(i)
            reverse[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(n):
        max_len = max(max_len, len(reverse[i][0]))

    # Reference "trick": pad each list by repetition to a common length for
    # efficient gather at inference.
    for i in range(n):
        cur0 = reverse[i][0]
        cur1 = reverse[i][1]
        reverse[i][0] = (cur0 + cur0 * 10)[:max_len]
        reverse[i][1] = (cur1 + cur1 * 10)[:max_len]

    reverse_index1: List[int] = []
    reverse_index2: List[int] = []
    for i in range(n):
        reverse_index1 += reverse[i][0]
        reverse_index2 += reverse[i][1]

    idx = torch.tensor(np.array(meanface_indices), dtype=torch.long)
    return idx, reverse_index1, reverse_index2, max_len


# Channels emitted by the final ResNet stage (layer4) per torchvision arch.
_RESNET_FEAT_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}

# torchvision default-weights enum names per arch (weights= API, tv >= 0.13).
_RESNET_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


def _build_resnet(backbone: str, pretrained: bool) -> nn.Module:
    """Construct a torchvision ResNet, tolerant of old/new weights API.

    torchvision >= 0.13 removed the ``pretrained=`` kwarg in favor of
    ``weights=``. This maps the reference's boolean ``pretrained`` onto whichever
    API the installed torchvision exposes.
    """
    if backbone not in _RESNET_FEAT_CHANNELS:
        raise ValueError(
            f"Unsupported pipnet backbone {backbone!r}. "
            f"Supported: {sorted(_RESNET_FEAT_CHANNELS)}"
        )
    ctor = getattr(tvm, backbone)
    try:
        # Modern weights= API.
        weights = None
        if pretrained:
            enum = getattr(tvm, _RESNET_WEIGHT_ENUMS[backbone])
            weights = enum.DEFAULT
        return ctor(weights=weights)
    except TypeError:
        # Legacy pretrained= API.
        return ctor(pretrained=pretrained)


@register_model("pipnet")
class PIPNet(nn.Module):
    """Pixel-in-Pixel Net — reference-faithful.

    Args:
        num_landmarks: Number of landmarks (N). Must be >= 2 for NRM.
        backbone: torchvision ResNet name (resnet18/34/50/101/152).
        pretrained: Load ImageNet backbone weights.
        input_size: Square input side length.
        net_stride: Grid stride; grid side = input_size / net_stride. One of
            {16, 32, 64, 128}. 32 appends no extra layers; 64/128 append strided
            convs; 16 appends a transposed conv, matching the reference.
        num_nb: Neighbors per landmark for the NRM.
        meanface_indices: (N, num_nb) long neighbor indices. Registered as a
            buffer. When None, a self-referential placeholder is used so the
            model is still constructible (the loss should be given real indices).
    """

    _SUPPORTED_STRIDES = (16, 32, 64, 128)

    def __init__(
        self,
        num_landmarks: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        input_size: int = 512,
        net_stride: int = 32,
        num_nb: int = 10,
        meanface_indices: Optional[torch.Tensor] = None,
        use_star: bool = False,
        **kwargs,
    ):
        super().__init__()
        if num_landmarks < 2:
            raise ValueError(
                f"pipnet requires num_landmarks >= 2 (NRM needs neighbors), "
                f"got {num_landmarks}"
            )
        if net_stride not in self._SUPPORTED_STRIDES:
            raise ValueError(
                f"Unsupported net_stride {net_stride}; expected one of "
                f"{self._SUPPORTED_STRIDES}"
            )
        if num_nb <= 0:
            raise ValueError(f"num_nb must be positive, got {num_nb}")

        self.num_landmarks = num_landmarks
        self.num_nb = num_nb
        self.input_size = input_size
        self.net_stride = net_stride
        self.backbone_name = backbone
        self.use_star = use_star

        # --- Backbone: unpack torchvision ResNet exactly as the reference ---
        resnet = _build_resnet(backbone, pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        feat_ch = _RESNET_FEAT_CHANNELS[backbone]

        # --- Extra stride/deconv layers per net_stride (reference layout) ---
        self._extra = None  # tag describing which extra path forward() runs
        head_ch = feat_ch
        if net_stride == 32:
            self._extra = None
        elif net_stride == 64:
            self.layer5 = nn.Conv2d(feat_ch, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self._init_conv_bn(self.layer5, self.bn5)
            self._extra = "s64"
            head_ch = 512
        elif net_stride == 128:
            self.layer5 = nn.Conv2d(feat_ch, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self._init_conv_bn(self.layer5, self.bn5)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            self._init_conv_bn(self.layer6, self.bn6)
            self._extra = "s128"
            head_ch = 512
        elif net_stride == 16:
            self.deconv1 = nn.ConvTranspose2d(
                feat_ch, 512, kernel_size=4, stride=2, padding=1, bias=False
            )
            self.bn_deconv1 = nn.BatchNorm2d(512)
            self._init_conv_bn(self.deconv1, self.bn_deconv1)
            self._extra = "s16"
            head_ch = 512

        # --- Five 1x1 conv heads (reference channel counts) ---
        self.cls_layer = nn.Conv2d(head_ch, num_landmarks, kernel_size=1)
        self.x_layer = nn.Conv2d(head_ch, num_landmarks, kernel_size=1)
        self.y_layer = nn.Conv2d(head_ch, num_landmarks, kernel_size=1)
        self.nb_x_layer = nn.Conv2d(head_ch, num_nb * num_landmarks, kernel_size=1)
        self.nb_y_layer = nn.Conv2d(head_ch, num_nb * num_landmarks, kernel_size=1)
        for head in (
            self.cls_layer,
            self.x_layer,
            self.y_layer,
            self.nb_x_layer,
            self.nb_y_layer,
        ):
            nn.init.normal_(head.weight, std=0.001)
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)

        # --- STAR uncertainty head (Option A): per-landmark, per-cell Cholesky
        # params [log_L11, L21, log_L22]. Read at the GT/argmax cell just like
        # the offset heads. Zero-init so the model starts isotropic (Sigma = I),
        # i.e. STAR begins as plain L2 and must EARN any anisotropy. Only created
        # when use_star, so the paper-faithful model is byte-identical otherwise.
        self.sigma_layer = None
        if use_star:
            self.sigma_layer = nn.Conv2d(head_ch, 3 * num_landmarks, kernel_size=1)
            nn.init.constant_(self.sigma_layer.weight, 0.0)
            nn.init.constant_(self.sigma_layer.bias, 0.0)

        # --- Neighbor index buffer ---
        if meanface_indices is None:
            # Placeholder: each landmark neighbors the "next" ones cyclically.
            # Real indices should be supplied for training; this keeps the model
            # constructible for shape/registration tests.
            base = torch.arange(num_landmarks).unsqueeze(1)
            offs = torch.arange(1, num_nb + 1).unsqueeze(0)
            placeholder = (base + offs) % num_landmarks
            meanface_indices = placeholder.to(torch.long)
        else:
            meanface_indices = torch.as_tensor(meanface_indices, dtype=torch.long)
            if tuple(meanface_indices.shape) != (num_landmarks, num_nb):
                raise ValueError(
                    f"meanface_indices shape {tuple(meanface_indices.shape)} != "
                    f"(num_landmarks, num_nb) = ({num_landmarks}, {num_nb})"
                )
        self.register_buffer("meanface_indices", meanface_indices)

        # Reverse-neighbor index for the optional merged-decode inference. Set
        # via set_reverse_index(); None means merged decode is unavailable and
        # predict_coords(merge=True) falls back to the direct decode.
        self._reverse_index1 = None
        self._reverse_index2 = None
        self._reverse_maxlen = None

        # --- Verify the grid matches input_size / net_stride ---
        expected = input_size // net_stride
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            feat = self._backbone_forward(dummy)
            gh, gw = feat.shape[2], feat.shape[3]
        if gh != expected or gw != expected:
            raise AssertionError(
                f"pipnet grid mismatch: backbone produced {gh}x{gw} but "
                f"input_size/net_stride = {expected}. Check backbone/net_stride."
            )
        self.grid_h = gh
        self.grid_w = gw

    @staticmethod
    def _init_conv_bn(conv: nn.Module, bn: nn.Module) -> None:
        nn.init.normal_(conv.weight, std=0.001)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone + any extra stride/deconv layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self._extra == "s64":
            x = F.relu(self.bn5(self.layer5(x)))
        elif self._extra == "s128":
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self._extra == "s16":
            x = F.relu(self.bn_deconv1(self.deconv1(x)))
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the five raw prediction maps (reference forward signature).

        Args:
            x: (B, 3, input_size, input_size) images.

        Returns:
            (cls, off_x, off_y, nb_x, nb_y), all raw (un-activated):
              - cls/off_x/off_y: (B, N, Hg, Wg)
              - nb_x/nb_y:       (B, num_nb*N, Hg, Wg)
        """
        feat = self._backbone_forward(x)
        cls = self.cls_layer(feat)
        off_x = self.x_layer(feat)
        off_y = self.y_layer(feat)
        nb_x = self.nb_x_layer(feat)
        nb_y = self.nb_y_layer(feat)
        return cls, off_x, off_y, nb_x, nb_y

    def forward_star(self, x: torch.Tensor):
        """Like forward, but also returns the STAR sigma map.

        Returns (cls, off_x, off_y, nb_x, nb_y, sigma) where sigma is
        (B, 3*N, Hg, Wg) holding per-landmark, per-cell Cholesky params. Raises
        if the model was not built with use_star=True.
        """
        if self.sigma_layer is None:
            raise RuntimeError(
                "forward_star requires the model built with use_star=True."
            )
        feat = self._backbone_forward(x)
        cls = self.cls_layer(feat)
        off_x = self.x_layer(feat)
        off_y = self.y_layer(feat)
        nb_x = self.nb_x_layer(feat)
        nb_y = self.nb_y_layer(feat)
        sigma = self.sigma_layer(feat)
        return cls, off_x, off_y, nb_x, nb_y, sigma

    def predict_coords(self, x: torch.Tensor, merge: bool = False) -> torch.Tensor:
        """Convenience: forward + decode to (B, N, 2) coordinates in ~[0, 1].

        Args:
            x: (B, 3, input_size, input_size) images.
            merge: When True, average each landmark's direct estimate with the
                neighbor-predicted estimates of it (the reference NRM inference
                merge). Requires ``reverse_index1/2`` set via
                ``set_reverse_index`` (the training engine does this in setup).
        """
        cls, off_x, off_y, nb_x, nb_y = self.forward(x)
        if merge and self._reverse_index1 is not None:
            return decode_pip_merge(
                cls, off_x, off_y, nb_x, nb_y,
                self.input_size, self.net_stride, self.num_nb,
                self._reverse_index1, self._reverse_index2, self._reverse_maxlen,
            )
        return decode_pip(cls, off_x, off_y, self.input_size, self.net_stride)

    def set_reverse_index(self, reverse_index1, reverse_index2, max_len: int) -> None:
        """Store the reverse-neighbor index used by the merged-decode inference.

        These come from ``get_meanface_indices`` alongside ``meanface_indices``.
        Kept as plain Python lists (not buffers) since they are only used for the
        optional inference-time neighbor averaging.
        """
        self._reverse_index1 = reverse_index1
        self._reverse_index2 = reverse_index2
        self._reverse_maxlen = max_len


def decode_pip(
    cls: torch.Tensor,
    off_x: torch.Tensor,
    off_y: torch.Tensor,
    input_size: int,
    net_stride: int,
) -> torch.Tensor:
    """Decode coordinates from PIPNet maps, matching the reference ``forward_pip``.

    Picks the argmax cell per landmark (detached), reads that cell's raw x/y
    offset, and returns ``((col + off_x) / Wg, (row + off_y) / Hg)``.

    Args:
        cls: (B, N, Hg, Wg) score map.
        off_x: (B, N, Hg, Wg) within-cell x offset (raw).
        off_y: (B, N, Hg, Wg) within-cell y offset (raw).
        input_size: Square input side length.
        net_stride: Grid stride; input_size / net_stride == Hg == Wg.

    Returns:
        (B, N, 2) coordinates in normalized units (approximately [0, 1]).
    """
    b, n, gh, gw = cls.shape
    cls_flat = cls.view(b * n, -1)
    max_ids = torch.argmax(cls_flat, dim=1).detach().view(-1, 1)  # (B*N, 1)

    off_x_sel = torch.gather(off_x.view(b * n, -1), 1, max_ids).squeeze(1)
    off_y_sel = torch.gather(off_y.view(b * n, -1), 1, max_ids).squeeze(1)

    col = (max_ids.squeeze(1) % gw).float()
    row = (max_ids.squeeze(1) // gw).float()

    denom = 1.0 * input_size / net_stride  # == gw == gh
    x = (col + off_x_sel) / denom
    y = (row + off_y_sel) / denom
    return torch.stack([x, y], dim=1).view(b, n, 2)


def decode_pip_merge(
    cls: torch.Tensor,
    off_x: torch.Tensor,
    off_y: torch.Tensor,
    nb_x: torch.Tensor,
    nb_y: torch.Tensor,
    input_size: int,
    net_stride: int,
    num_nb: int,
    reverse_index1,
    reverse_index2,
    max_len: int,
) -> torch.Tensor:
    """Neighbor-averaged decode, reproducing the reference ``forward_pip`` + the
    ``test.py`` merge.

    In addition to each landmark's direct (col+off)/grid estimate, the neighbor
    heads predict, from every landmark's argmax cell, where that landmark's
    neighbors are. The reverse index gathers, for each landmark, all the
    neighbor-predicted estimates OF it; those are averaged together with the
    direct estimate. On the reference this measurably improves NME.

    Args:
        cls, off_x, off_y: (B, N, Hg, Wg) direct maps.
        nb_x, nb_y: (B, num_nb*N, Hg, Wg) neighbor-offset maps.
        input_size, net_stride, num_nb: grid / NRM parameters.
        reverse_index1, reverse_index2, max_len: from get_meanface_indices.

    Returns:
        (B, N, 2) merged coordinates in normalized units (~[0, 1]).
    """
    b, n, gh, gw = cls.shape
    denom = 1.0 * input_size / net_stride
    device = cls.device
    rev1 = torch.as_tensor(reverse_index1, dtype=torch.long, device=device)
    rev2 = torch.as_tensor(reverse_index2, dtype=torch.long, device=device)

    outs = []
    for bi in range(b):
        cls_flat = cls[bi].view(n, -1)
        max_ids = torch.argmax(cls_flat, dim=1).view(-1, 1)  # (N, 1)
        max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)  # (N*num_nb, 1)

        ox = torch.gather(off_x[bi].view(n, -1), 1, max_ids).squeeze(1)  # (N,)
        oy = torch.gather(off_y[bi].view(n, -1), 1, max_ids).squeeze(1)

        nbx = torch.gather(
            nb_x[bi].view(num_nb * n, -1), 1, max_ids_nb
        ).squeeze(1).view(n, num_nb)
        nby = torch.gather(
            nb_y[bi].view(num_nb * n, -1), 1, max_ids_nb
        ).squeeze(1).view(n, num_nb)

        col = (max_ids.squeeze(1) % gw).float()
        row = (max_ids.squeeze(1) // gw).float()

        # Direct estimate (normalized).
        x_dir = (col + ox) / denom  # (N,)
        y_dir = (row + oy) / denom
        # Neighbor estimates: each landmark i predicts neighbor j at
        # (cell_i + nb_offset). Normalize.
        nb_x_est = (col.view(n, 1) + nbx) / denom  # (N, num_nb)
        nb_y_est = (row.view(n, 1) + nby) / denom

        # Reverse gather: for each landmark, all neighbor-predicted estimates of
        # it, shape (N, max_len).
        tmp_nb_x = nb_x_est[rev1, rev2].view(n, max_len)
        tmp_nb_y = nb_y_est[rev1, rev2].view(n, max_len)

        x_merge = torch.mean(
            torch.cat([x_dir.view(n, 1), tmp_nb_x], dim=1), dim=1
        )
        y_merge = torch.mean(
            torch.cat([y_dir.view(n, 1), tmp_nb_y], dim=1), dim=1
        )
        outs.append(torch.stack([x_merge, y_merge], dim=1))  # (N, 2)

    return torch.stack(outs, dim=0)  # (B, N, 2)
