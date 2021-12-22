#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
import torch.nn.functional as F
from torch import nn

from . import utils


@utils.public
class Zoom(nn.Module):
    def __init__(self, zoom=1.0):
        super().__init__()
        self.zoom = zoom

    def __repr__(self):
        return f"Zoom({self.zoom})"

    def __str__(self):
        return repr(self)

    def forward(self, a):
        assert a.ndim == 4
        if self.zoom == 1.0:
            return a
        return F.interpolate(a, scale_factor=self.zoom, recompute_scale_factor=False)


@utils.public
class HeightTo(nn.Module):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def __repr__(self):
        return f"HeightTo({self.height})"

    def __str__(self):
        return repr(self)

    def forward(self, a):
        assert a.ndim == 4
        zoom = float(self.height) / float(a.shape[2])
        result = F.interpolate(a, scale_factor=zoom, recompute_scale_factor=False)
        return result


@utils.public
class GrayDocument(nn.Module):
    def __init__(self, noise=0.0, autoinvert=True):
        super().__init__()
        self.noise = noise
        self.autoinvert = autoinvert

    def __repr__(self):
        return f"GrayDocument(noise={self.noise}, autoinvert={self.autoinvert})"

    def __str__(self):
        return repr(self)

    def forward(self, a):
        assert a.ndim == 3 or a.ndim == 4
        assert isinstance(a, torch.Tensor)
        if a.dtype == torch.uint8:
            a = a.float() / 255.0
        if a.ndim == 4:
            a = torch.mean(a, 1)
        if a.ndim == 3:
            a = a.unsqueeze(1)
        for i in range(a.shape[0]):
            a[i] -= a[i].min().item()
            a[i] /= max(0.5, a[i].max().item())
            if self.autoinvert and a[i].mean().item() > 0.5:
                a[i] = 1.0 - a[i]
            if self.noise > 0:
                d, h, w = a[i].shape
                a[i] += self.noise * torch.randn(d, h, w, device=a.device)
            a[i] = a[i].clip(0, 1)
        return a


@utils.public
class Spectrum(nn.Module):
    def __init__(self, nonlin="logplus1"):
        nn.Module.__init__(self)
        self.nonlin = nonlin

    def forward(self, x):
        inputs = torch.stack([x, torch.zeros_like(x)], dim=-1)
        mag = torch.fft.fftn(torch.view_as_complex(inputs), dim=(2, 3)).abs()
        if self.nonlin is None:
            return mag
        elif self.nonlin == "logplus1":
            return (1.0 + mag).log()
        elif self.nonlin == "sqrt":
            return mag ** 0.5
        else:
            raise ValueError(f"{self.nonlin}: unknown nonlinearity")

    def __repr__(self):
        return f"Spectrum-{self.nonlin}"


@utils.public
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))[:, :, 0, 0]


@utils.public
class MaxReduce(nn.Module):
    d: int

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x):
        return x.max(self.d)[0]
