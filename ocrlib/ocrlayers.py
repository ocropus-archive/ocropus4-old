#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
from torch import nn
import torch.nn.functional as F


class Zoom(nn.Moduel):
    def __init__(self, zoom=1.0):
        super().__init__()
        self.zoom = zoom

    def __str__(self):
        return f"Zoom({self.zoom})"

    def forward(self, a):
        assert a.ndim == 4
        if self.zoom == 1.0:
            return a
        return F.interpolate(a, self.zoom)


class GrayDocument(nn.Module):
    def __init__(self, noise=0.03, autoinvert=True):
        super().__init__()
        self.noise = noise
        self.autoinvert = autoinvert

    def __str__(self):
        return f"GrayDocument(noise={self.noise}, autoinvert={self.autoinvert})"

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
            a[i] += self.noise * torch.randn(*a[i].shape)
            a[i] = a[i].clip(0, 1)
        return a
