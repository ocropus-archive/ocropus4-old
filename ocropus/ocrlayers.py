#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

@utils.public
class Zoom(nn.Module):
    """Zoom layer."""
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
    """Ensure that the input height is equal to the given height."""
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
    """Ensure that the output is a single channel image.

    Images are normalized and a small amount of noise is added."""

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
    """Generate a spectrum from an image."""
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
    """Adaptive 2D global average pooling."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))[:, :, 0, 0]


@utils.public
class MaxReduce(nn.Module):
    """Max Reduce Layer."""
    d: int

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x):
        return x.max(self.d)[0]


@utils.public
class Log(nn.Module):
    """Simple logarithm layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.log()


class Nonlin(nn.Module):
    """Generic nonlinearity."""
    def __init__(self, kind="relu", label="default", inplace=False, slope=0.2):
        super().__init__()
        self.inplace = inplace
        self.kind = kind
        self.label = label
        self.slope = slope

    def forward(self, x):
        kind = self.kind
        if kind == "relu":
            return F.relu(x, inplace=self.inplace)
        elif kind == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope, inplace=self.inplace)
        elif kind == "elu":
            return F.elu(x, inplace=self.inplace)
        elif kind == "selu":
            return F.selu(x, inplace=self.inplace)
        elif kind == "prelu":
            return F.prelu(x, inplace=self.inplace)
        elif kind == "softplus":
            return F.softplus(x, inplace=self.inplace)
        elif kind == "softshrink":
            return F.softshrink(x, inplace=self.inplace)
        elif kind == "tanh":
            return F.tanh(x, inplace=self.inplace)
        elif kind == "sigmoid":
            return F.sigmoid(x, inplace=self.inplace)
        elif kind == "softmax":
            return F.softmax(x, dim=-1)
        elif kind == "log_softmax":
            return F.log_softmax(x, dim=-1)
        elif kind == "none":
            return x
        else:
            raise ValueError("Unknown nonlinearity: {}".format(kind))

    def __repr__(self):
        return self.kind


class Norm2d(nn.Module):
    """Generic 2D normalization module."""
    def __init__(self, kind="batch", label="default"):
        super().__init__()
        self.kind = kind
        self.norm = None
        self.label = label

    def forward(self, x):
        if self.norm is None:
            if self.kind == "batch":
                self.norm = nn.BatchNorm2d(x.shape[1]).to(x.device)
            elif self.kind == "instance":
                self.norm = nn.InstanceNorm2d(x.shape[1]).to(x.device)
            elif self.kind == "layer":
                self.norm = nn.LayerNorm(x.shape[1]).to(x.device)
            else:
                raise ValueError("Unknown normalization: {}".format(self.kind))
        return self.norm(x)

    def __repr__(self):
        return f"Norm2d({self.kind}, {self.label})"


class ResnetBypass(nn.Module):
    """Resnet bypass layer."""
    def __init__(self, *args, weight=1.0):
        super(ResnetBypass, self).__init__()
        if len(args) > 1:
            self.block = nn.Sequential(*args)
        else:
            self.block = args[0]
        self.weight = weight

    def forward(self, x):
        return self.block(x) + x * self.weight


def init_weights(net, type:str="normal", gain:float=0.02):
    """Initialize network weights."""

    weight_init = dict(
        normal=lambda x: nn.init.normal_(x, 0, gain),
        xavier=lambda x: nn.init.xavier_normal_(x, gain=gain),
        kaiming=lambda x: nn.init.kaiming_normal_(x, a=0, mode="fan_in"),
        orthogonal=lambda x: nn.init.orthogonal_(x, gain=gain),
    )

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            weight_init[type](m.weight.data)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
