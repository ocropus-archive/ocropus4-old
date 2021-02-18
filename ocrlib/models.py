import torch
import torch.nn.functional as F
from torch import nn
from torchmore import layers
from torchmore import flex
from torchmore import combos
from torchmore import inputstats

from . import ocrlayers


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


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))[:, :, 0, 0]


def binarization_210113():
    r = 3
    model = nn.Sequential(
        nn.Conv2d(1, 8, r, padding=r // 2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        layers.BDHW_LSTM(8, 4),
        nn.Conv2d(8, 1, 1),
        nn.Sigmoid(),
    )
    return model


def page_orientation_210113(size=256):
    def block(s, r, repeat=2):
        result = []
        for i in range(repeat):
            result += [flex.Conv2d(8, r, padding=r // 2), flex.BatchNorm2d(), nn.ReLU()]
        result += [nn.MaxPool2d(2)]
        return result

    r = 3
    B, D, H, W = (2, 128), (1, 512), size, size
    model = nn.Sequential(
        layers.CheckSizes(B, D, H, W),
        *block(32, r),
        *block(64, r),
        *block(96, r),
        *block(128, r),
        GlobalAvgPool2d(),
        flex.Linear(64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        flex.Linear(4),
        layers.CheckSizes(B, 4),
    )
    flex.shape_inference(model, (2, 1, size, size))
    return model


def page_skew_210113(noutput, size=256, r=5, nf=8, r2=5, nf2=4):
    B, D, H, W = (2, 128), (1, 512), size, size
    model = nn.Sequential(
        layers.CheckSizes(B, D, H, W),
        nn.Conv2d(1, nf, r, padding=r // 2),
        nn.BatchNorm2d(nf),
        nn.ReLU(),
        Spectrum(),
        nn.Conv2d(nf, nf2, r2, padding=r2 // 2),
        nn.BatchNorm2d(nf2),
        nn.ReLU(),
        layers.Reshape(0, [1, 2, 3]),
        nn.Linear(nf2 * W * H, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, noutput),
        layers.CheckSizes(B, noutput),
    )
    return model


def text_model_210218(noutput):
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("textmodel"),
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.ConvTranspose1d(400, 1, stride=2),
        flex.Conv1d(100, 3),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model


def segmentation_model_210218(noutput=4):
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        # ocrlayers.Zoom(0.5),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("segmodel"),
        layers.ModPad(8),
        layers.KeepSize(
            sub=nn.Sequential(
                *combos.conv2d_block(32, 3, mp=2, repeat=2),
                *combos.conv2d_block(48, 3, mp=2, repeat=2),
                *combos.conv2d_block(96, 3, mp=2, repeat=2),
                flex.BDHW_LSTM(100),
            )
        ),
        flex.BDHW_LSTM(40),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model
