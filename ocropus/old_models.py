from torch import nn
from torchmore import layers
from torchmore import flex
from torchmore import combos
from torchmore import inputstats
from .utils import model
from . import ocrlayers


@model
def binarization_210113():
    """A small model combining convolutions and 2D LSTM for binarization."""
    r = 5
    model = nn.Sequential(
        flex.Conv2d(16, r, padding=r // 2),
        flex.BatchNorm2d(),
        nn.ReLU(),
        flex.BDHW_LSTM(4),
        flex.BatchNorm2d(),
        flex.Conv2d(8, 3, padding=1),
        flex.BatchNorm2d(),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, (2, 1, 161, 391))
    return model


@model
def cbinarization_210429():
    """A purely convolutional U-net based model."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("cbinarization"),
        layers.ModPadded(
            32,
            combos.make_unet(
                [32, 64, 128, 256, 512], sub=nn.Sequential(*combos.conv2d_block(256, 3, repeat=1))
            ),
        ),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, (2, 1, 161, 391))
    return model


@model
def cbinarization_210819():
    """A purely convolutional U-net based model."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("cbinarization", mode="nocheck"),
        layers.ModPadded(
            32,
            combos.make_unet(
                [8, 16, 24, 32, 64], sub=nn.Sequential(*combos.conv2d_block(64, 3, repeat=1))
            ),
        ),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, (2, 1, 161, 391))
    return model


@model
def page_orientation_210113(size=256):
    """A model for page orientation using a VGG-like architecture."""

    def block(s, r, repeat=2):
        result = []
        for i in range(repeat):
            result += [flex.Conv2d(8, r, padding=r // 2),
                       flex.BatchNorm2d(), nn.ReLU()]
        result += [nn.MaxPool2d(2)]
        return result

    r = 3
    B, D, H, W = (1, 128), (1, 3), size, size
    model = nn.Sequential(
        layers.CheckSizes(B, D, H, W),
        *block(32, r),
        *block(64, r),
        *block(96, r),
        *block(128, r),
        ocrlayers.GlobalAvgPool2d(),
        flex.Linear(64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        flex.Linear(4),
        layers.CheckSizes(B, 4),
    )
    flex.shape_inference(model, (2, 1, size, size))
    return model


@model
def page_skew_210301(noutput, size=256, r=5, nf=8, r2=5, nf2=4):
    """A model for page skew using Fourier transforms."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        flex.Conv2d(nf, r, padding=r // 2),
        flex.BatchNorm2d(),
        nn.ReLU(),
        ocrlayers.Spectrum(),
        flex.Conv2d(nf2, r2, padding=r2 // 2),
        flex.BatchNorm2d(),
        nn.ReLU(),
        # layers.Reshape(0, [1, 2, 3]),
        layers.Collapse(1, 3),
        flex.Linear(noutput * 10),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Linear(noutput),
    )
    flex.shape_inference(model, (2, 1, size, size))
    return model


@model
def page_scale_210301(noutput, size=(512, 512), r=5, nf=8, r2=5, nf2=4):
    """A model for page scale using Fourier transforms."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        flex.Conv2d(nf, r, padding=r // 2),
        flex.BatchNorm2d(),
        nn.ReLU(),
        ocrlayers.Spectrum(),
        flex.Conv2d(nf2, r2, padding=r2 // 2),
        flex.BatchNorm2d(),
        nn.ReLU(),
        # layers.Reshape(0, [1, 2, 3]),
        layers.Collapse(1, 3),
        flex.Linear(noutput * 10),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Linear(noutput),
    )
    flex.shape_inference(model, (2, 1, size[0], size[1]))
    return model


@model
def text_model_210218(noutput):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("textmodel"),
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
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


@model
def segmentation_model_210429(noutput=4):
    """Page segmentation using U-net and LSTM combos."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        # ocrlayers.Zoom(0.5),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("segmodel"),
        layers.ModPadded(
            8,
            combos.make_unet([32, 64, 96], sub=flex.BDHW_LSTM(100)),
        ),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 512, 512))
    return model


@model
def publaynet_model_210429(noutput=4):
    """Layout model tuned for PubLayNet."""
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        # ocrlayers.Zoom(0.5),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("segmodel"),
        layers.ModPadded(
            16,
            combos.make_unet([40, 60, 80, 100], sub=flex.BDHW_LSTM(100)),
        ),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 512, 512))
    return model


@model
def text_model_210113(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
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


@model
def segmentation_model_210113(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
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


@model
def segmentation_model_210117(noutput=4):
    model = nn.Sequential(
        inputstats.InputStats("segmodel"),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
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


@model
def text_model_210118(noutput):
    model = nn.Sequential(
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


@model
def segmentation_model_210118(noutput=4):
    model = nn.Sequential(
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


@model
def segmentation_model_210218x(noutput=4):
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
    flex.shape_inference(model, (1, 1, 512, 512))
    return model


@model
def segmentation_model_210218(noutput=4):
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        # ocrlayers.Zoom(0.5),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("segmodel"),
        layers.ModPad(8),
        combos.make_unet([32, 64, 96], sub=flex.BDHW_LSTM(100)),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 512, 512))
    return model


@model
def publaynet_model_210321(noutput=4):
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        # ocrlayers.Zoom(0.5),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("segmodel"),
        layers.ModPad(16),
        combos.make_unet([40, 60, 80, 100], sub=flex.BDHW_LSTM(100)),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 512, 512))
    return model


@model
def cbinarization_210317():
    model = nn.Sequential(
        ocrlayers.GrayDocument(),
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        inputstats.InputStats("cbinarization"),
        layers.ModPad(32),
        combos.make_unet([32, 64, 128, 256, 512], sub=nn.Sequential(
            *combos.conv2d_block(256, 3, repeat=1))),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, (2, 1, 161, 391))
    return model
