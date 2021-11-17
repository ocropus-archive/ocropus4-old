import re

import torch
import torch.nn.functional as F
import typer
from torch import nn
from torchmore import combos, flex, inputstats, layers

from . import loading, ocrlayers, utils
from .utils import model
from .ocrorec import ctc_decode

app = typer.Typer()


ninput = 3

def make_rgb_float(inputs):
    assert inputs.ndim == 4, inputs.shape
    if inputs.dtype == torch.uint8:
        inputs = inputs.float() / 255.0
    elif inputs.dtype == torch.float16 or inputs.dtype == torch.float32:
        assert inputs.max() <= 1.0 and inputs.min() >= 0.0
    if inputs.shape[1] == 1:
        inputs = inputs.repeat(1, 3, 1, 1)
    else:
        assert inputs.shape[1] == 3, inputs.shape
    return inputs

class TextModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        inputs = make_rgb_float(inputs)
        return self.model(inputs)

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs.detach().cpu().softmax(1)

    def predict_batch(self, inputs, **kw):
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result


@model
def binarization_210910(shape=(2, ninput, 161, 391)):
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
    flex.shape_inference(model, shape)
    return model


@model
def cbinarization_large_210910(shape=(2, ninput, 161, 391)):
    """A purely convolutional U-net based model."""
    model = nn.Sequential(
        layers.ModPadded(
            32,
            combos.make_unet(
                [32, 64, 128, 256, 512], sub=nn.Sequential(*combos.conv2d_block(256, 3, repeat=1))
            ),
        ),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, shape)
    return model


@model
def cbinarization_210910(shape=(2, ninput, 161, 391)):
    """A purely convolutional U-net based model."""
    model = nn.Sequential(
        layers.ModPadded(
            32,
            combos.make_unet(
                [8, 16, 24, 32, 64], sub=nn.Sequential(*combos.conv2d_block(64, 3, repeat=1))
            ),
        ),
        flex.Conv2d(1, 3, padding=1),
        nn.Sigmoid(),
    )
    flex.shape_inference(model, shape)
    return model


@model
def page_orientation_210910(shape=(2, ninput, 256, 256)):
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
    flex.shape_inference(model, shape)
    return model


@model
def page_skew_210910(noutput, shape=(2, ninput, 256, 256), r=5, nf=8, r2=5, nf2=4):
    """A model for page skew using Fourier transforms."""
    model = nn.Sequential(
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
    flex.shape_inference(model, shape)
    return model


@model
def page_scale_210910(noutput, shape=(2, ninput, 512, 512), r=5, nf=8, r2=5, nf2=4):
    """A model for page scale using Fourier transforms."""
    model = nn.Sequential(
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
    flex.shape_inference(model, shape)
    return model


@model
def text_model_210910(noutput=1024, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = TextModel(
        nn.Sequential(
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
    )
    flex.shape_inference(model, shape)
    return model


@model
def segmentation_model_210910(noutput=4, shape=(1, ninput, 512, 512)):
    """Page segmentation using U-net and LSTM combos."""
    model = nn.Sequential(
        layers.ModPadded(
            8,
            combos.make_unet([32, 64, 96], sub=flex.BDHW_LSTM(100)),
        ),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, shape)
    return model


@model
def publaynet_model_210910(noutput=4, shape=(1, ninput, 512, 512)):
    """Layout model tuned for PubLayNet."""
    model = nn.Sequential(
        layers.ModPadded(
            16,
            combos.make_unet([40, 60, 80, 100], sub=flex.BDHW_LSTM(100)),
        ),
        *combos.conv2d_block(48, 3, repeat=2),
        flex.BDHW_LSTM(32),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, shape)
    return model


@app.command()
def list(long: bool = False):
    for model in utils.all_models:
        if long:
            print(model.__name__)
            print()
            print(re.sub(r"(?m)^", "    ", model.__doc__))
            print()
        else:
            print(model.__name__)


@app.command()
def show(name: str, kw: str = ""):
    for model in utils.all_models:
        if model.__name__ == name:
            instance = model(**eval(f"dict({kw})"))
            print(instance)
            break


@app.command()
def load(fname: str, kw: str = ""):
    kw = eval(f"dict({kw})")
    model = loading.load_or_construct_model(fname, **kw)
    print(model)


if __name__ == "__main__":
    app()
