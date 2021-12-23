import torch
from torch import nn
from torchmore import combos, flex, layers
from typing import List, Tuple, Union
import scipy.ndimage as ndi
import numpy as np
from numpy import amax, arange, newaxis, tile

from . import ocrlayers, utils

ninput = 3


def ctc_decode(
    probs: torch.Tensor, sigma: float = 1.0, threshold: float = 0.7, full: bool = False
) -> Union[List[int], Tuple[List[int], List[float]]]:
    """Perform simple CTC decoding of the probability outputs from an LSTM or similar model.

    Args:
        probs (torch.Tensor): probabilities in BDL format.
        sigma (float, optional): smoothing. Defaults to 1.0.
        threshold (float, optional): thresholding of output probabilities. Defaults to 0.7.
        full (bool, optional): return both classes and probabilities. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], List[float]]]: [description]
    """
    if not isinstance(probs, np.ndarray):
        probs = probs.detach().cpu().numpy()
    probs = probs.T
    delta = np.amax(abs(probs.sum(1) - 1))
    assert delta < 1e-4, f"input not normalized ({delta}); did you apply .softmax()?"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, newaxis]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = tile(labels[:, newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


def pack_for_ctc(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a list of tensors into tensors in the format required by CTC.

    Args:
        seqs (List[torch.Tensor]): list of tensors (integer sequences)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: packed tensor
    """
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def charset_ascii():
    return "".join([chr(c) for c in range(128)])


class TextModel(nn.Module):
    """Word-level text model."""

    def __init__(
        self, mname, *, config={}, charset: str = "ocropus.textmodels.charset_ascii", unknown_char: int = 26
    ):
        super().__init__()
        self.charset = utils.load_symbol(charset)()
        noutput = len(self.charset)
        self.model = utils.load_symbol(mname)(noutput=noutput, **config)
        self.unknown_char = unknown_char

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        result = torch.zeros(len(s), dtype=torch.int64)
        for i, c in enumerate(s):
            result[i] = self.charset.index(c) if c in self.charset else self.unknown_char
        return result

    @torch.jit.export
    def decode_str(self, l: torch.Tensor) -> str:
        result = ""
        for c in l:
            result += self.charset[c] if c < len(self.charset) else chr(self.unknown_char)
        return result

    @torch.jit.export
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.min() >= 0.0 and images.max() <= 1.0
        b, c, h, w = images.shape
        for i in range(b):
            images[i] -= images[i].min()
            images[i] /= images[i].max() + 1e-6
        assert b >= 1 and b <= 16384
        assert c == 3
        assert h >= 12 and h <= 512 and w > 15 and w <= 2048
        result = self.model.forward(images)
        assert result.shape[:2] == (b, len(self.charset))
        # assert result.shape[2] >= w - 32 and result.shape[2] <= w + 16, (images.shape, result.shape)
        return result

    @torch.jit.export
    def standardize(self, images: torch.Tensor) -> None:
        b, c, h, w = images.shape
        assert c == torch.tensor(1) or c == torch.tensor(3)
        if c == torch.tensor(1):
            images = images.repeat(1, 3, 1, 1)
            b, c, h, w = images.shape
        assert images.min() >= 0.0 and images.max() <= 1.0
        for i in range(len(images)):
            images[i] -= images[i].min()
            images[i] /= torch.max(images[i].amax(), torch.tensor([0.01], device=images[i].device))
            if images[i].mean() > 0.5:
                images[i] = 1 - images[i]


@utils.model
def ctext_model_211124(noutput=None, shape=(1, ninput, 48, 300)):
    model = nn.Sequential(
        layers.ModPadded(
            64,
            combos.make_unet(
                [96, 96, 96, 128, 192, 256],
                sub=nn.Sequential(*combos.conv2d_block(256, 3, repeat=1)),
            ),
        ),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(128, 11, padding=11 // 2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_210910(noutput=None, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(100, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211217(noutput=None, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(100, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(300, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211215(
    noutput=None,
    shape=(1, ninput, 48, 300),
    depth=4,
    width=32,
    growth=1.5,
    xshrink=10,
    yshrink=10,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    fmpx = 1.0 / (float(xshrink) ** (1.0 / depth))
    fmpy = 1.0 / (float(yshrink) ** (1.0 / depth))
    initial = []
    if lstm_initial > 0:
        initial += flex.Lstm(lstm_initial)
    for i in range(depth):
        initial += combos.conv2d_block(int(width * (growth ** depth)), fmp=(fmpy, fmpx), repeat=2)
    model = nn.Sequential(
        layers.KeepSize(sub=nn.Sequential(*initial)),
        flex.Lstm2(lstm_2d),
        ocrlayers.MaxReduce(2),
        # flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(width * 4, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(lstm_final, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211221(
    noutput=None,
    shape=(1, ninput, 64, 300),
    levels=4,
    depth=32,
    height=64,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [int(0.5 + 32 * (1.5 ** i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2 ** depth,
            combos.make_unet(depths, sub=flex.Lstm2d(depths[-1])),
        ),
        flex.Lstm2d(lstm_2d),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(depth * 4, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Lstm1d(lstm_final, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def ctext_model_211221(
    noutput=None,
    shape=(1, ninput, 48, 300),
    levels=4,
    depth=5,
    height=64,
    last=256,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [int(0.5 + 32 * (1.5 ** i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2 ** depth,
            combos.make_unet(depths, sub=flex.Conv2d(depths[-1], 3, padding=1)),
        ),
        flex.Conv2d(depth * 4, 3, padding=1),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def local_text_model_211221(
    noutput=None,
    shape=(1, ninput, 64, 300),
    levels=4,
    depth=5,
    height=64,
    last=256,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [int(0.5 + 32 * (1.5 ** i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2 ** depth,
            combos.make_unet(depths, sub=flex.Conv2d(depths[-1], 3, padding=1)),
        ),
        flex.Conv2d(depth * 4, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv2d(noutput, 1),
        nn.Softmax(1),
        ocrlayers.MaxReduce(2),
        ocrlayers.Log(),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211222(noutput=None, height=48, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [32, 64, 96, 128]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            16,
            combos.make_unet(depths, sub=flex.Lstm2d(196)),
        ),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(300, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Lstm1d(300, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


