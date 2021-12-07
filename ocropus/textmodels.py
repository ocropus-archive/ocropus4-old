
import torch
from torch import nn
from torchmore import combos, flex, inputstats, layers

from . import ocrlayers, utils
from .utils import model

ninput = 3


def charset_ascii():
    return "".join([chr(c) for c in range(128)])


class TextModel(nn.Module):
    """Word-level text model."""

    def __init__(self, mname, *, config={}, charset: str = "ascii", unknown_char: int = 26):
        super().__init__()
        factory = globals()[mname]
        self.model = factory(**config)
        self.charset = globals()[f"charset_{charset}"]()
        self.unknown_char = unknown_char

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        result = torch.zeros(len(s), dtype=torch.int64)
        for i, c in enumerate(s):
            result[i] = (
                self.charset.index(c) if c in self.charset else self.unknown_char
            )
        return result

    @torch.jit.export
    def decode_str(self, l: torch.Tensor) -> str:
        result = ""
        for c in l:
            result += (
                self.charset[c] if c < len(self.charset) else chr(self.unknown_char)
            )
        return result

    @torch.jit.export
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.min() >= 0 and images.max() <= 1
        self.standardize(images)
        b, c, h, w = images.shape
        assert b >= 1 and b <= 16384
        assert c == 3
        assert h >= 12 and h <= 512 and w > 15 and w <= 2048
        result = self.model.forward(images)
        assert result.shape[:2] == (b, len(self.charset))
        assert result.shape[2] >= w - 32 and result.shape[2] <= w + 16
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
            images[i] /= torch.max(
                images[i].amax(), torch.tensor([0.01], device=images[i].device)
            )
            if images[i].mean() > 0.5:
                images[i] = 1 - images[i]


@model
def ctext_model_211124(noutput=1024, shape=(1, ninput, 48, 300)):
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
            flex.ConvTranspose1d(400, 1, stride=2, padding=1),
            flex.Conv1d(100, 3, padding=1),
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
