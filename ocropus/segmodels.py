import torch
from torch import nn
from torchmore import combos, flex, layers

from . import utils
from .utils import model

ninput = 3


class SegModel(nn.Module):
    def __init__(self, mname, *, config={}):
        super().__init__()
        self.model = utils.load_symbol(mname)(**config)

    @torch.jit.export
    def forward(self, images):
        assert images.min() >= 0 and images.max() <= 1
        self.standardize(images)
        b, c, h, w = images.shape
        assert b >= 1 and b <= 16384
        assert c == 3
        assert h >= 64 and h <= 16384
        assert w >= 64 and w <= 16384
        result = self.model(images)
        assert result.shape[0] == b
        assert result.shape[1] >= 1 and result.shape[1] <= 16
        assert result.shape[2:] == images.shape[2:]
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
