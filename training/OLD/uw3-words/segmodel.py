from torch import nn
from torchmore import flex, layers, combos


def make_model(noutput=4):
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
