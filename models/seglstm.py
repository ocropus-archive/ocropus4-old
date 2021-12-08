from torch import nn
from torchmore import combos, flex, layers


def make_model(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.ModPad(8),
        combos.make_unet([64, 128, 256], sub=flex.BDHW_LSTM(100)),
        *combos.conv2d_block(64, repeat=2),
        flex.BDHW_LSTM(16),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model
