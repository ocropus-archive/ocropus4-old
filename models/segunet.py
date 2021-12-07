from torch import nn
from torchmore import combos, flex, layers


def make_model(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=2),
        combos.make_unet([96, 128, 192, 256]),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model
