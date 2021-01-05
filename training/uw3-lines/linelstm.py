from torchmore import flex, layers, combos
from torch import nn


class ModPad(nn.Module):
    def __init__(self, mod=8):
        super().__init__()
        self.mod = mod

    def forward(self, a):
        mod = self.mod
        bs, d, h, w = a.shape
        nh = ((h + mod - 1) // mod) * mod
        nw = ((w + mod - 1) // mod) * mod
        result = nn.functional.pad(a, (0, nw - w, 0, nh - h))
        # print(a.shape, result.shape, file=sys.stderr)
        nbs, nd, nh, nw = result.shape
        assert nh % mod == 0 and nw % mod == 0
        assert nbs == bs and nd == d and nh >= h and nw >= w
        return result


def project_and_lstm(d, noutput, num_layers=1):
    return [
        layers.Fun("lambda x: x.sum(2)"),  # BDHW -> BDW
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    ]


def make_model(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        ModPad(8),
        *combos.conv2d_block(32, 3, repeat=2),
        combos.make_unet([32, 64, 128], sub=flex.BDHW_LSTM(256)),
        *combos.conv2d_block(64, 3, repeat=2),
        *project_and_lstm(100, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model
