from torch import nn
from torchmore import combos, flex, layers


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


def project_and_conv1d(d, noutput, r=5):
    return [
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    ]


################################################################
# uw3-related


def make_lstm2(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        flex.Lstm2(400),
        *project_and_conv1d(800, noutput),
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model


def make_lstm2transpose(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        flex.Lstm2(400),
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.ConvTranspose1d(400, 1, stride=2),
        flex.Conv1d(400, 3),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model


def make_lstm2a(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, repeat=2),
        flex.Lstm2(200),
        *project_and_conv1d(400, noutput),
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model


def make_lstm2_transpose(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, repeat=2),
        *combos.conv2d_block(100, 3, repeat=2),
        *combos.conv2d_block(150, 3, repeat=2),
        flex.ConvTranspose2d(300, 1, stride=(1, 2)),  # <-- undo too tight spacing
        flex.Lstm2(200),
        *project_and_conv1d(400, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model


def make_lstm2_transpose_2(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, repeat=2, mp=2),
        *combos.conv2d_block(100, 3, repeat=2, mp=(2, 1)),
        *combos.conv2d_block(150, 3, repeat=2, mp=(2, 1)),
        flex.Lstm2(200),
        *project_and_lstm(400, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model


def make_lstm_unet(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        ModPad(8),
        *combos.conv2d_block(64, 3, repeat=2),
        combos.make_unet([64, 128]),
        *combos.conv2d_block(128, 3, repeat=2),
        *project_and_lstm(100, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model


def make_lstmb_unet(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet([64, 128, 256, 512]),
        *combos.conv2d_block(128, 3, repeat=2),
        *project_and_lstm(100, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model


################################################################


#
# more
#


def make_lstm_transpose(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, repeat=2),
        *combos.conv2d_block(100, 3, repeat=2),
        *combos.conv2d_block(150, 3, repeat=2),
        *combos.conv2d_block(200, 3, repeat=2),
        layers.Fun("lambda x: x.sum(2)"),  # BDHW -> BDW
        flex.ConvTranspose1d(800, 1, stride=2),  # <-- undo too tight spacing
        # flex.BatchNorm1d(), nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model


def make_lstm_keep(noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(
            mode="nearest",
            dims=[3],
            sub=nn.Sequential(
                *combos.conv2d_block(50, 3, repeat=2),
                *combos.conv2d_block(100, 3, repeat=2),
                *combos.conv2d_block(150, 3, repeat=2),
                layers.Fun("lambda x: x.sum(2)"),  # BDHW -> BDW
            ),
        ),
        flex.Conv1d(500, 5, padding=2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(200, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model


def make_lstm_resnet(noutput, blocksize=5):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm(100, noutput),
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model
