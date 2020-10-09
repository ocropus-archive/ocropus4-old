import torch
from torch import nn
from torchmore import flex, layers, combos
import os
import sys
import glob
import re

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53


def make(name, *args, device=default_device, **kw):
    model = eval("make_" + name)(*args, **kw)
    if device is not None:
        model.to(device)
    model.model_name = name
    return model


def extract_save_info(fname):
    fname = re.sub(r".*/", "", fname)
    match = re.search(r"([0-9]{3})+-([0-9]{9})", fname)
    if match:
        return int(match.group(1)), float(match.group(2)) * 1e-6
    else:
        return 0, -1


def load_latest(model, pattern=None, error=False):
    if pattern is None:
        name = model.model_name
        pattern = f"models/{name}-*.pth"
    saves = sorted(glob.glob(pattern))
    if error:
        assert len(saves) > 0, f"no {pattern} found"
    elif len(saves) == 0:
        print(f"no {pattern} found", file=sys.stderr)
        return 0, -1
    else:
        print(f"loading {saves[-1]}", file=sys.stderr)
        model.load_state_dict(torch.load(saves[-1]))
        return extract_save_info(saves[-1])


################################################################
# layer combinations
################################################################

# ocr_output = "BLD"
ocr_output = "BDL"


def project_and_lstm(d, noutput, num_layers=1):
    return [
        layers.Fun("lambda x: x.sum(2)"),  # BDHW -> BDW
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output),
    ]


def project_and_conv1d(d, noutput, r=5):
    return [
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output),
    ]


################################################################
# segmentation models
################################################################


def make_seg_lstm(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        combos.make_unet([64, 128, 256], sub=flex.BDHW_LSTM(100)),
        *combos.conv2d_block(64, repeat=2),
        flex.BDHW_LSTM(16),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model


def make_seg_lstm_simple(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        combos.make_unet([64, 128], sub=flex.BDHW_LSTM(100)),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model


def make_seg_lstm_small(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(
            sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=2),
                *combos.conv2d_block(100, 3, repeat=2),
                flex.BDHW_LSTM(100),
            )
        ),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model


def make_seg_unet_lstm(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(32, 3, repeat=2),
        combos.make_unet([64, 96, 128], sub=flex.BDHW_LSTM(100)),
        *combos.conv2d_block(32, 3, repeat=2),
        flex.BDHW_LSTM(20),
        flex.Conv2d(noutput, 3),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model


def make_seg_unet(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=2),
        combos.make_unet([96, 128, 192, 256]),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 3, padding=1),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model


################################################################
# old segmentation models
################################################################


def make_seg_conv(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(
            sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=3),
                *combos.conv2d_block(100, 3, mp=2, repeat=3),
                *combos.conv2d_block(200, 3, mp=2, repeat=3),
            )
        ),
        *combos.conv2d_block(400, 5),
        flex.Conv2d(noutput, 3),
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model
