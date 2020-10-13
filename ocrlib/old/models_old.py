import torch
from torch import nn
from torchmore import flex, layers, combos
import torch.nn.functional as F
import os
import sys
import glob
import re

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53

def make(name, *args, device=default_device, **kw):
    model = eval("make_"+name)(*args, **kw)
    if device is not None:
        model.to(device)
    model.model_name = name
    return model

def extract_save_info(fname):
    fname = re.sub(r'.*/', '', fname)
    match = re.search(r'([0-9]{3})+-([0-9]{9})', fname)
    if match:
        return int(match.group(1)), float(match.group(2))*1e-6
    else:
        return 0, -1

def load_latest(model, pattern=None, error=False):
    if pattern is None:
        name = model.model_name
        pattern = f"models/{name}-*.pth"
    saves = sorted(glob.glob(pattern))
    if error:
        assert len(saves)>0, f"no {pattern} found"
    elif len(saves)==0:
        print(f"no {pattern} found", file=sys.stderr)
        return 0, -1
    else:
        print(f"loading {saves[-1]}", file=sys.stderr)
        model.load_state_dict(torch.load(saves[-1]))
        return extract_save_info(saves[-1])

################################################################
# ## layer combinations
# ###############################################################

#ocr_output = "BLD"
ocr_output = "BDL"

def project_and_lstm(d, noutput, num_layers=1):
    return [
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    ]

def project_and_conv1d(d, noutput, r=5):
    return [
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    ]


################################################################
### entire OCR models
################################################################

def make_conv_only(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_conv_resnet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, mp=2),
        *combos.resnet_blocks(5, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(5, 128),
        *combos.conv2d_block(192, 3, mp=2),
        *combos.resnet_blocks(5, 192),
        *combos.conv2d_block(256, 3, mp=(2, 1)),
        *combos.resnet_blocks(5, 256),
        *combos.conv2d_block(512, 3),
        *project_and_conv1d(512, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_ctc(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, mp=(2, 1)),
        *combos.conv2d_block(100, 3, mp=(2, 1)),
        *combos.conv2d_block(150, 3, mp=2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_normalized(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1),
                     sizes=[None, 1, 80, None]),
        *combos.conv2d_block(50, 3, mp=(2, 1)),
        *combos.conv2d_block(100, 3, mp=(2, 1)),
        *combos.conv2d_block(150, 3, mp=2),
        layers.Reshape(0, [1, 2], 3),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output))
    flex.shape_inference(model, (1, 1, 80, 200))
    return model

def make_lstm_transpose(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, repeat=2),
        *combos.conv2d_block(100, 3, repeat=2),
        *combos.conv2d_block(150, 3, repeat=2),
        *combos.conv2d_block(200, 3, repeat=2),
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        flex.ConvTranspose1d(800, 1, stride=2), # <-- undo too tight spacing
        #flex.BatchNorm1d(), nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_keep(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(
            mode="nearest",
            dims=[3],
            sub=nn.Sequential(
                *combos.conv2d_block(50, 3, repeat=2),
                *combos.conv2d_block(100, 3, repeat=2),
                *combos.conv2d_block(150, 3, repeat=2),
                layers.Fun("lambda x: x.sum(2)") # BDHW -> BDW
            )
        ),
        flex.Conv1d(500, 5, padding=2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(200, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_resnet(noutput=noutput, blocksize=5):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_unet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet([64, 128, 256, 512]),
        *combos.conv2d_block(128, 3, repeat=2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model

def make_lstm2_ctc(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        flex.Lstm2(400),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_seg_conv(noutput=3):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=3),
                *combos.conv2d_block(100, 3, mp=2, repeat=3),
                *combos.conv2d_block(200, 3, mp=2, repeat=3)
            )
        ),
        *combos.conv2d_block(400, 5),
        flex.Conv2d(noutput, 3)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_lstm(noutput=3):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=3),
                *combos.conv2d_block(100, 3, mp=2, repeat=3),
                *combos.conv2d_block(200, 3, mp=2, repeat=3),
                flex.BDHW_LSTM(200)
            )
        ),
        *combos.conv2d_block(400, 5),
        flex.Conv2d(noutput, 3)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_unet(noutput=3):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet([128, 256, 512]),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model
