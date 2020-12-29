#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
from ocrlib import utils
from ocrlib import ocroseg
from ocrlib import slog


def test_segtrainer():
    with open("models/segunet.py") as stream:
        text = stream.read()
    mmod = slog.load_module("mmod", text)
    model = mmod.make_model()
    trainer = ocroseg.SegTrainer(model, savedir=False)
    trainer.set_lr(1e-3)
    xs = torch.zeros((1, 1, 512, 512))
    ys = torch.zeros((1, 512, 512)).long()
    trainer.train_batch(xs, ys)
