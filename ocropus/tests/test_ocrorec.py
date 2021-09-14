#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
from ocropus import ocrorec
from ocropus import slog
from ocropus import models

def test_linetrainer():
    with open("models/linelstm.py") as stream:
        text = stream.read()
    mmod = slog.load_module("mmod", text)
    model = mmod.make_model(96)
    trainer = ocrorec.TextTrainer(model)
    trainer.set_lr(1e-3)
    xs = torch.zeros((1, 1, 48, 277))
    ys = [torch.tensor([0, 1, 0])]
    trainer.train_batch(xs, ys)


def test_linetrainer():
    model = models.text_model_210910()
    trainer = ocrorec.TextTrainer(model)
    trainer.set_lr(1e-3)
    xs = torch.zeros((1, 1, 48, 277))
    ys = [torch.tensor([0, 1, 0])]
    trainer.train_batch(xs, ys)