#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#
from __future__ import unicode_literals

import torch
from ocrlib import ocrmodels
from ocrlib import ocrhelpers


def test_segtrainer():
    model = ocrmodels.make("seg_conv")
    trainer = ocrhelpers.SegTrainer(model, savedir=False)
    trainer.set_lr(1e-3)
    xs = torch.zeros((1, 1, 64, 64))
    ys = torch.zeros((1, 64, 64)).long()
    trainer.train_batch(xs, ys)
    trainer.train((xs, ys) for _ in range(10))
