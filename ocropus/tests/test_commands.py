#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import os
import pytest
import webdataset as wds
from ocropus import texttrain
from ocropus import segtrain
import torch.jit

bucket = "pipe:curl -sL https://storage.googleapis.com/ocropus4-test"
mbucket = "pipe:curl -sL https://storage.googleapis.com/ocropus4-models"


def test_data():
    ds = wds.WebDataset(f"{bucket}/gsub-words-test.tar")
    next(iter(ds))


def test_texttrain(tmpdir):
    texttrain.train([])

