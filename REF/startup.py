from importlib import reload

import os
import sys
import re
import glob
import time
import pickle
import scipy.ndimage as ndi
from itertools import islice
import IPython
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmore import layers, flex
import torchtrainers as tt
from torch.utils.data import DataLoader

from webdataset import WebDataset

import helpers
from helpers import method, ctc_decode, asnp

import matplotlib.pyplot as plt

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

import scipy.ndimage as ndi


def RUN(x):
    print(x, ":", os.popen(x).read().strip())


def scale_to(a, shape):
    scales = array(a.shape, "f") / array(shape, "f")
    result = ndi.affine_transform(a, diag(scales), output_shape=shape, order=1)
    return result


def tshow(a, order, b=0, ax=None, **kw):
    ax = ax or gca()
    if set(order) == set("BHWD"):
        a = layers.reorder(a.detach().cpu(), order, "BHWD")[b].numpy()
    elif set(order) == set("HWD"):
        a = layers.reorder(a.detach().cpu(), order, "HWD").numpy()
    elif set(order) == set("HW"):
        a = layers.reorder(a.detach().cpu(), order, "HW").numpy()
    else:
        raise ValueError(f"{order}: unknown order")
    if a.shape[-1] == 1:
        a = a[..., 0]
    ax.imshow(a, **kw)


if False:
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def encode_str(s):
        return [charset.find(c) + 1 for c in s]

    def decode_str(l):
        return "".join([charset[k - 1] for k in l])

    transforms = [
        lambda x: (torch.tensor(x).float() / 255.0).unsqueeze(2),
        lambda s: torch.tensor(encode_str(s)).long(),
    ]
    # training = helpers.LmdbDataset("data/words-training.lmdb", transforms=transforms)
    # testing = helpers.LmdbDataset("data/word-testing.lmdb", transforms=transforms)
    training = WebDataset(
        "data/words-training.tar",
        decoder="l8",
        extensions="jpg;jpeg;ppm;png txt",
        transforms=transforms,
    )
    testing = WebDataset(
        "data/words-testing.tar",
        decoder="l8",
        extensions="jpg;jpeg;ppm;png txt",
        transforms=transforms,
    )
    training_dl = DataLoader(training, batch_size=5, collate_fn=helpers.collate4ocr)
    testing_dl = DataLoader(testing, batch_size=20, collate_fn=helpers.collate4ocr)

device = torch.device(os.environ.get("device", "cuda:0"))

print("=" * 60)
RUN("date")
RUN("hostname")
RUN("whoami")
RUN("pwd")
print("=" * 60)
print()


reload(helpers)
reload(flex)
reload(layers)
