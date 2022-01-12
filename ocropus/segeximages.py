import numpy as np
import random
import pytorch_lightning as pl
import torch
import webdataset as wds
from scipy import ndimage as ndi
from webdataset.filters import default_collation_fn
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import partial
import warnings

from . import confparse, utils, jittable


import typer

app = typer.Typer()


@app.command()
def extract(source: str, dest: str, target_class: int = 0, maxsize: str = "800, 800"):
    maxsize = eval(f"({maxsize})")
    src = wds.WebDataset(source).decode("rgb8").rename(ppm="jpg;jpeg;png;gif;ppm")
    sink = wds.TarWriter(dest)
    for i, sample in enumerate(src):
        if i % 100 == 0:
            print(i)
        image = sample["ppm"]
        assert image.ndim == 3, image.shape
        assert isinstance(image, np.ndarray), type(image)
        assert image.dtype == np.uint8, image.dtype
        assert image.shape[-1] == 3, image.shape
        if image.shape[0] > maxsize[0]:
            delta = image.shape[0] - maxsize[0]
            offset = random.randint(0, delta)
            image = image[offset : offset + maxsize[0], ...]
        if image.shape[1] > maxsize[1]:
            delta = image.shape[1] - maxsize[1]
            offset = random.randint(0, delta)
            image = image[:, offset : offset + maxsize[1], ...]
        target = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        target[:, :, :] = target_class
        output = {}
        output["__key__"] = sample["__key__"]
        output["jpg"] = image
        output["seg.png"] = target
        sink.write(sample)


if __name__ == "__main__":
    app()
