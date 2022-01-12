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
import ray
import os
import sys

from . import confparse, utils, jittable, degrade


import typer

app = typer.Typer()
@app.command()
def extract(
        source: str, dest: str, target_class: int = 0, maxsize: str = "800, 800", minsize: str = "256, 256", show: float = -1.0,
):
    minsize = eval(f"({minsize})")
    maxsize = eval(f"({maxsize})")
    src = wds.WebDataset(source).decode("rgb").rename(ppm="jpg;jpeg;png;gif;ppm")
    sink = wds.TarWriter(dest)
    for i, sample in enumerate(src):
        if i % 100 == 0:
            print(i)
        image = sample["ppm"]
        assert image.ndim == 3, image.shape
        assert isinstance(image, np.ndarray), type(image)
        assert image.dtype == np.float32, image.dtype
        assert image.shape[-1] == 3, image.shape

        # randomly scale up/down
        scale = random.uniform(0.25, 1.5)
        image = ndi.zoom(image, (scale, scale, 1), order=1).clip(0, 1)

        # make sure it's large enough
        if image.shape[0] < minsize[0] or image.shape[1] < minsize[1]:
            continue

        # crop if too large
        if image.shape[0] > maxsize[0]:
            delta = image.shape[0] - maxsize[0]
            offset = random.randint(0, delta)
            image = image[offset : offset + maxsize[0], ...]
            assert image.shape[0] == maxsize[0], image.shape
        if image.shape[1] > maxsize[1]:
            delta = image.shape[1] - maxsize[1]
            offset = random.randint(0, delta)
            image = image[:, offset : offset + maxsize[1], ...]
            assert image.shape[1] == maxsize[1], image.shape

        assert image.shape[0] <= maxsize[0] and image.shape[1] <= maxsize[1], image.shape

        # apply random gamma correction
        image -= image.min()
        image /= image.max()
        gamma = np.exp(random.uniform(np.log(0.3), np.log(3.0)))
        image **= gamma

        if random.random() < 0.7:
            # binary / gray scale processing
            image = image.mean(axis=2)
            image -= image.min()
            image /= image.max()
            if random.random() < 0.7:
                # unsharp masking
                sigma = random.uniform(1.0, 5.0)
                alpha = random.uniform(0.0, 1.0)
                image -= alpha * ndi.gaussian_filter(image, (sigma, sigma), mode="reflect")
                # absolute value after unsharp masking
                if random.random() < 0.5:
                    image = np.abs(image)
                image -= image.min()
                image /= image.max()
            if random.random() < 0.5:
                # random thresholding
                image = (image > random.uniform(0.2, 0.8)).astype(np.float32)
                # sigma = random.uniform(0.0, 2.0)
                # image = ndi.gaussian_filter(image, (sigma, sigma), mode="reflect")
                # image = degrade.noisify(image, amp1=0.1, amp2=0.1)
            if random.random() < 0.5:
                # random inversion
                image = 1.0 - image
            image = np.stack([image] * 3, axis=2)
            assert image.ndim == 3, image.shape
            assert image.shape[-1] == 3, image.shape
        assert image.ndim == 3, image.shape
        assert image.shape[-1] == 3, image.shape
        if show > 0.0:
            plt.clf()
            plt.imshow(image)
            plt.ginput(1, show)
        target = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        target[:, :, :] = target_class
        output = {}
        output["__key__"] = sample["__key__"]
        output["jpg"] = image
        output["seg.png"] = target
        sink.write(output)


@ray.remote
def gsextract(fname, srcbucket="gs://nvdata-openimages", destbucket="gs://nvdata-synthfigs"):
    print(f"extracting {fname}", file=sys.stderr)
    os.system(f"gsutil cp {srcbucket}/{fname}  /tmp/{fname}")
    extract("/tmp/{fname}", "/tmp/{fname}-extracted")
    os.system(f"gsutil cp /tmp/{fname}-extracted {destbucket}/{fname}")
    os.system(f"rm /tmp/{fname}")
    os.system(f"rm /tmp/{fname}-extracted")


@app.command()
def openimages():
    sources = [x.strip() for x in os.popen("gsutil gs://nvdata-openimages/openimages-train*.tar").readlines()]
    os.system("gsutil mb gs://nvdata-synthfigs")
    os.system("gsutil rm gs://nvdata-synthfigs/*.tar")
    result = ray.get([gsextract.remote(fname) for fname in sources])
    print(result)


if __name__ == "__main__":
    app()
