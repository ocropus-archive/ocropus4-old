import os
import random as pyrand
from typing import List
from functools import partial

from math import exp, log, cos, sin
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import typer
from torch.utils.data import DataLoader
from webdataset import Dataset
import ocrodeg

from . import slog

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


app = typer.Typer()


def get_patch(image, shape, center, m=np.eye(2), order=1):
    yx = np.array(center, "f")
    hw = np.array(shape, "f")
    offset = yx - np.dot(m, hw / 2.0)
    return ndi.affine_transform(
        image, m, offset=offset, output_shape=shape, order=order
    )


def degrade(patch):
    p = pyrand.uniform(0.0, 1.0)
    if p < 0.3:
        return patch
    if p < 0.7:
        sigma = pyrand.uniform(0.2, 5.0)
        maxdelta = pyrand.uniform(0.5, 3.0)
        noise = ocrodeg.bounded_gaussian_noise(patch.shape, sigma, maxdelta)
        patch = ocrodeg.distort_with_noise(patch, noise)
        gsigma = pyrand.uniform(0.0, 2.5)
        if gsigma > 0.5:
            patch = ndi.gaussian_filter(patch, gsigma)
        return patch
    else:
        return 1.0 - ocrodeg.printlike_multiscale(1.0 - patch)


def patches(
    page,
    shape=(256, 256),
    spacing=64,
    flip=False,
    invert=False,
    alpha=(0.0, 0.0),
    scale=(1.0, 1.0),
    order=1,
):
    h, w = page.shape[:2]
    smooth = ndi.uniform_filter(page, 100)
    mask = smooth > np.percentile(smooth, 70)
    for i in range(0, h, spacing):
        for j in range(1, w, spacing):
            y = i + pyrand.randrange(0, spacing)
            x = j + pyrand.randrange(1, spacing)
            if y >= h or x >= w:
                continue
            if not mask[y, x]:
                continue
            a = pyrand.uniform(*alpha)
            s = exp(pyrand.uniform(log(scale[0]), log(scale[1])))
            m = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]], "f") / s
            result = get_patch(page, shape, (y, x), m=m, order=order)
            rangle = 0
            if flip:
                rangle = pyrand.choice([0, 1, 2, 3]) * 90
                result = ndi.rotate(result, rangle, order=1)
            inv = 0
            if invert:
                if pyrand.uniform(0.0, 1.0) > 0.5:
                    inv = 1
                    result = 1.0 - result
            result = degrade(result)
            yield result, (rangle // 90, a, s, inv)


def binned(x, r, n):
    assert x >= -r and x <= r
    return np.clip(int(n * float(x + r) / 2.0 / r), 0, n)


def rot_pipe(source, shape=(256, 256), spacing=64):
    for (page,) in source:
        if np.mean(page) > 0.5:
            page = 1.0 - page
        for patch, params in patches(page, shape=shape, spacing=spacing, flip=True):
            r, _, _, _ = params
            yield torch.tensor(patch), r


def skew_pipe(
    source, shape=(256, 256), spacing=64, alpha=0.1, scale=3.0, abins=20, sbins=20
):
    for (page,) in source:
        if np.mean(page) > 0.5:
            page = 1.0 - page
        for patch, params in patches(
            page,
            shape=shape,
            spacing=spacing,
            flip=False,
            alpha=(-alpha, alpha),
            scale=(1.0 / scale, scale),
        ):
            _, a, s, _ = params
            abin = binned(a, alpha, abins)
            sbin = binned(log(s), log(scale), sbins)
            yield torch.tensor(patch), abin, sbin


def make_loader(
    urls,
    batch_size=16,
    extensions="image.png;framed.png;page.png;png;page.jpg;jpg;jpeg",
    shuffle=0,
    num_workers=4,
    pipe=rot_pipe,
):
    training = Dataset(urls).shuffle(shuffle).decode("l").to_tuple(extensions)
    training.pipe(pipe)
    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


def load_model(fname):
    assert fname is not None, "provide model with --mdef or --load"
    assert os.path.exists(fname), f"{fname} does not exist"
    assert fname.endswith(".py"), f"{fname} must be a .py file"
    src = open(fname).read()
    mod = slog.load_module("mmod", src)
    assert "make_model" in dir(
        mod
    ), f"{fname} source does not define make_model function"
    return mod, src


@app.command()
def show_rot(urls: List[str]):
    """Show training samples for page rotation"""
    training = make_loader(urls, shuffle=5000)
    plt.ion()
    for patches, rots in training:
        plt.imshow(patches[0].numpy())
        plt.title(repr(int(rots[0])))
        plt.show()
        plt.ginput(1, 100)
        plt.clf()


@app.command()
def show_skew(urls: List[str], abins: int = 20, sbins: int = 20):
    """Show training samples for skew"""
    training = make_loader(urls, pipe=partial(skew_pipe, abins=abins, sbins=sbins), shuffle=5000)
    plt.ion()
    for patches, abin, sbin in training:
        plt.imshow(patches[0].numpy())
        plt.title(f"{abin[0]-abins//2} {sbin[0]-sbins//2}")
        plt.show()
        plt.ginput(1, 100)
        plt.clf()


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
