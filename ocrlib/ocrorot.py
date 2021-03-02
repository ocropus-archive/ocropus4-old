import sys
import os
import random as pyrand
from functools import partial
from math import cos, exp, log, sin
from typing import List
from itertools import islice
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import typer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import webdataset as wds
import torch.fft

from . import slog
from . import utils
from . import loading
from .utils import public

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


app = typer.Typer()


def binned(x, bins):
    return np.argmin(np.abs(np.array(bins) - x))


def unbinned(b, bins):
    return bins[b]


def get_patch(image, shape, center, m=np.eye(2), order=1):
    assert np.amin(image) >= 0 and np.amax(image) <= 1.0
    yx = np.array(center, "f")
    hw = np.array(shape, "f")
    offset = yx - np.dot(m, hw / 2.0)
    return ndi.affine_transform(
        image, m, offset=offset, output_shape=shape, order=order
    ).clip(0, 1)


def rot_samples(page, npatches=32, ntrials=32, shape=(256, 256), alpha=(-0.03, 0.03), scale=(1.0, 1.0)):
    h, w = page.shape[:2]
    smooth = ndi.uniform_filter(page, 100)
    mask = smooth > np.percentile(smooth, 70)
    samples = []
    for _ in range(ntrials):
        if len(samples) >= npatches:
            break
        y, x = pyrand.randrange(0, h), pyrand.randrange(0, w)
        if mask[y, x]:
            samples.append((x, y))
    pyrand.shuffle(samples)
    for x, y in samples:
        a = pyrand.uniform(*alpha)
        s = exp(pyrand.uniform(log(scale[0]), log(scale[1])))
        m = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]], "f") / s
        result = get_patch(page, shape, (y, x), m=m, order=1)
        c = random.randint(0, 3)
        rotated = ndi.rotate(result, 90 * c, order=1).clip(0, 1)
        yield rotated, c


def rot_pipe(source, **kw):
    for (page,) in source:
        yield from rot_samples(page, **kw)


def get_patches(
    page,
    shape=(256, 256),
    flip=False,
    invert=False,
    alpha=(0.0, 0.0),
    scale=(1.0, 1.0),
    order=1,
    npatches=100,
    ntrials=10000,
):
    h, w = page.shape[:2]
    smooth = ndi.uniform_filter(page, 100)
    mask = smooth > np.percentile(smooth, 70)
    samples = []
    for _ in range(ntrials):
        if len(samples) >= npatches:
            break
        y, x = pyrand.randrange(0, h), pyrand.randrange(0, w)
        if mask[y, x]:
            samples.append((x, y))
    pyrand.shuffle(samples)
    for x, y in samples:
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
        result = result.clip(0, 1)
        yield result, (rangle // 90, a, s, inv)


def skew_pipe(source, shape=(256, 256), alpha=(-0.1, 0.1), scale=(0.7, 1.4)):
    for (page,) in source:
        if np.mean(page) > 0.5:
            page = 1.0 - page
        for patch, params in get_patches(
            page,
            shape=shape,
            alpha=alpha,
            scale=scale,
        ):
            _, a, s, _ = params
            yield torch.tensor(patch), a, s


def make_loader(
    urls,
    batch_size=16,
    extensions="nrm.jpg;image.png;framed.png;page.png;png;page.jpg;jpg;jpeg",
    shuffle=0,
    num_workers=4,
    pipe=rot_pipe,
    invert="Auto",
    limit=-1,
):
    training = (
        wds.WebDataset(urls)
        .shuffle(shuffle)
        .decode("l")
        .to_tuple(extensions)
        .map_tuple(lambda image: utils.autoinvert(image, invert))
        .pipe(pipe)
        .shuffle(shuffle)
    )
    if limit > 0:
        training = wds.ResizedDataset(training, limit, limit)
    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


@public
class PageOrientation:
    def __init__(self, fname, check=True):
        self.model = loading.load_only_model(fname)
        self.check = check
        self.debug = int(os.environ.get("DEBUG_PAGEORIENTATION", 0))

    def orientation(self, page, npatches=200, bs=50):
        if self.check:
            assert np.mean(page) < 0.5
        try:
            self.model.cuda()
            patches = get_patches(page, npatches=npatches)
            result = []
            while True:
                batch = [x[0] for x in islice(patches, bs)]
                if len(batch) == 0:
                    break
                inputs = torch.tensor(batch).unsqueeze(1).cuda()
                outputs = self.model(inputs).softmax(1).cpu().detach()
                result.append(outputs)
                if self.debug:
                    for i in range(len(inputs)):
                        plt.ion()
                        plt.imshow(inputs[i, 0].detach().cpu().numpy())
                        plt.title(
                            str(i) + ": " + repr(inputs.shape) + " " + repr(outputs[i])
                        )
                        plt.ginput(1, 1000.0)
                    pass
            self.last = torch.cat(result)
            self.hist = self.last.mean(0)
            return int(self.hist.argmax()) * 90
        finally:
            self.model.cpu()

    def make_upright(self, page):
        angle = self.orientation(page)
        return ndi.rotate(page, -angle)


@public
class PageSkew:
    def __init__(self, fname, check=True):
        self.model = loading.load_only_model(fname)
        self.bins = self.model.extra_["bins"]
        self.check = check

    def skew(self, page, npatches=200, bs=50):
        if self.check:
            assert np.mean(page) < 0.5
        try:
            self.model.cuda()
            patches = get_patches(page, npatches=npatches)
            result = []
            while True:
                batch = [x[0] for x in islice(patches, bs)]
                if len(batch) == 0:
                    break
                inputs = torch.tensor(batch).unsqueeze(1).cuda()
                outputs = self.model(inputs).softmax(1).cpu().detach()
                result.append(outputs)
            self.last = torch.cat(result)
            self.hist = self.last.mean(0)
            bucket = int(self.last.mean(0).argmax())
            return unbinned(bucket, self.bins)
        finally:
            self.model.cpu()

    def deskew(self, page):
        self.angle = self.skew(page) * 180.0 / np.pi
        return ndi.rotate(page, self.angle, order=1)


@public
class PageScale:
    def __init__(self, fname=None, check=True):
        self.model = loading.load_only_model(fname)
        self.check = check

    def skew(self, page, npatches=200, bs=50):
        if self.check:
            assert np.mean(page) < 0.5
        try:
            self.model.cuda()
            patches = get_patches(page, npatches=npatches)
            result = []
            while True:
                batch = [x[0] for x in islice(patches, bs)]
                if len(batch) == 0:
                    break
                inputs = torch.tensor(batch).unsqueeze(1).cuda()
                outputs = self.model(inputs).softmax(1).cpu().detach()
                result.append(outputs)
            self.last = torch.cat(result)
            self.hist = self.last.mean(0)
            bucket = int(self.last.mean(0).argmax())
            return unbinned(bucket, self.bins)
        finally:
            self.model.cpu()

    def deskew(self, page):
        self.angle = self.skew(page) * 180.0 / np.pi
        return ndi.rotate(page, self.angle, order=1)


default_extensions = (
    "bin.png;nrm.jpg;nrm.png;image.png;framed.png;page.png;png;page.jpg;jpg;jpeg"
)


@app.command()
def train_rot(
    urls: List[str],
    nsamples: int = 1000000,
    num_workers: int = 8,
    replicate: int = 1,
    bs: int = 64,
    prefix: str = "rot",
    lrfun="0.3**(3+n//5000000)",
    output: str = "",
    model: str = "page_orientation_210113",
    extensions: str = default_extensions,
    display: float = 0.0,
    invert: str = "Auto",
):
    logger = slog.Logger(fname=output, prefix=prefix)
    logger.sysinfo()
    logger.json("args", sys.argv)
    model = loading.load_or_construct_model(model)
    model.cuda()
    print(model)
    urls = urls * replicate
    training = make_loader(
        urls,
        shuffle=10000,
        num_workers=num_workers,
        batch_size=bs,
        extensions=extensions,
        invert=invert,
        pipe=rot_pipe,
    )
    criterion = nn.CrossEntropyLoss().cuda()
    lrfun = eval(f"lambda n: {lrfun}")
    lr = lrfun(0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    count = 0
    losses = []
    errs = []

    def save():
        avgloss = np.mean(losses[-100:])
        loading.log_model(logger, model, step=count, loss=avgloss)
        print("\nsaved at", count)

    schedule = utils.Schedule()

    for patches, targets in utils.repeatedly(training):
        if count > nsamples:
            break
        if len(patches) < 2:
            print("skipping small batch", file=sys.stderr)
            continue
        patches = patches.type(torch.float).unsqueeze(1).cuda()
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        count += len(patches)
        losses.append(float(loss))
        probs = outputs.detach().cpu().softmax(1)
        pred = probs.argmax(1)
        erate = (pred != targets).sum() * 1.0 / len(pred)
        errs.append(erate)
        if schedule("info", 60, initial=True):
            print(count, np.mean(losses[-50:]), np.mean(errs[-50:]), lr, flush=True)
        if schedule("log", 15 * 60):
            avgloss = np.mean(losses[-100:])
            logger.scalar(
                "train/loss",
                avgloss,
                step=count,
                json=dict(lr=lr),
            )
            logger.flush()
        if display > 0 and schedule("display", display):
            plt.ion()
            plt.imshow(patches[0, 0].detach().cpu().numpy())
            plt.title(f"{targets[0]} {list((100*probs[0].numpy()).astype(int))}")
            plt.ginput(1, 0.001)
        if schedule("save", 15 * 60):
            save()
        if lrfun(count) != lr:
            lr = lrfun(count)
            optimizer = optim.SGD(model.parameters(), lr=lr)

    save()


@app.command()
def train_skew(
    urls: List[str],
    nsamples: int = 1000000,
    num_workers: int = 8,
    bins: str = "np.linspace(-0.1, 0.1, 21)",
    alpha: str = "-0.1, 0.1",
    scale: str = "0.5, 2.0",
    replicate: int = 1,
    bs: int = 64,
    prefix: str = "skew",
    lrfun: str = "0.3**(3+n//5000000)",
    do_scale: bool = False,
    output: str = "",
    limit: int = -1,
    model: str = "page_skew_210113",
    extensions: str = default_extensions,
    display: float = 0.0,
):
    """Trains either skew (=small rotation) or scale models."""

    scale = eval(f"[{scale}]")
    alpha = eval(f"[{alpha}]")
    bins = eval(bins)
    logger = slog.Logger(fname=output, prefix=prefix)
    logger.sysinfo()
    logger.json("args", sys.argv)
    model = loading.load_or_construct_model(model, len(bins))
    model.cuda()
    print(model)
    urls = urls * replicate

    pipe = partial(skew_pipe, scale=scale, alpha=alpha)

    training = make_loader(
        urls,
        shuffle=10000,
        num_workers=num_workers,
        batch_size=bs,
        pipe=pipe,
        limit=limit,
        extensions=extensions,
    )
    criterion = nn.CrossEntropyLoss().cuda()
    lrfun = eval(f"lambda n: {lrfun}")
    lr = lrfun(0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    count = 0
    losses = []
    schedule = utils.Schedule()

    def save():
        avgloss = np.mean(losses[-100:])
        loading.log_model(logger, model, step=count, loss=avgloss, bins=bins)
        logger.flush()
        print("# saved", count, file=sys.stderr)

    for patches, angles, scales in utils.repeatedly(training, verbose=True):
        if count >= nsamples:
            print("# finished at", count)
            break
        targets = angles if not do_scale else scales
        targets = torch.tensor([binned(float(t), bins) for t in targets])
        patches = patches.type(torch.float).unsqueeze(1).cuda()
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        count += len(patches)
        losses.append(float(loss))
        if schedule("info", 60, initial=True):
            print(count, np.mean(losses[-50:]), lr, flush=True)
        if schedule("log", 10 * 60):
            avgloss = np.mean(losses[-100:])
            logger.scalar(
                "train/loss",
                avgloss,
                step=count,
                json=dict(lr=lr),
            )
            logger.flush()
        if schedule("save", 15 * 60):
            save()
        if display > 0 and schedule("display", display):
            plt.ion()
            plt.imshow(patches[0, 0].detach().cpu().numpy())
            plt.title(repr(targets[0]))
            plt.ginput(1, 0.001)
        if lrfun(count) != lr:
            lr = lrfun(count)
            optimizer = optim.SGD(model.parameters(), lr=lr)
    save()


@app.command()
def train_scale(
    urls: List[str],
    nsamples: int = 1000000,
    num_workers: int = 8,
    alpha: str = "-0.1, 0.1",
    scale: str = "0.5, 2.0",
    bins: str = "np.linspace(0.5, 2.0, 21)",
    replicate: int = 1,
    bs: int = 64,
    prefix: str = "scale",
    lrfun: str = "0.3**(3+n//5000000)",
    output: str = "",
    limit: int = -1,
    extensions: str = default_extensions,
    display: float = 0.0,
):
    return train_skew(
        urls,
        nsamples=nsamples,
        num_workers=num_workers,
        bins=bins,
        alpha=alpha,
        scale=scale,
        replicate=replicate,
        bs=bs,
        prefix=prefix,
        lrfun=lrfun,
        output=output,
        do_scale=True,
        limit=limit,
        extensions=extensions,
        display=display,
    )


@app.command()
def predict(
    urls: List[str],
    rotmodel: str = "",
    skewmodel: str = "",
    extensions: str = default_extensions,
    limit: int = 999999999,
    invert: str = "Auto",
):
    rotest = PageOrientation(rotmodel) if rotmodel != "" else None
    skewest = PageSkew(skewmodel) if skewmodel != "" else None
    dataset = wds.Dataset(urls).decode("l").to_tuple("__key__ " + extensions)
    for key, image in islice(dataset, limit):
        image = utils.autoinvert(image, invert)
        rot = rotest.orientation(image) if rotest else None
        skew = skewest.skew(image) if skewest else None
        print(key, image.shape, rot, skew)


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
    training = make_loader(
        urls, pipe=partial(skew_pipe, abins=abins, sbins=sbins), shuffle=5000
    )
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
