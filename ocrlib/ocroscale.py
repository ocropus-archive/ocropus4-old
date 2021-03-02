import os
import sys
import random as pyrand
from math import cos, exp, log, sin
from typing import List
from itertools import islice
import random
from functools import partial

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


patch_size = (512, 512)


default_extensions = (
    "bin.png;nrm.jpg;nrm.png;image.png;framed.png;page.png;png;page.jpg;jpg;jpeg"
)


def binned(x, bins):
    return np.argmin(np.abs(np.array(bins) - x))


def unbinned(b, bins):
    return bins[b]


def get_patch(image, patchsize, center, m=np.eye(2), order=1):
    assert np.amin(image) >= 0 and np.amax(image) <= 1.0
    yx = np.array(center, "f")
    hw = np.array(patchsize, "f")
    offset = yx - np.dot(m, hw / 2.0)
    return ndi.affine_transform(
        image, m, offset=offset, output_shape=patchsize, order=order
    ).clip(0, 1)


def scale_samples(
    page,
    npatches=32,
    ntrials=32,
    patchsize=patch_size,
    alpha=(-0.1, 0.1),
    logscale=(0.0, 0.0),
):
    if not isinstance(alpha, tuple):
        alpha = (-alpha, alpha)
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
        s = exp(pyrand.uniform(logscale[0], logscale[1]))
        m = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]], "f") / s
        result = get_patch(page, patchsize, (y, x), m=m, order=1)
        yield result, s


def scale_pipe(source, logscale=None, nbins=20, **kw):
    assert logscale is not None
    for (page,) in source:
        yield from scale_samples(page, logscale=logscale, **kw)


def make_loader(
    urls,
    batch_size=16,
    extensions=default_extensions,
    shuffle=0,
    num_workers=4,
    pipe=scale_pipe,
    bins=None,
    invert="Auto",
):

    def binning(x):
        return binned(log(x), bins)

    training = (
        wds.WebDataset(urls)
        .shuffle(shuffle)
        .decode("l")
        .to_tuple(extensions)
        .map_tuple(lambda image: utils.autoinvert(image, invert))
        .pipe(pipe)
        .map_tuple(lambda x: x, binning)
        .shuffle(shuffle)
    )
    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


@public
class PageScale:
    def __init__(self, fname, check=True):
        self.model = loading.load_only_model(fname)
        self.bins = self.model.extra_["bins"]
        self.patchsize = self.model.extra_.get("patchsize", (200, 600))
        self.check = check
        self.debug = int(os.environ.get("DEBUG_PAGESCALE", 0))

    def getscale(self, page, npatches=200, bs=50):
        if self.check:
            assert np.mean(page) < 0.5
        try:
            self.model.cuda()
            self.model.eval()
            patches = scale_samples(page, alpha=(0, 0), patchsize=self.patchsize)
            result = []
            while True:
                batch = [x[0] for x in islice(patches, bs)]
                if len(batch) == 0:
                    break
                inputs = torch.tensor(batch).unsqueeze(1).cuda()
                with torch.no_grad():
                    outputs = self.model(inputs).softmax(1).cpu().detach()
                result.append(outputs)
                if self.debug:
                    for i in range(len(inputs)):
                        v = list((outputs[0].numpy() * 100).astype(int))
                        plt.ion()
                        plt.imshow(inputs[i, 0].detach().cpu().numpy())
                        plt.title(
                            str(i) + ": " + repr(inputs.shape) + " " + repr(v)
                        )
                        plt.ginput(1, 1000.0)
                    pass
            self.last = torch.cat(result).numpy()
            self.hist = self.last.mean(0)
            return exp(unbinned(int(self.hist.argmax()), self.bins))
        finally:
            self.model.cpu()

    def make_upright(self, page):
        angle = self.getscale(page)
        return ndi.rotate(page, -angle)


@app.command()
def train(
    urls: List[str],
    nsamples: int = 1000000,
    num_workers: int = 8,
    replicate: int = 1,
    bs: int = 16,
    prefix: str = "rot",
    lrfun="0.3**(3+n//5000000)",
    output: str = "",
    model: str = "page_scale_210301",
    extensions: str = default_extensions,
    alpha: float = 0.1,
    scale: str = "0.3, 3.0",
    nbins: int = 11,
    display: float = 0.0,
    invert: str = "Auto",
    patchsize: str = "200, 600",
):
    logger = slog.Logger(fname=output, prefix=prefix)
    logger.sysinfo()
    logger.json("args", sys.argv)
    patchsize = eval(f"({patchsize})")
    model = loading.load_or_construct_model(model, nbins, size=patchsize)
    model.cuda()
    print(model)
    urls = urls * replicate
    scale = eval(f"({scale})")
    logscale = (log(scale[0]), log(scale[1]))
    bins = np.linspace(logscale[0], logscale[1], nbins)
    training = make_loader(
        urls,
        shuffle=10000,
        num_workers=num_workers,
        batch_size=bs,
        extensions=extensions,
        invert=invert,
        bins=bins,
        pipe=partial(scale_pipe, logscale=logscale, patchsize=patchsize),
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
        loading.log_model(logger, model, step=count, loss=avgloss, bins=bins, patchsize=patchsize)
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
            pc = np.argmax(probs[0].numpy())
            plt.title(f"{targets[0]} {pc} {list((100*probs[0].numpy()).astype(int))}")
            plt.ginput(1, 0.001)
        if schedule("save", 15 * 60):
            save()
        if lrfun(count) != lr:
            lr = lrfun(count)
            optimizer = optim.SGD(model.parameters(), lr=lr)

    save()


@app.command()
def correct(
    urls: List[str],
    output: str = "/dev/null",
    model: str = "",
    extensions: str = default_extensions,
    limit: int = 999999999,
    invert: str = "Auto",
):
    scaletest = PageScale(model)
    dataset = wds.Dataset(urls).decode("l").to_tuple("__key__ " + extensions)
    sink = wds.TarWriter(output)
    for key, image in islice(dataset, limit):
        image = utils.autoinvert(image, invert)
        scale = scaletest.getscale(image)
        print(f"{key} {image.shape} {scale:.2f}")
        if output != "/dev/null":
            scaled = ndi.zoom(image, 1.0/scale, order=2).clip(0, 1)
            result = dict(__key__=key, jpg=scaled)
            sink.write(result)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
