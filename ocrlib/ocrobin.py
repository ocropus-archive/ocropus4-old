import random as pyrand
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ocrodeg
import typer
import webdataset as wds
import torch
from torch import nn, optim
from torch.utils import data
from itertools import islice
import scipy.ndimage as ndi
from math import exp, log, cos, sin

from . import slog
from . import loading
from . import utils


app = typer.Typer()

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()


def get_patch(image, shape, center, m=np.eye(2), order=1):
    assert np.amin(image) >= 0 and np.amax(image) <= 1.0
    yx = np.array(center, "f")
    hw = np.array(shape, "f")
    offset = yx - np.dot(m, hw / 2.0)
    return ndi.affine_transform(
        image, m, offset=offset, output_shape=shape, order=order
    ).clip(0, 1)


def bin_samples(
    page,
    binpage,
    npatches=32,
    ntrials=32,
    shape=(256, 1024),
    alpha=(-0.03, 0.03),
    scale=(1.0, 1.0),
    rotate=True,
):
    h, w = page.shape[:2]
    assert isinstance(page, np.ndarray), repr(page)[:200]
    assert isinstance(binpage, np.ndarray), repr(binpage)[:200]
    assert page.ndim == 2, page.shape
    assert binpage.ndim == 2, binpage.shape
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
        patch = get_patch(page, shape, (y, x), m=m, order=1)
        binpatch = get_patch(binpage, shape, (y, x), m=m, order=1)
        yield patch, binpatch


def bin_pipe(source, **kw):
    for (page, binpage) in source:
        yield from bin_samples(page, binpage, **kw)


def make_loader(
    urls,
    batch_size=16,
    extensions="nrm.jpg;image.png;framed.png;page.png;png;page.jpg;jpg;jpeg",
    shuffle=0,
    num_workers=4,
    pipe=bin_pipe,
    invert="Auto",
):

    def inverter(image):
        return utils.autoinvert(image, invert)

    def astensor(image):
        return torch.tensor(image).unsqueeze(0)

    training = (
        wds.WebDataset(urls)
        .shuffle(shuffle)
        .decode("l")
        .to_tuple(extensions)
        .map_tuple(inverter, inverter)
        .pipe(pipe)
        .shuffle(shuffle)
        .map_tuple(astensor, astensor)
    )
    return data.DataLoader(training, batch_size=batch_size, num_workers=num_workers)


def normalize(a):
    a = a - np.amin(a)
    a /= max(1e-6, np.amax(a))
    return a


class BinTrainer:
    def __init__(
        self,
        model,
        lr=1e-3,
        lr_schedule=None,
        savedir=True,
    ):
        super().__init__()
        self.model = model
        self.count = 0
        self.losses = []
        self.last_lr = -1
        self.set_lr(lr)
        self.lr_schedule = lr_schedule
        self.criterion = nn.MSELoss().cuda()
        self.every_batch = lambda _: None
        self.maxcount = 1e21
        self.nsamples = 0

    def to(self, device="cpu"):
        self.device = device
        self.model.to(device)
        self.criterion.to(device)
        # self.optimizer.to(device)

    def set_lr(self, lr, momentum=0.9):
        if lr == self.last_lr:
            return
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0
        )
        self.last_lr = lr

    def train_batch(self, inputs, targets):
        assert inputs.ndim == 4
        assert targets.ndim == 4
        assert len(inputs) == len(targets)
        if self.lr_schedule:
            self.set_lr(self.lr_schedule(self.nsamples))
        inputs = inputs.mean(1, keepdim=True)
        targets = targets.mean(1, keepdim=True)
        self.optimizer.zero_grad()
        self.model.train()
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(self.device))
        loss.backward()
        self.optimizer.step()
        self.count += len(inputs)
        self.losses.append(float(loss))
        self.last = (
            inputs.detach().cpu(),
            targets.detach().cpu(),
            outputs.detach().cpu(),
        )
        self.every_batch(self)
        self.nsamples += len(inputs)

    def show_batch(self):
        inputs, targets, outputs = self.last
        plt.ion()
        plt.clf()
        plt.subplot(311)
        plt.imshow(inputs[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.subplot(312)
        plt.imshow(targets[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.subplot(313)
        plt.imshow(outputs[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.ginput(1, 0.001)

    def predict_batch(self, inputs):
        pass


class Binarizer:
    def __init__(self, fname=None):
        self.model = loading.load_only_model(fname)

    def activate(self, yes=True):
        if yes:
            self.model.cuda()
        else:
            self.model.cpu()

    def binarize(self, image, nocheck=False, unzoom=True):
        self.activate(True)
        if image.ndim == 3:
            image = np.mean(image, 2)
        inputs = torch.tensor(image).unsqueeze(0).unsqueeze(0).cuda()
        outputs = self.model(inputs)[0, 0]
        result = np.array(outputs.detach().cpu().numpy(), dtype=np.float)
        return result


@app.command()
def generate(
    input: str,
    output: str = None,
    ngen: int = 1,
    extensions: str = "png;jpg;jpeg;page.png;page.jpg;image.png;image.jpg",
    limit: int = 999999999,
):
    """Given binary image training data, generate artificial binarization data using ocrodeg."""
    ds = wds.Dataset(input).decode("l").rename(__key__="__key__", image=extensions)
    sink = wds.TarWriter(output)
    for i, sample in enumerate(islice(ds, limit)):
        key = sample["__key__"]
        print(i, key)
        for v in range(ngen):
            page = normalize(sample["image"])
            if np.mean(page) < 0.5:
                page = 1.0 - page
            p = pyrand.uniform(0.0, 1.0)
            if p < 0.5:
                degraded = ocrodeg.printlike_multiscale(page, blotches=1e-6)
            else:
                degraded = ocrodeg.printlike_fibrous(page, blotches=1e-6)
            degraded = normalize(degraded)
            result = {"__key__": f"{key}/{v}", "jpg": degraded, "bin.jpg": page}
            sink.write(result)
    sink.close()


@app.command()
def train(
    fnames: List[str],
    extensions: str = "png;page.png;jpg;page.jpg bin.png",
    num_workers: int = 4,
    model: str = "binarization_210113",
    bs: int = 32,
    lr: float = 1e-3,
    show: int = 0,
    num_epochs: int = 100,
    output: str = "_ocrobin.sqlite3",
    replicate: int = 1,
    shuffle: int = 10000,
    display: float = 0.0,
    save_interval: float = 600.0,
    nsamples: int = 999999999999,
    invert: str = "Auto",
    tvals: str = "0.0, 1.0",
):
    fnames = fnames * replicate
    loader = make_loader(fnames, extensions=extensions, shuffle=shuffle, invert=invert, pipe=bin_pipe)
    logger = slog.Logger(fname=output, prefix="bin")
    model = loading.load_or_construct_model(model)
    model.cuda()
    print(model)
    trainer = BinTrainer(model, lr=lr)
    trainer.count = int(getattr(model, "step_", 0))
    trainer.to("cuda")

    tvals = eval(f"({tvals})")

    def save():
        logger.save_smodel(
            model, step=trainer.count, scalar=np.mean(trainer.losses[-100:])
        )
        logger.flush()

    schedule = utils.Schedule()

    for inputs, targets in islice(utils.repeatedly(loader), 0, nsamples):
        targets *= tvals[1] - tvals[0]
        targets += tvals[0]
        trainer.train_batch(inputs, targets)
        save()
        if schedule("progress", 60, initial=True):
            print(f"epoch {trainer.count} loss {np.mean(trainer.losses[-500:])}")
        if schedule("save", save_interval, initial=True):
            save()
        if display > 0 and schedule("display", display, initial=True):
            trainer.show_batch()

    save()


@app.command()
def binarize(
    fname: str,
    output: str = None,
    model: str = None,
    iext: str = "png",
    oext: str = "bin.png",
    keep: bool = True,
    show: int = 0,
    limit: int = 99999999,
):
    src = wds.Dataset(fname).decode("rgb")
    binarizer = Binarizer(model)
    with wds.TarWriter(output) as sink:
        for index, sample in enumerate(islice(src, limit)):
            print(f"{sample['__key__']}")
            image = sample.get(iext)
            result = binarizer.binarize(image)
            if not keep:
                del sample[iext]
            sample[oext] = result
            sink.write(sample)
            if show > 0 and index % show == 0:
                plt.ion()
                plt.clf()
                plt.subplot(121)
                plt.imshow(image)
                plt.subplot(122)
                plt.imshow(result)
                plt.ginput(1, 0.001)
            if show > 0:
                plt.ginput(1, 0.001)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
