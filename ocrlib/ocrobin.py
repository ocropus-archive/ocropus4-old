import random as pyrand
from typing import List
import sys

import matplotlib.pyplot as plt
import numpy as np
import ocrodeg
import typer
import webdataset as wds
import torch
from torch import nn, optim
from torch.utils import data
from itertools import islice

from . import slog
from . import loading


app = typer.Typer()

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()


def normalize(a):
    a = a - np.amin(a)
    a /= max(1e-6, np.amax(a))
    return a


default_model = """
from torch import nn
from torchmore import layers

def make_model():
    r = 3
    model = nn.Sequential(
        nn.Conv2d(1, 8, r, padding=r // 2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        layers.BDHW_LSTM(8, 4),
        nn.Conv2d(8, 1, 1),
        nn.Sigmoid(),
    )
    return model
"""


def tiles(src, r=256):
    for inputs, targets in src:
        assert inputs.shape == targets.shape
        d, h, w = inputs.shape
        for i in range(0, h - r + 1, r):
            for j in range(0, w - r + 1, r):
                yield (
                    inputs[:, i : i + r, j : j + r],
                    targets[:, i : i + r, j : j + r],
                )


def nothing(trainer):
    pass


class BinTrainer:
    def __init__(
        self,
        model,
        lr=1e-3,
        savedir=True,
    ):
        super().__init__()
        self.model = model
        self.count = 0
        self.losses = []
        self.last_lr = -1
        self.set_lr(lr)
        self.criterion = nn.MSELoss().cuda()
        self.every_batch = nothing
        self.maxcount = 1e21

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
        inputs = inputs.mean(1, keepdim=True)
        targets = targets.mean(1, keepdim=True)
        self.optimizer.zero_grad()
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

    def train_epoch(self, loader, show=-1):
        count = 0
        for inputs, targets in loader:
            self.train_batch(inputs, targets)
            print(
                f"{self.count:10d} {np.mean(self.losses[-10:]):.5f}",
                end="\r",
                file=sys.stderr,
            )
            if show > 0 and count % show == 0:
                self.show_batch()
            count += 1
            if self.count >= self.maxcount:
                return

    def show_batch(self):
        inputs, targets, outputs = self.last
        plt.ion()
        plt.clf()
        plt.subplot(121)
        plt.imshow(inputs[0, 0].detach().cpu())
        plt.subplot(122)
        plt.imshow(outputs[0, 0].detach().cpu())
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
            result = {"__key__": f"{key}/{v}", "png": degraded, "bin.png": page}
            sink.write(result)
    sink.close()


@app.command()
def train(
    fnames: List[str],
    extensions: str = "png;page.png;jpg;page.jpg bin.png",
    num_workers: int = 4,
    model: str = "binarization_210113",
    bs: int = 32,
    lr: float = 1e-4,
    show: int = 0,
    num_epochs: int = 100,
    maxcount: float = 1e21,
    output: str = "",
    replicate: int = 1,
):
    fnames = fnames * replicate
    ds = (
        wds.WebDataset(fnames, handler=wds.warn_and_continue)
        .decode("torchrgb")
        .to_tuple(extensions)
        .pipe(tiles)
        .shuffle(5000)
    )
    dl = data.DataLoader(ds, num_workers=num_workers, batch_size=bs)
    logger = slog.Logger(fname=output, prefix="bin")
    model = loading.load_or_construct_model(model)
    model.cuda()
    trainer = BinTrainer(model, lr=lr)
    trainer.count = int(getattr(model, "step_", 0))
    trainer.to("cuda")
    trainer.maxcount = maxcount

    def save():
        logger.save_smodel(
            model, step=trainer.count, scalar=np.mean(trainer.losses[-100:])
        )
        logger.flush()

    for epoch in range(num_epochs):
        trainer.train_epoch(dl, show=show)
        print(f"\nepoch {epoch} loss {np.mean(trainer.losses[-500:])}")
        # obj = dict(mstate=model.state_dict())
        save()
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
