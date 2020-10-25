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
from torchmore import layers

from . import slog


app = typer.Typer()

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()


def normalize(a):
    a = a - np.amin(a)
    a /= max(1e-6, np.amax(a))
    return a


def make_model_lstm():
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


model = make_model_lstm()
model.cuda()


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


def train_epoch(model, src, lr, show=0):
    model.count = getattr(model, "count", 0)
    model.losses = getattr(model, "losses", [])
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.MSELoss().cuda()
    index = 0
    for inputs, targets in src:
        inputs = inputs.mean(1, keepdim=True)
        targets = targets.mean(1, keepdim=True)
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        model.count += len(inputs)
        model.losses.append(float(loss))
        print(
            f"{model.count:8d} {np.mean(model.losses[-50:]):.3f}", end="\r", flush=True
        )
        if show > 0 and index % show == 0:
            plt.ion()
            plt.clf()
            plt.subplot(121)
            plt.imshow(inputs[0, 0].detach().cpu())
            plt.subplot(122)
            plt.imshow(outputs[0, 0].detach().cpu())
            plt.ginput(1, 0.001)
        index += 1


@app.command()
def generate(
    input: str,
    output: str = None,
    ngen: int = 1,
    extensions: str = "png;jpg;jpeg;page.png;page.jpg;image.png;image.jpg",
):
    """Given binary image training data, generate artificial binarization data using ocrogen."""
    ds = wds.Dataset(input).decode("l").rename(__key__="__key__", image=extensions)
    sink = wds.TarWriter(output)
    for i, sample in enumerate(ds):
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
    inputs: str = "png",
    outputs: str = "bin.png",
    num_workers: int = 4,
    bs: int = 32,
    lr: float = 1e-4,
    show: int = 0,
    num_epochs: int = 100,
):
    ds = (
        wds.Dataset(
            fnames, handler=wds.warn_and_continue, tarhandler=wds.warn_and_continue
        )
        .decode("torchrgb")
        .to_tuple(inputs, outputs)
        .pipe(tiles)
        .shuffle(5000)
    )
    dl = data.DataLoader(ds, num_workers=num_workers, batch_size=bs)
    logger = slog.Logger(prefix="bin")
    model = make_model_lstm()
    model.cuda()
    for epoch in range(num_epochs):
        train_epoch(model, dl, lr=lr, show=show)
        print(f"\nepoch {epoch} loss {np.mean(model.losses[-500:])}")
        obj = dict(mstate=model.state_dict())
        logger.save_model(obj, step=model.count, scalar=np.mean(model.losses[-100:]))


@app.command()
def binarize(
    fname: str,
    output: str = None,
    model: str = None,
    iext: str = "png",
    oext: str = "bin.png",
    keep: bool = True,
    show: int = 0,
):
    src = wds.Dataset(fname).decode("rgb")
    with open(model, "rb") as stream:
        obj = torch.load(stream)
    model = make_model_lstm()
    model.load_state_dict(obj["mstate"])
    model.cuda()
    model.eval()
    with wds.TarWriter(output) as sink:
        for index, sample in enumerate(src):
            print(f"{sample['__key__']}")
            image = sample.get(iext)
            inputs = torch.tensor(np.mean(image, 2)).unsqueeze(0).unsqueeze(0).cuda()
            outputs = model(inputs)[0, 0]
            result = outputs.detach().cpu().numpy() + 0.0
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
