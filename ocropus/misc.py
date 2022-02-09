#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
import typer
import webdataset as wds
import os, fnmatch


app = typer.Typer()


def isimage(a):
    return isinstance(a, np.ndarray) and (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in [3, 4]))


def find_checkpoints(root, pattern="*.ckpt"):
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


@app.command()
def ckpt(fname: str, model: bool = False):
    import os.path
    import torch

    if os.path.isdir(fname):
        files = sorted(list(find_checkpoints(fname)))
    else:
        files = [fname]
    for f in files:
        print(f"\n=== {f}:\n")
        data = torch.load(open(f, "rb"))
        for k, v in data.items():
            s = repr(v)[:60]
            print("%-30s %s" % (k, s))
        print()
        for k, v in data.get("hyper_parameters", {}).items():
            s = repr(v)[:60]
            print("%-30s %s" % (k, s))


@app.command()
def showjit(fname: str):
    import torch.jit

    data = torch.jit.load(open(fname, "rb"))
    print(data)


@app.command()
def show(fname: str, labels: str = "txt;gt.txt"):
    ds = wds.WebDataset(fname).decode("rgb")
    labels = labels.split() if labels != "" else []
    plt.ion()
    fig = plt.gcf()
    fig.canvas.mpl_connect("close_event", lambda _: sys.exit(0))
    for sample in ds:
        images = [(k, v) for k, v in sample.items() if isimage(v)]
        images = sorted(images)
        plt.clf()
        for i, (k, v) in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            v = v - np.amin(v)
            v /= max(1e-3, np.amax(v))
            plt.imshow(v)
            s = k
            if i < len(labels):
                s += " : "
                s += wds.getfirst(sample, labels[i], None, missing_is_error=False)
            plt.title(s)
        for k, v in sorted(sample.items()):
            value = str(v)[:60].replace("\n", "\\n")
            print(f"{k:20s} {value}")
        plt.ginput(1, 10000)
    sink.close()


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
