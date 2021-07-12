#!/usr/bin/env python

import sys

import argparse

import PIL
import numpy as np
import typer
import matplotlib.pyplot as plt
from itertools import islice
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import webdataset as wds
from webdataset.iterators import getfirst
from . import utils


app = typer.Typer()


def isimage(a):
    return isinstance(a, np.ndarray) and (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in [3, 4]))


@app.command()
def show(fname: str, labels: str="txt;gt.txt"):
    ds = wds.WebDataset(fname).decode("rgb")
    labels = labels.split() if labels != "" else []
    plt.ion()
    fig = plt.gcf()
    fig.canvas.mpl_connect('close_event', lambda _: sys.exit(0))
    for sample in ds:
        images = [(k, v) for k, v in sample.items() if isimage(v)]
        images = sorted(images)
        plt.clf()
        for i, (k, v) in enumerate(images):
            plt.subplot(1, len(images), i+1)
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
