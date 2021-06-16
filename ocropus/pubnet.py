import os
import sys
import time
import signal

import typer
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import amin, median, mean
from scipy import ndimage as ndi
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from webdataset import Dataset
import webdataset as wds
import torchmore.layers
import skimage
import skimage.filters
from itertools import islice

from .utils import Schedule, repeatedly
from . import slog
from . import utils
from . import loading
from . import patches
from . import slices as sl
from . import ocroseg
from .utils import useopt, junk


logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


app = typer.Typer()


def done_exn(*args, **kw):
    raise Exception("done")

def enable_kill():
    def handler(signal_received, frame):
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)

def allboxes(a):
    return ndi.find_objects(ndi.label(a)[0])

def removed(l, x):
    return [y for y in l if y != x]


def intersects_any(x, bg):
    for y in bg:
        if sl.intersections(x, y) is not None:
            return True
    return False


def mergeall(fg, bg):
    fg = fg.copy()
    result = []
    while len(fg) > 0:
        a = fg.pop(0)
        i = 0
        while i < len(fg):
            b = fg[i]
            u = sl.unions(a, b)
            if intersects_any(u, bg):
                i = i + 1
            else:
                a = u
                del fg[i]
        result.append(a)
    return result


def opening(a, shape, shape2=None):
    result = ndi.maximum_filter(ndi.minimum_filter(a, shape), shape)
    if shape2 is not None:
        result2 = ndi.maximum_filter(ndi.minimum_filter(a, shape2), shape2)
        result = nd.maximum(result, result2)
    return result


class PubLaynetSegmenter:
    def __init__(self, model):
        if isinstance(model, str):
            model = loading.load_or_construct_model(model)
            model.eval()
        assert callable(model)
        self.model = model
        self.hystthresh = (0.5, 0.9)
        self.text_opening = [(2, 20)]
        self.images_opening = [(3, 3)]
        self.tables_opening = [(3, 3)]
        self.offset = (-2, -2)

    def activate(self, active=True):
        if active:
            self.model.cuda()
        else:
            self.model.cpu()

    def predict_probs(self, im, check=True):
        if check:
            assert im.shape[0] > 500 and im.shape[0] < 1200, im.shape
            assert im.shape[1] > 300 and im.shape[1] < 1000, im.shape
            assert np.mean(im) > 0.5
        input = 1 - torch.tensor(im).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input).detach().cpu()[0].softmax(0).numpy().transpose(1, 2, 0)
        assert output.shape[2] == 5
        return output

    def predict(self, im, merge=True, check=True):
        output = self.predict_probs(im, check=check)
        self.last_probs = output
        lo, hi = self.hystthresh

        # use hysteresis thresholding for all the major regions
        tables = skimage.filters.apply_hysteresis_threshold(output[..., 1], lo, hi)
        images = skimage.filters.apply_hysteresis_threshold(output[..., 2], lo, hi)
        markers = skimage.filters.apply_hysteresis_threshold(output[..., 4], lo, hi)
        regions = skimage.filters.apply_hysteresis_threshold(
            np.maximum(output[..., 3], output[..., 4]), 0.3, 0.9
        )

        # perform additional cleanup
        tables = opening(tables, *self.tables_opening)
        images = opening(images, *self.images_opening)

        # compute boxes
        table_boxes = allboxes(tables)
        image_boxes = allboxes(images)

        # text regions are obtained through marker segmentation
        text = ocroseg.marker_segmentation(markers, regions, maxdist=20)  # this is labels
        text = opening(text, *self.text_opening)  # nb: morphology on labels is OK
        text_boxes = ndi.find_objects(text)

        if merge:
            merged_table_boxes = mergeall(table_boxes, text_boxes + image_boxes)
            merged_image_boxes = mergeall(image_boxes, text_boxes + table_boxes)
        else:
            merged_table_boxes = table_boxes
            merged_image_boxes = image_boxes

        return self.fix(text_boxes), self.fix(merged_table_boxes), self.fix(merged_image_boxes)


    def fix(self, boxes):
        if boxes is None:
            return []
        result = []
        dy, dx = self.offset
        for ys, xs in removed(boxes, None):
            xs = slice(xs.start+dx, xs.stop+dx)
            ys = slice(ys.start+dy, ys.stop+dy)
            result.append((ys, xs))
        return result

    def predict_map(self, im, **kw):
        textobj, tableobj, imgobj = self.predict(im, **kw)
        z = np.zeros(im.shape[:2], dtype=int)
        for s in textobj:
            if s is None:
                continue
            z[tuple(s)] = 3
        for s in tableobj:
            if s is None:
                continue
            z[tuple(s)] = 1
        for s in imgobj:
            if s is None:
                continue
            z[tuple(s)] = 2
        return z


def rescale(im, scale, target=(800, 800)):
    if scale is None or scale == 1:
        return im
    if isinstance(scale, (int, float)):
        im = ndi.zoom(im, [scale, scale, 1][: im.ndim], order=1)
        return im
    if isinstance(scale, tuple):
        scale = np.amin(np.array(target, dtype=float) / np.array(scale))
        scale = min(scale, 1.0)
        im = ndi.zoom(im, [scale, scale, 1][: im.ndim], order=1)
        return im


@app.command()
def pageseg(
    src: str,
    model: str = "publaynet-model.pth",
    scale: str = "(800, 800)",
    nomerge: bool = False,
    probs: bool = False,
    slice: str = "999999999",
    timeout: float = 1e9,
    check: bool = True,
    offset: str = "-2, -2",
):
    scale = eval(scale)
    segmenter = PubLaynetSegmenter(model)
    segmenter.offset = eval(f"({offset})")
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").to_tuple("__key__", "png;jpg;jpeg")
    slicer = eval(f"lambda x: islice(x, {slice})")
    plt.ion()
    plt.gcf().canvas.mpl_connect('close_event', done_exn)
    for count, (key, im) in slicer(enumerate(ds)):
        im = rescale(im, scale, target=(800, 800))
        text, tables, images = segmenter.predict(im)

        if True:
            from matplotlib.patches import Rectangle

            plt.clf()
            plt.subplot(122)
            z = segmenter.last_probs
            assert z.shape[2] == 5
            z[..., 3] = np.maximum(z[..., 3], z[..., 4])
            plt.imshow(z[..., 1:4])
            plt.subplot(121)
            plt.title(f"{count}: {key}")
            ax = plt.gca()
            plt.imshow(im)
            for (boxes, color) in zip([text, tables, images], ["blue", "red", "green"]):
                for ys, xs in boxes:
                    w, h = xs.stop - xs.start, ys.stop - ys.start
                    ax.add_patch(Rectangle((xs.start, ys.start), w, h, color=color, alpha=0.4))
            enable_kill()
            plt.ginput(1, timeout)


class PubTabnetSegmenter:
    def __init__(self, model):
        if isinstance(model, str):
            model = loading.load_or_construct_model(model)
            model.eval()
        assert callable(model)
        self.model = model
        self.hystthresh = (0.5, 0.9)
        self.opening = []
        self.do_interpolate = True
        self.offset = (-2, -2)

    def activate(self, active=True):
        if active:
            self.model.cuda()
        else:
            self.model.cpu()

    def predict_probs(self, im, check=True):
        h, w, d = im.shape
        input = 1 - torch.tensor(im).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input)
            if output.shape[3] != h or output.shape[4] != w:
                if self.do_interpolate:
                    output = F.interpolate(output, size=(h, w))
                else:
                    output = output[:, :, :h, :w]
            output = output.detach().cpu()[0].softmax(0).numpy().transpose(1, 2, 0)
        assert output.shape[2] == 5
        return output

    def predict(self, im, merge=True, check=True):
        output = self.predict_probs(im, check=check)
        self.last_probs = output
        lo, hi = self.hystthresh

        # use hysteresis thresholding for all the major regions
        markers = skimage.filters.apply_hysteresis_threshold(output[..., 1], lo, hi)
        regions = np.maximum(output[..., 1], output[..., 2])
        regions = skimage.filters.apply_hysteresis_threshold(regions, lo, hi)

        if len(self.opening) > 0:
            markers = opening(markers, *self.opening)
            regions = opening(regions, *self.opening)

        # text regions are obtained through marker segmentation
        text = ocroseg.marker_segmentation(markers, regions, maxdist=20)  # this is labels
        text_boxes = ndi.find_objects(text)

        return self.fix(text_boxes)

    def predict_map(self, im, **kw):
        textobj = self.predict(im, **kw)
        z = np.zeros(im.shape[:2], dtype=int)
        for s in textobj:
            if s is None:
                continue
            z[tuple(s)] = 3
        return z

    def fix(self, boxes):
        if boxes is None:
            return []
        result = []
        dy, dx = self.offset
        for ys, xs in removed(boxes, None):
            xs = slice(xs.start+dx, xs.stop+dx)
            ys = slice(ys.start+dy, ys.stop+dy)
            result.append((ys, xs))
        return result



@app.command()
def tabseg(
    src: str,
    model: str = "pubtabnet-model.pth",
    scale: str = "1.0",
    nomerge: bool = False,
    probs: bool = False,
    sliced: str = "999999999",
    timeout: float = 1e9,
    offset: str = "-2,-2",
    check: bool = True,
    verbose: bool = False,
):
    scale = eval(scale)
    segmenter = PubTabnetSegmenter(model)
    segmenter.offset = eval(f"({offset})")
    if verbose:
        print(segmenter.model)
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").to_tuple("__key__", "png;jpg;jpeg")
    slicer = eval(f"lambda x: islice(x, {sliced})")
    plt.ion()
    plt.gcf().canvas.mpl_connect('close_event', done_exn)
    for count, (key, im) in slicer(enumerate(ds)):
        im = rescale(im, scale, target=(800, 800))
        boxes = segmenter.predict(im)
        if True:
            plt.clf()
            plt.subplot(122)
            plt.title(f"{count}: {key}")
            z = segmenter.last_probs
            assert z.shape[2] == 5
            z[..., 3] = np.maximum(z[..., 3], z[..., 4])
            plt.imshow(im * 0.05 + z[..., 1:4] * 0.95)
            plt.subplot(121)
            from matplotlib.patches import Rectangle
            ax = plt.gca()
            plt.imshow(im)
            for ys, xs in boxes:
                w, h = xs.stop - xs.start, ys.stop - ys.start
                ax.add_patch(Rectangle((xs.start, ys.start), w, h, color="red", alpha=0.2))
            enable_kill()
            plt.ginput(1, timeout)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
