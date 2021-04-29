import os
import sys
import time

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


def allboxes(a):
    return ndi.find_objects(ndi.label(a)[0])


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

        return text_boxes, merged_table_boxes, merged_image_boxes

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


@app.command()
def pageseg(
    src: str,
    model: str = "publaynet-model.pth",
    scale=1.0,
    nomerge: bool = False,
    probs: bool = False,
    slice: str = "999999999",
    timeout: float = 1e9,
    check: bool = True,
):
    segmenter = PubLaynetSegmenter(model)
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").to_tuple("__key__", "png;jpg;jpeg")
    slicer = eval(f"lambda x: islice(x, {slice})")
    for count, (key, im) in slicer(enumerate(ds)):
        if scale != 1.0:
            im = ndi.zoom(im, [scale, scale, 1][: im.ndim], order=1)
        plt.clf()
        plt.subplot(121)
        plt.title(f"{count}: {key}")
        plt.imshow(im)
        if probs:
            z = segmenter.predict_probs(im)
            assert z.shape[2] == 5
            z[..., 3] = np.maximum(z[..., 3], z[..., 4])
            plt.subplot(122)
            plt.imshow(z[..., 1:4])
        else:
            z = segmenter.predict_map(im, merge=(not nomerge), check=check)
            plt.subplot(122)
            plt.imshow(z, cmap=plt.cm.viridis, vmax=3)
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

        return text_boxes

    def predict_map(self, im, **kw):
        textobj = self.predict(im, **kw)
        z = np.zeros(im.shape[:2], dtype=int)
        for s in textobj:
            if s is None:
                continue
            z[tuple(s)] = 3
        return z


@app.command()
def tabseg(
    src: str,
    model: str = "pubtabnet-model.pth",
    scale=1.0,
    nomerge: bool = False,
    probs: bool = False,
    slice: str = "999999999",
    timeout: float = 1e9,
    check: bool = True,
    verbose: bool = False,
):
    segmenter = PubTabnetSegmenter(model)
    if verbose:
        print(segmenter.model)
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").to_tuple("__key__", "png;jpg;jpeg")
    slicer = eval(f"lambda x: islice(x, {slice})")
    for count, (key, im) in slicer(enumerate(ds)):
        if scale != 1.0:
            im = ndi.zoom(im, [scale, scale, 1], order=1)
        plt.clf()
        if probs:
            plt.title(f"{count}: {key}")
            z = segmenter.predict_probs(im)
            assert z.shape[2] == 5
            z[..., 3] = np.maximum(z[..., 3], z[..., 4])
            plt.imshow(im * 0.5 + z[..., 1:4] * 0.5)
        else:
            from matplotlib.patches import Rectangle

            fig, ax = plt.subplots()
            boxes = segmenter.predict(im)
            plt.imshow(im)
            for ys, xs in boxes:
                w, h = xs.stop - xs.start, ys.stop - ys.start
                ax.add_patch(Rectangle((xs.start, ys.start - h), w, h, color="red", alpha=0.2))
        plt.ginput(1, timeout)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
