import os
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.filters
import torch
import torch.nn.functional as F
import typer
import webdataset as wds
from scipy import ndimage as ndi
from itertools import islice

from . import loading, ocroseg
from . import slices as sl
from . import ocroseg
from .utils import useopt, junk
from matplotlib.patches import Rectangle
from . import slog


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


def large_rect(r):
    if r[0].stop - r[0].start < 20:
        return False
    if r[1].stop - r[1].start < 20:
        return False
    return True


def merge_against_background(fg, bg):
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
        result = np.maximum(result, result2)
    return result


def closing(a, shape, shape2=None):
    result = ndi.minimum_filter(ndi.maximum_filter(a, shape), shape)
    if shape2 is not None:
        result2 = ndi.minimum_filter(ndi.maximum_filter(a, shape2), shape2)
        result = nd.maximum(result, result2)
    return result


def simple_binarize(im):
    im = im.astype(float)
    if im.ndim == 3:
        im = np.mean(im, axis=2)
    im = im - np.amin(im)
    im /= np.amax(im)
    if np.mean(im) > 0.5:
        im = 1.0 - im
    threshold = np.mean([np.median(im), np.mean(im)])
    return (im > threshold).astype(int)


def makergb(r, g, b=None):
    b = b if b is not None else g
    dimage = np.array([r, g, b]).transpose(1, 2, 0).astype(float)
    dimage /= np.amax(dimage)
    return dimage


def covering_rectangle(rect, bin, exclude=None, prepad=0, postpad=5, debug=0):
    assert bin.dtype in [bool, np.uint8, int]
    assert np.amin(bin) >= 0 and np.amax(bin) <= 1
    center = lambda x: int(np.mean([x.stop, x.start]))
    center_y, center_x = center(rect[0]), center(rect[1])
    bin = bin.astype(int)
    if debug:
        plt.clf()
        plt.subplot(231)
        plt.imshow(bin)
    mask = np.zeros_like(bin)
    mask[rect[0], rect[1]] = 1
    assert mask[center_y, center_x] == 1
    mask = ndi.maximum_filter(mask, prepad)
    if exclude is not None:
        mask = np.minimum(mask, exclude)
        if mask[center_y, center_x] != 1:
            # excluded the center, so we'll just return nothing
            return None
    if debug:
        plt.subplot(232)
        plt.imshow(mask)
    mbin = np.maximum(bin, mask) > 0
    assert mbin[center_y, center_x] == 1
    components, n = ndi.label(mbin)
    if debug:
        plt.subplot(233)
        plt.imshow(np.sin(components * 17.3), cmap=plt.cm.viridis)
    the_component = components[center_y, center_x]
    assert the_component != 0
    cmask = (components == the_component).astype(int)
    if debug:
        plt.subplot(234)
        plt.imshow(cmask)
    if debug:
        plt.subplot(235)
        plt.imshow(makergb(cmask, exclude, mbin))
    if exclude is not None:
        cmask = np.minimum(cmask, exclude)
    cmask = ndi.maximum_filter(cmask, postpad)
    cmask = np.clip(cmask, 0, 1)
    objects = ndi.find_objects(cmask)
    rrect = objects[0]
    if debug:
        plt.subplot(236)
        dresult = np.zeros_like(bin)
        dresult[rrect[0], rrect[1]] = 1
        dimage = np.array([cmask, bin, dresult]).transpose(1, 2, 0).astype(float)
        plt.imshow(dimage)
        plt.ginput(1, debug)
    assert len(objects) == 1
    return objects[0]


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

    def predict_probs(self, im, scale=1.0, check=True):
        h, w, _ = im.shape
        if scale != 1.0:
            im = ndi.zoom(im, (scale, scale, 1.0), order=1)
        if check:
            assert im.shape[0] > 500 and im.shape[0] < 1200, im.shape
            assert im.shape[1] > 300 and im.shape[1] < 1000, im.shape
            assert np.mean(im) > 0.5
        with torch.no_grad():
            input = 1 - torch.tensor(im).permute(2, 0, 1).unsqueeze(0)
            output = self.model(input)
            assert output.ndim == 4
            if output.shape[2] != h or output.shape[3] != w:
                output = F.interpolate(output, size=(h, w))
            output = output.detach().cpu()[0].softmax(0).numpy().transpose(1, 2, 0)
        assert output.shape == (h, w, 5)
        return output

    def predict(self, im, scale=1.0, merge=True, check=True, nocover=False):
        output = self.predict_probs(im, scale=1.0, check=check)
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
            merged_table_boxes = merge_against_background(table_boxes, text_boxes + image_boxes)
            merged_image_boxes = merge_against_background(image_boxes, text_boxes + table_boxes)
        else:
            merged_table_boxes = table_boxes
            merged_image_boxes = image_boxes

        text, tables, images = (
            self.fix(text_boxes),
            self.fix(merged_table_boxes),
            self.fix(merged_image_boxes),
        )

        # FIXME: add code for self-overlapping table/image boxes

        if not nocover:
            bin = closing(simple_binarize(im), 5)
            exclude = np.ones_like(bin, dtype=int)
            for b in text:
                exclude[b[0], b[1]] = 0
            exclude = ndi.minimum_filter(exclude, (5, 20))
            tables = [covering_rectangle(b, bin, exclude) for b in tables if large_rect(b)]
            tables = [x for x in tables if x is not None]
            images = [covering_rectangle(b, bin, exclude) for b in images if large_rect(b)]
            images = [x for x in images if x is not None]

        self.last_result = result = text, tables, images
        return result

    def fix(self, boxes):
        if boxes is None:
            return []
        result = []
        dy, dx = self.offset
        for ys, xs in removed(boxes, None):
            xs = slice(xs.start + dx, xs.stop + dx)
            ys = slice(ys.start + dy, ys.stop + dy)
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


def showtypes(d):
    for k, v in d.items():
        print("#", k, type(v), getattr(v, "dtype", None))


def simplify(x):
    if isinstance(x, (int, float, bool, type(None))):
        return x
    if isinstance(x, (tuple, list)):
        return [simplify(y) for y in x]
    if isinstance(x, dict):
        return {str(key): simplify(value) for key, value in x.items()}
    if isinstance(x, slice):
        result = dict(start=simplify(x.start), stop=simplify(x.stop))
        if x.step is not None:
            result["step"] = x.step
        return result
    raise ValueError(f"{type(x)}: unknown type, value: {str(x)[:100]}")


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


def pageseg_display(im, segmenter, result=None, title="", timeout=10.0):
    if timeout < 0:
        return
    text, tables, images = result or segmenter.last_result
    plt.clf()
    plt.subplot(122)
    z = segmenter.last_probs
    assert z.shape[2] == 5
    z[..., 3] = np.maximum(z[..., 3], z[..., 4])
    plt.imshow(z[..., 1:4])
    plt.subplot(121)
    plt.title(title)
    ax = plt.gca()
    plt.imshow(im)
    for (boxes, color) in zip([text, tables, images], ["blue", "red", "green"]):
        for ys, xs in boxes:
            w, h = xs.stop - xs.start, ys.stop - ys.start
            ax.add_patch(Rectangle((xs.start, ys.start), w, h, color=color, alpha=0.4))
    enable_kill()
    plt.ginput(1, timeout)


@app.command()
def pageseg(
    src: str,
    model: str = "publaynet-model.pth",
    scale: float = 1.0,
    nomerge: bool = False,
    probs: bool = False,
    slice: str = "999999999",
    timeout: float = 1e9,
    check: bool = True,
    offset: str = "-2, -2",
    output: str = "",
    display: float = -1,
    outputs: str = "PFT",
    nocover: bool = False,
):
    segmenter = PubLaynetSegmenter(model)
    segmenter.offset = eval(f"({offset})")
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").rename(jpg="png;jpg;jpeg")
    sink = None
    if output != "":
        assert not os.path.exists(output)
        sink = wds.TarWriter(output)
    slicer = eval(f"lambda x: islice(x, {slice})")
    plt.ion()
    plt.gcf().canvas.mpl_connect("close_event", done_exn)
    for count, sample in slicer(enumerate(ds)):
        key, im = sample["__key__"], sample["jpg"]
        print(key)
        text, tables, images = segmenter.predict(im, scale, nocover=nocover)
        if display >= 0:
            pageseg_display(
                im, segmenter, result=(text, tables, images), title=f"{count}: {key}", timeout=display
            )
        seg = segmenter.last_probs.copy()
        assert seg.shape[2] == 5
        seg[..., 3] = np.maximum(seg[..., 3], seg[..., 4])
        seg = seg[..., 1:4]
        result = dict(sample)
        add = {
            "probs.jpg": seg,
            "text.json": simplify(text),
            "tables.json": simplify(tables),
            "images.json": simplify(images),
        }
        result.update(add)
        if sink is not None and "p" in outputs.lower():
            sink.write(result)
        if sink is not None and "f" in outputs.lower():
            for index, bounds in enumerate(images):
                image = im[bounds[0], bounds[1], ...]
                result = dict(__key__=key + f"/fig{index}", jpg=image)
                sink.write(result)
                print(result["__key__"])
        if sink is not None and "t" in outputs.lower():
            for index, bounds in enumerate(tables):
                image = im[bounds[0], bounds[1], ...]
                result = dict(__key__=key + f"/tab{index}", jpg=image)
                sink.write(result)
                print(result["__key__"])


class PubTabnetSegmenter:
    def __init__(self, model):
        if isinstance(model, str):
            model = loading.load_or_construct_model(model)
            model.eval()
        assert callable(model)
        self.model = model
        self.hystthresh = (0.5, 0.9)
        self.opening = []
        self.offset = (-2, -2)

    def activate(self, active=True):
        if active:
            self.model.cuda()
        else:
            self.model.cpu()

    def predict_probs(self, im, scale=1.0, check=True):
        h, w, d = im.shape
        if scale != 1.0:
            im = ndi.zoom(im, (scale, scale, 1.0), order=1)
        with torch.no_grad():
            input = 1 - torch.tensor(im).permute(2, 0, 1).unsqueeze(0)
            output = self.model(input)
            assert output.ndim == 4
            if output.shape[2] != h or output.shape[3] != w:
                output = F.interpolate(output, size=(h, w))
            output = output.detach().cpu()[0].softmax(0).numpy().transpose(1, 2, 0)
        assert output.shape == (h, w, 5)
        return output

    def predict(self, im, scale=1.0, merge=True, check=True):
        output = self.predict_probs(im, check=check, scale=scale)
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

        result = self.fix(text_boxes)
        self.last_result = result
        return result

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
            xs = slice(xs.start + dx, xs.stop + dx)
            ys = slice(ys.start + dy, ys.stop + dy)
            result.append((ys, xs))
        return result


def tabseg_display(im, segmenter, title="", timeout=10.0):
    if timeout < 0:
        return
    boxes = segmenter.last_result
    plt.clf()
    plt.subplot(122)
    plt.title(title)
    z = segmenter.last_probs
    assert z.shape[2] == 5
    z[..., 3] = np.maximum(z[..., 3], z[..., 4])
    plt.imshow(im * 0.05 + z[..., 1:4] * 0.95)
    plt.subplot(121)
    ax = plt.gca()
    plt.imshow(im)
    for ys, xs in boxes:
        w, h = xs.stop - xs.start, ys.stop - ys.start
        ax.add_patch(Rectangle((xs.start, ys.start), w, h, color="red", alpha=0.2))
    enable_kill()
    plt.ginput(1, timeout)


@app.command()
def tabseg(
    src: str,
    model: str = "pubtabnet-model.pth",
    scale: float = 1.0,
    nomerge: bool = False,
    probs: bool = False,
    sliced: str = "999999999",
    timeout: float = -1,
    offset: str = "-2,-2",
    check: bool = True,
    verbose: bool = False,
    select: str = "",
    output: str = "",
):
    segmenter = PubTabnetSegmenter(model)
    segmenter.offset = eval(f"({offset})")
    if verbose:
        print(segmenter.model)
    segmenter.activate()
    ds = wds.WebDataset(src).decode("rgb").rename(jpg="png;jpg;jpeg")
    slicer = eval(f"lambda x: islice(x, {sliced})")
    sink = None if output == "" else wds.TarWriter(output)
    if timeout > 0:
        plt.ion()
        plt.gcf().canvas.mpl_connect("close_event", done_exn)
    for count, sample in slicer(enumerate(ds)):
        key, im = sample["__key__"], sample["jpg"]
        if select != "" and select not in key:
            continue
        boxes = segmenter.predict(im, scale=scale)
        if timeout > 0:
            tabseg_display(im, segmenter, title=f"{count}: {key}", timeout=timeout)
        result = {
            "__key__": key,
            "jpg": im,
            "cells.json": boxes
        }
        if sink is not None:
            sink.write(result)
    sink.close()


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
