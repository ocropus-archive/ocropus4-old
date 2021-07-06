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
from . import utils


app = typer.Typer()


parser = argparse.ArgumentParser(
    """
Image binarization using non-linear processing.

This is a compute-intensive binarization method that works on degraded
and historical book pages.
"""
)

debug_nlbin = False


def check_page(image):
    """Checks whether the input roughly conforms to the requirements of a page."""
    if len(image.shape) == 3:
        raise ValueError("input image is color image %s" % (image.shape,))
    if np.mean(image) < np.median(image):
        raise ValueError("image may be inverted")
    h, w = image.shape
    if h < 600:
        raise ValueError("image not tall enough for a page image %s" % (image.shape,))
    if h > 10000:
        raise ValueError("image too tall for a page image %s" % (image.shape,))
    if w < 600:
        raise ValueError("image too narrow for a page image %s" % (image.shape,))
    if w > 10000:
        raise ValueError("line too wide for a page image %s" % (image.shape,))


def estimate_skew_angle(image, angles):
    """Estimate page skew angle from projections."""
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode="constant"), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    if debug_nlbin > 0:
        plt.plot([y for x, y in estimates], [x for x, y in estimates])
        plt.ginput(1, debug_nlbin)
    _, a = max(estimates)
    return a


def H(s):
    """Height of a slice."""
    return s[0].stop - s[0].start


def W(s):
    """Width of a slice."""
    return s[1].stop - s[1].start


def A(s):
    """Angle of a slice."""
    return W(s) * H(s)


def dshow(image, info):
    """Plot an image with info."""
    if debug_nlbin <= 0:
        return
    plt.ion()
    plt.gray()
    plt.imshow(image)
    plt.title(info)
    plt.ginput(1, debug_nlbin)


def normalize_raw_image(raw):
    """Perform simple image normalization."""
    image = raw - np.amin(raw)
    if np.amax(image) == np.amin(image):
        raise ValueError("image is empty")
    image /= np.amax(image)
    return image


def estimate_local_whitelevel(image, zoom=0.5, perc=80, dist=20, debug=0):
    """flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    dist for filters, default: %(default)s
    """
    m = interpolation.zoom(image, zoom)
    m = filters.percentile_filter(m, perc, size=(dist, 2))
    m = filters.percentile_filter(m, perc, size=(2, dist))
    m = interpolation.zoom(m, 1.0 / zoom)
    if debug > 0:
        plt.clf()
        plt.imshow(m, vmin=0, vmax=1)
        plt.ginput(1, debug)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)
    if debug > 0:
        plt.clf()
        plt.imshow(flat, vmin=0, vmax=1)
        plt.ginput(1, debug)
    return flat


def estimate_skew_and_fix(flat, bignore=0.1, maxskew=2, skewsteps=8):
    """ estimate skew angle and rotate"""
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)  # border ignore
    flat = np.amax(flat) - flat
    flat -= np.amin(flat)
    est = flat[o0 : d0 - o0, o1 : d1 - o1]
    ma = maxskew
    ms = int(2 * maxskew * skewsteps)
    # print(linspace(-ma,ma,ms+1))
    angle = estimate_skew_angle(est, np.linspace(-ma, ma, ms + 1))
    flat = interpolation.rotate(flat, angle, mode="constant", reshape=0)
    flat = np.amax(flat) - flat
    return flat, angle


def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
    """# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    """
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)
    est = flat[o0 : d0 - o0, o1 : d1 - o1]
    if escale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est - filters.gaussian_filter(est, e * 20.0)
        v = filters.gaussian_filter(v ** 2, e * 20.0) ** 0.5
        v = v > 0.3 * np.amax(v)
        v = morphology.binary_dilation(v, structure=np.ones((int(e * 50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e * 50))))
        if debug > 0:
            plt.imshow(v)
            plt.ginput(1, debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    return lo, hi


def nlbin(raw, args, deskew=True):
    """Nonlinear image binarization and deskewing."""
    assert raw.dtype == float
    image = normalize_raw_image(raw)
    flat = estimate_local_whitelevel(
        image, args.zoom, args.perc, args.dist, debug_nlbin
    )
    if deskew:
        flat, angle = estimate_skew_and_fix(flat, args.bignore, args.maxskew, args.skewsteps)
    lo, hi = estimate_thresholds(
        flat, args.bignore, args.escale, args.lo, args.hi, debug_nlbin
    )
    flat -= lo
    flat /= hi - lo
    flat = np.clip(flat, 0, 1)
    return flat


@app.command()
def binarize1(
    fname: str,
    threshold: float = -1,
    zoom: float = 0.5,
    escale: float = 1.0,
    bignore: float = 0.1,
    perc: float = 80,
    dist: int = 20,
    maxskew: float = 2,
    gray: bool = False,
    lo: float = 5,
    hi: float = 90,
    skewsteps: int = 8,
    debug: float = 0,
    output: str = "",
    deskew: bool = False,
):
    """Binarize a single image."""
    args = utils.Record(**locals())
    image = PIL.Image.open(args.fname)
    image = image.convert("L")
    image = np.asarray(image)
    assert image.dtype == np.uint8
    image = image / 255.0
    result = nlbin(image, args, deskew=deskew)
    assert result.dtype == float
    if args.threshold >= 0:
        result = np.array(result > args.threshold, dtype=np.uint8) * 255
    else:
        result = np.array(result * 255, dtype=np.uint8)
    result = PIL.Image.fromarray(result)
    result.save(args.output)


@app.command()
def binarize(
    fname: str,
    extensions: str = "jpg;jpeg;png",
    threshold: float = 0.5,
    zoom: float = 0.5,
    escale: float = 1.0,
    bignore: float = 0.1,
    perc: float = 80,
    dist: int = 20,
    maxskew: float = 2,
    gray: bool = False,
    lo: float = 5,
    hi: float = 90,
    skewsteps: int = 8,
    debug: float = 0,
    output: str = "",
    maxrec: int = 999999999999,
    deskew: bool = False,
    display: int = -1,
):
    """Binarize a shard of images."""
    args = utils.Record(**locals())
    ds = wds.WebDataset(fname).decode("l8").to_tuple("__key__", extensions)
    sink = wds.TarWriter(output)
    count = 0
    for key, image in islice(ds, 0, maxrec):
        print(f"{count}/{maxrec} {key}", file=sys.stderr)
        assert image.dtype == np.uint8
        image = image / 255.0
        flat = nlbin(image, args, deskew=deskew)
        assert flat.dtype == float
        result = dict(__key__=key)
        if gray:
            result["jpg"] = image
        if threshold >= 0:
            result["bin.png"] = np.array(flat > args.threshold, dtype=np.uint8) * 255
        result["nrm.jpg"] = np.array(flat * 255, dtype=np.uint8)
        sink.write(result)
        if display > 0 and count % display == 0:
            plt.ion()
            plt.clf()
            plt.subplot(131)
            plt.imshow(image, cmap="gray")
            plt.subplot(132)
            plt.imshow(result["nrm.jpg"], cmap="gray")
            if threshold >= 0:
                plt.subplot(133)
                plt.imshow(result["bin.png"], cmap="gray")
        if display > 0:
            plt.ginput(1, 0.02)
        count += 1
    sink.close()


if __name__ == "__main__":
    app()
