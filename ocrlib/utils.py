import itertools as itt
import os
import sys
import time
from functools import wraps
import re
import matplotlib.pyplot as plt

import numpy as np
import scipy.ndimage as ndi
import torch
from torchmore import layers

debug = int(os.environ.get("UTILS_DEBUG", "0"))

do_trace = int(os.environ.get("OCROTRACE", "0"))


def junk(message="don't use this function anymore"):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kw):
            raise Exception(message)

        return wrapped

    return decorator


def trace(f):
    import functools

    @functools.wraps(f)
    def wrapped(*args, **kw):
        global do_trace
        name = f.__name__
        args_summary = f"{args} {kw}"[:70]
        if do_trace:
            print(f"> {name} {args_summary}")
        result = f(*args, **kw)
        if do_trace:
            print(f"< {name}")
        return result

    return wrapped


def enumerated(source, start=0, limit=999999999999, message=None):
    """Like enumerate, but with limit and message."""
    count = 0
    for x in source:
        if count >= limit:
            if message is not None:
                print(message, file=sys.stderr)
            return
        yield count, x


class Record:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Every(object):
    """Trigger an action every given number of seconds."""

    def __init__(self, seconds, atstart=False):
        self.seconds = seconds
        if atstart:
            self.last = 0
        else:
            self.last = time.time()

    def __call__(self):
        now = time.time()
        if now - self.last >= self.seconds:
            self.last = now
            return True
        return False


def fix_quotes(s):
    assert isinstance(s, str)
    s, = re.sub("[\u201c\u201d]", '"', s),
    s, = re.sub("[\u2018\u2019]", "'", s),
    s, = re.sub("[\u2014]", "-", s),
    return s


class Charset:
    def __init__(self, *, charset=None, chardef=None):
        self.charset = "".join([chr(i) for i in range(32, 128)])
        self.normalize = [fix_quotes]

    def __len__(self):
        return len(self.charset)

    def encode_chr(self, c):
        charset = self.charset
        index = charset.find(c)
        if index < 0:
            assert "~" in charset
            return charset.find("~") + 1
        return index + 1

    def encode_str(self, s):
        for f in self.normalize:
            s = f(s)
        return [self.encode_chr(c) for c in s]

    def decode_chr(self, k):
        return self.charset[k - 1]

    def decode_str(self, lst):
        return "".join([self.charset[k - 1] for k in lst])

    def preptargets(self, s):
        return torch.tensor(self.encode_str(s)).long()


def array_info(data):
    """Summarize information about an array. Works for both torch and numpy."""
    if isinstance(data, np.ndarray):
        tp = f"np:{data.dtype}"
        lo = np.min(data)
        med = np.mean(data)
        hi = np.max(data)
        shape = ",".join(map(str, data.shape))
    else:
        tp = f"t:{data.dtype}"
        lo = float(data.min())
        med = float(data.mean())
        hi = float(data.max())
        shape = ",".join(map(str, tuple(data.size())))
    return f"<{tp} {shape} [{lo:.2e}:{med:.2e}:{hi:.2e}]>"


def array_infos(**kw):
    """Summarize information about a list of arrays. Works for both torch and numpy."""
    return " ".join(f"{k}={array_info(v)}" for k, v in sorted(list(kw.items())))


def imshow_tensor(a, order, b=0, ax=None, **kw):
    """Display a torch array with imshow."""
    from matplotlib.pyplot import gca

    ax = ax or gca()
    if set(order) == set("BHWD"):
        a = layers.reorder(a.detach().cpu(), order, "BHWD")[b].numpy()
    elif set(order) == set("HWD"):
        a = layers.reorder(a.detach().cpu(), order, "HWD").numpy()
    elif set(order) == set("HW"):
        a = layers.reorder(a.detach().cpu(), order, "HW").numpy()
    else:
        raise ValueError(f"{order}: unknown order")
    if a.shape[-1] == 1:
        a = a[..., 0]
    ax.imshow(a, **kw)


def autoinvert(image, mode):
    image = image - np.amin(image)
    image /= max(0.1, np.amax(image))
    if mode == "False":
        return image
    elif mode == "True":
        return 1.0 - image
    elif mode == "Auto":
        if np.mean(image) > np.mean([np.amax(image), np.amin(image)]):
            return 1.0 - image
        else:
            return image


def safe_randint(lo, hi):
    from numpy.random import randint

    return randint(lo, max(lo + 1, hi))


def batch_images(*args):
    images = [a.cpu().float() for a in args]
    dims = np.array([tuple(a.shape) for a in images])
    maxdims = [x for x in np.max(dims, 0)]
    result = torch.zeros([len(images)] + maxdims)
    for i, a in enumerate(images):
        d, h, w = a.shape
        if d == 1:
            result[i, :, :h, :w] = a
        else:
            result[i, :d, :h, :w] = a
    return result


def pad_slice(sl, r):
    """Pad a slice by #pixels (r>=1) or proportionally (0<r<1)."""
    if isinstance(r, int):
        return slice(max(sl.start - r, 0), sl.stop + r)
    elif isinstance(r, float):
        d = int((0.5 + sl.stop - sl.start) * r)
        return slice(max(sl.start - d, 0), sl.stop + d)
    else:
        raise ValueError(f"range {r}")


def find_first(a, default):
    """Find the index of the first non-zero element in a."""
    index = np.flatnonzero(a)
    if len(index) == 0:
        return default
    else:
        return index[0]


def make_unique(fname):
    """Make a filename unique by appending an integer if necessary."""
    if not os.path.exists(fname):
        return fname
    for i in range(1000):
        s = fname + f"-{i}"
        if not os.path.exists(s):
            return s
    return None


def normalize_image(image, lo=0.1):
    """Normalize an image to the range (0, 1)."""
    image = image.astype(np.float32) - np.amin(image)
    image /= max(lo, np.amax(image))
    return image


class BBox:
    """A simple bounding box class, for compatibility
    with slice-based code."""

    def __init__(self, y0, y1, x0, x1):
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1

    def __getitem__(self, index):
        assert index >= 0 and index <= 1
        if index == 0:
            return slice(self.y0, self.y1)
        elif index == 1:
            return slice(self.x0, self.x1)

    def union(self, other):
        return BBox(
            min(self.y0, other.y0),
            max(self.y1, other.y1),
            min(self.x0, other.x0),
            max(self.x1, other.x1),
        )

    def coords(self):
        return tuple(int(x) for x in [self.y0, self.y1, self.x0, self.x1])


class Schedule:
    def __init__(self):
        self.jobs = {}

    def __call__(self, key, seconds, initial=False, verbose=False):
        now = time.time()
        last = self.jobs.setdefault(key, 0 if initial else now)
        if now - last > seconds:
            if verbose:
                print("# Schedule", now, last, seconds, file=sys.stderr)
            self.jobs[key] = now
            return True
        else:
            return False


def repeatedly(loader, nepochs=999999999, nbatches=999999999999, verbose=False):
    for epoch in range(nepochs):
        if verbose:
            print("# epoch", epoch, file=sys.stderr)
        for sample in itt.islice(loader, nbatches):
            yield sample


def label_correspondences(nlabels, olabels):
    n = np.amax(olabels) + 1
    m = 1000000
    assert n < m
    assert nlabels.shape == olabels.shape
    a = nlabels.ravel() * m + olabels.ravel()
    a = np.unique(a)
    result = np.zeros(n, dtype="i")
    for p in a:
        result[p % m] = p // m
    return result


def fix_bounding_boxes(bimage, bboxes):
    # global mcomponents, components, remap, markers
    components, _ = ndi.label(bimage)
    markers = np.zeros_like(bimage, dtype="i")
    for x0, y0, x1, y1 in bboxes:
        ym, _ = int(np.mean([y1, y0])), int(np.mean([x1, x0]))
        xoff = 3
        yoff = (y1 - y0) // 3
        x0, x1 = x0 + xoff, x1 - xoff
        y0, y1 = ym - yoff, ym + yoff
        if y0 < 0 or x1 <= x0 or y1 <= y0:
            continue
        markers[y0:y1, x0:x1] = 1
    mcomponents, _ = ndi.label(markers)
    remap = label_correspondences(mcomponents, components)
    remap[0] = 0
    components = remap[components]
    result = []
    for y, x in ndi.find_objects(components):
        result.append((x.start, y.start, x.stop, y.stop))
    return result
