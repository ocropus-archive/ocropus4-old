import importlib
import itertools as itt
import os
import re
import sys
import time
from functools import wraps
from typing import Union

from lxml import html
import numpy as np
import torch
from torchmore import layers

debug = int(os.environ.get("UTILS_DEBUG", "0"))

do_trace = int(os.environ.get("OCROTRACE", "0"))

all_models = []


def as_npimage(a: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a numpy image .

    Args:
        a (Union[torch.Tensor, np.ndarray]): some image

    Returns:
        np.ndarray: rank 3 floating point image
    """
    assert a.ndim == 3
    if isinstance(a, torch.Tensor):
        assert int(a.shape[0]) in [1, 3]
        a = a.detach().cpu().permute(1, 2, 0).numpy()
    assert isinstance(a, np.ndarray)
    assert a.shape[2] in [1, 3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return a


def as_torchimage(a: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert a numpy array into a torch . Image .

    Args:
        a (Union[torch.Tensor, np.ndarray]): some image in numpy or torch format

    Returns:
        torch.Tensor: floating point tensor representing an image
    """
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            a = np.stack((a,) * 3, axis=-1)
        assert int(a.shape[2]) in [1, 3]
        a = torch.tensor(a.transpose(2, 0, 1))
    assert a.ndim == 3
    assert isinstance(a, torch.Tensor)
    assert a.shape[0] in [1, 3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return a


def device(s):
    if s is None:
        if torch.cuda.is_available():
            result = torch.device("cuda:0")
        else:
            result = torch.device("cpu")
    else:
        result = torch.device(s)
    print(result, file=sys.stderr)
    return result


def unused(f):
    """Used to mark functions that are currently not used but are kept for future reference."""
    return f


def useopt(f):
    """Used to mark functions that are somehow used in command line options.

    Usually, this is via eval(f"prefix_{option}").
    """
    return f


def model(f):
    """Used to mark functions that create models."""
    global all_models
    all_models.append(f)
    return f


def public(f):
    """Marks a function as public (adds it to __all__)."""
    mdict = sys.modules[f.__module__].__dict__
    mall = mdict.setdefault("__all__", [])
    assert isinstance(mall, list)
    mall.append(f.__name__)
    return f


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
    (s,) = (re.sub("[\u201c\u201d]", '"', s),)
    (s,) = (re.sub("[\u2018\u2019]", "'", s),)
    (s,) = (re.sub("[\u2014]", "-", s),)
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


def autoinvert(image, mode, normalize=True):
    assert isinstance(image, np.ndarray)
    if image.dtype in (float, np.float32, np.float64):
        if normalize:
            image = image - np.amin(image)
            image /= max(0.1, np.amax(image))
        if mode == "False" or mode is False:
            return image
        elif mode == "True" or mode is True:
            return 1.0 - image
        elif mode == "Auto":
            if np.mean(image) > np.mean([np.amax(image), np.amin(image)]):
                return 1.0 - image
            else:
                return image
        else:
            raise ValueError(f"{image.dtype}: unsupported dtype")
    elif image.dtype == np.uint8:
        if normalize:
            image = image.astype(float) - float(np.amin(image))
            image *= 255.0 / max(0.1, np.amax(image))
            image = image.astype(np.uint8)
        if mode == "False" or mode is False:
            return image
        elif mode == "True" or mode is True:
            return 255 - image
        elif mode == "Auto":
            if np.mean(image) > np.mean([np.amax(image), np.amin(image)]):
                return 255 - image
            else:
                return image
        else:
            raise ValueError(f"unknown mode {mode}")
    else:
        raise ValueError(f"{image.dtype}: unsupported dtype")


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


def python_to_json(obj):
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (tuple, list)):
        return [python_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: python_to_json(v) for k, v in obj.items()}
    raise ValueError("bad type in fix4json", type(obj), obj)


def lrschedule(expr):
    return eval(f"lambda n: {expr}")


def flatten_yaml(d, result={}, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            result[prefix + k] = flatten_yaml(v, result, prefix=k + ".")
        return result
    else:
        return d


def unflatten_yaml(d):
    result = {}
    for k, v in d.items():
        target = result
        path = k.split(".")
        for s in path[:-1]:
            target = setdefault(target, s, {})
        target[k[-1]] = v
    return result


def load_module(filename):
    import importlib.util

    spec = importlib.util.spec_from_file_location("module", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_symbol(name):
    assert "." in name, f"{name}: symbol name must be fully qualified"
    mname, sname = name.rsplit(".", 1)
    module = importlib.import_module(mname)
    assert hasattr(module, sname), f"module {mname} has no attribute {sname}"
    return getattr(module, sname)


def get_s3_listing(url):
    data = os.popen("curl {}".format(url)).read()
    parsed = html.fromstring(data.encode("utf-8"))
    return [x.text for x in parsed.xpath("//key")]