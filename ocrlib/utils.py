import glob
import os
import re
import sys
import time

import numpy as np
import scipy.ndimage as ndi
import torch


debug = int(os.environ.get("UTILS_DEBUG", "0"))

do_trace = int(os.environ.get("OCROTRACE", "0"))


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


class Charset:
    def __init__(self, *, charset=None, chardef=None):
        self.charset = "".join([chr(i) for i in range(32, 128)])

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


def find_best_model(base, ext="pth", which=1, reverse=False):
    """Given a base directory, find the best model for model names that follow conventions."""
    pattern = f"{base}-*.{ext}"
    files = glob.glob(pattern)
    assert len(files) > 0, f"no {pattern} found"

    def keyfn(fname):
        match = re.search("(-[0-9]+)+[.]{ext}$", fname)
        return match.group(which)

    files = np.sort(files, key=keyfn, reverse=reverse)


def model_name(base, ntrain, loss, nscale=1e-3, lscale=1e6):
    ierr = int(lscale * loss)
    itrain = int(nscale * ntrain)
    return f"{base}-{itrain:08d}-{ierr:010d}.pth"


def load_model(model, fname):
    try:
        model.load_state_dict(torch.load(fname))
        print(f"# loaded state dict {type(model)}")
    except Exception as e:
        print(f"error loading {fname}", file=sys.stderr)
        print(e, file=sys.stderr)
        model = torch.load(fname)
        print(f"# loaded whole model {type(model)}")
    return model


def int2rgb(image):
    """Convert an int32 RGB image into a uint8 RGB image."""
    result = np.array([image >> 16, image >> 8, image], dtype="uint8")
    return result.transpose(1, 2, 0)


def number_list(ctx, param, value):
    """Convert a comma-separated list of numbers into a float list."""
    return [float(x) for x in value.split(",")]


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


def find_boundary(a, threshold):
    """Find the index of the first above-threshold element in a; 0 if none."""
    index = find_first(a > threshold, 0)
    if index > 0:
        return index
    return 0


def bbox2list(bbox):
    """Convert a bounding box (two slices) into a list [x0,y0,x1,y1]."""
    s0, s1 = bbox
    return [s1.start, s0.start, s1.stop, s0.stop]


def encode_bbox(bbox):
    """Convert a bounding box (two slices) into a dict(x0=,y0=,x1=,y1=)."""
    s0, s1 = bbox
    return dict(x0=s1.start, y0=s0.start, x1=s1.stop, y1=s0.stop)


def repeatedly(source):
    """Repeatedly yield samples from an iterator."""
    while True:
        for sample in source:
            yield sample


class ChangeLoaderEpochs(object):
    """Given a PyTorch data loader, change the epoch size."""

    def __init__(self, loader, epoch_size, batch_size=None, verbose=False):
        self.loader = loader
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.real_epoch = 0
        self.epoch = 0
        self.total = 0
        self.src = None

    def __iter__(self):
        ntrain = 0
        while True:
            if self.src is None:
                self.src = iter(self.loader)
            if self.verbose:
                print(
                    f"### {self.epoch} {self.real_epoch} {self.total} (epoch size: {self.epoch_size}, batch size: {self.batch_size})",
                    file=sys.stderr,
                )
            for batch in self.src:
                if ntrain >= self.epoch_size:
                    self.epoch += 1
                    return
                yield batch
                if self.batch_size is None:
                    self.batch_size = len(batch[0])
                ntrain += self.batch_size
                self.total += self.batch_size
            self.src = None
            self.real_epoch += 1


def make_unique(fname):
    """Make a filename unique by appending an integer if necessary."""
    if not os.path.exists(fname):
        return fname
    for i in range(1000):
        s = fname + f"-{i}"
        if not os.path.exists(s):
            return s
    return None


def mrot(a):
    """Make a rotation matrix."""
    from math import sin, cos

    return np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])


def eye23(m):
    """2D to 3D matrix"""
    result = np.eye(3)
    result[:2, :2] = m
    return result


def off23(d):
    """2D to 3D offset"""
    result = np.zeros(3)
    result[:2] = d
    return result


def make_affine(src, dst):
    """Compute affine transformation from src to dst points."""
    assert len(dst) == len(src), (src, dst)
    assert len(dst) >= 4, (src, dst)
    assert len(dst[0]) == 2, (src, dst)
    assert len(dst[0]) == len(src[0]), (src, dst)
    dst0 = dst - np.mean(dst, 0)[None, :]
    src0 = src - np.mean(src, 0)[None, :]
    H = np.dot(dst0.T, src0)
    U, S, V = np.linalg.svd(H)
    m = np.dot(V.T, U)
    d = np.dot(m, np.mean(dst, 0)) - np.mean(src, 0)
    # print(d)
    return m, d


def apply_affine(image, size, md, **kw):
    """Apply an affine transformation to an image.

    This takes care of the ndim==2 and ndim==3 cases."""
    h, w = size
    m, d = md
    if image.ndim == 2:
        return ndi.affine_transform(image, m, offset=-d, output_shape=(h, w), **kw)
    elif image.ndim == 3:
        return ndi.affine_transform(
            image, eye23(m), offset=-off23(d), output_shape=(h, w, 3), **kw
        )


def get_affine_patch(image, size, coords, **kw):
    """Get patch of the given size from the given source coordinates.

    Keyword arguments are passed on to ndi.affine_transform."""
    h, w = size
    y0, y1, x0, x1 = coords
    src = [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
    dst = [(0, 0), (0, w), (h, w), (h, 0)]
    md = make_affine(src, dst)
    return apply_affine(image, size, md, **kw)


def normalize_image(image, lo=0.1):
    """Normalize an image to the range (0, 1)."""
    image = image.astype(np.float32) - np.amin(image)
    image /= max(lo, np.amax(image))
    return image


def simple_bg_fg(binimage, amplitude=0.3, imsigma=1.0, sigma=3.0):
    """Simple noisy grascale image from a binary image."""
    bg = np.random.uniform(size=binimage.shape)
    bg = amplitude * normalize_image(ndi.gaussian_filter(bg, sigma))
    fg = np.random.uniform(size=binimage.shape)
    fg = 1.0 - amplitude * normalize_image(ndi.gaussian_filter(bg, sigma))
    mask = normalize_image(ndi.gaussian_filter(binimage, imsigma))
    return mask * fg + (1.0 - mask) * bg


def get_affine_patches(dst, src, images, size=None):
    """Extracts patches from `images` under the affine transformation
    estimated by transforming the points in src to the points in dst."""
    if size is None:
        pts = np.array(dst, "i")
        size = np.amax(pts, axis=0)
        h, w = size
    m, d = make_affine(src, dst)
    result = []
    for image in images:
        patch = apply_affine(image, (h, w), (m, d), order=1)
        result.append(patch)
    return result


def autoinvert(image, mode):
    if mode == "False":
        return image
    elif mode == "True":
        return np.amax(image) - image
    elif mode == "Auto":
        if np.mean(image) > np.mean([np.amax(image), np.amin(image)]):
            return np.amax(image) - image
        else:
            return image


def safe_randint(lo, hi):
    from numpy.random import randint

    return randint(lo, max(lo + 1, hi))


def get_patch(image, y0, y1, x0, x1, **kw):
    return ndi.affine_transform(image, np.eye(2), offset=(y0, x0), output_shape=(y1 - y0, x1 - x0), **kw)


def interesting_patches(
    indicator_image, threshold, images, r=256, n=50, trials=500, margin=0.1, jitter=5
):
    """
    Find patches that are "interesting" according to the indicator image; i.e., they need
    to include more than `threshold` values when summed over the patch.
        :param indicator_image: indicator image
        :param threshold: threshold for determining whether a patch is interesting
        :param images: list of images
        :param r=256: size of patch
        :param n=50: number of patches
        :param trials=500: number of trials
        :param margin=0.1: margin for the indicator image
        :param jitter=5: small deformation of source rectangle
    """
    from numpy.random import uniform

    h, w = indicator_image.shape[:2]
    count = 0
    for i in range(trials):
        if count >= n:
            break
        y = safe_randint(-r // 2, h - r//2 - 1)
        x = safe_randint(-r // 2, w - r//2 - 1)
        rx, ry = int(uniform(0.8, 1.2) * r), int(uniform(0.8, 1.2) * r)
        if margin < 1.0:
            dx, dy = int(rx * margin), int(ry * margin)
        else:
            dx, dy = int(margin), int(margin)
        patch = get_patch(indicator_image, y + dy , y + ry - dy, x + dx , x + rx - dx, order=0)
        if np.sum(patch) < threshold:
            continue
        rect = [y, x, y + ry, x + rx]
        rect = [c + safe_randint(-jitter, jitter) for c in rect]
        y0, x0, y1, x1 = rect
        src = [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
        dst = [(0, 0), (0, r), (r, r), (r, 0)]
        # print("*", src, dst)
        patches = get_affine_patches(dst, src, images)
        yield i, (x, y), patches
        count += 1


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
