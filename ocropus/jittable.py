import math
import torch
import torch.jit
from torch.nn.functional import interpolate, pad
import random


@torch.jit.export
def findbbox1(line):
    lo = line.nonzero()[0][0]
    hi = line.nonzero()[-1][0]
    return lo, hi


@torch.jit.export
def findbbox(image):
    try:
        y0, y1 = findbbox1(image.sum(0).sum(1))
        x0, x1 = findbbox1(image.sum(0).sum(0))
        return x0, y0, x1, y1
    except IndexError:
        return -1, -1, -1, -1


@torch.jit.export
def quantscale(s, unit=math.sqrt(2)):
    assert s > 0, s
    assert unit > 0, unit
    return math.exp(math.floor(math.log(s) / math.log(unit) + 0.5) * math.log(unit))


@torch.jit.export
def standardize_image(im):
    if im.ndim == 2:
        im = im.unsqueeze(0).repeat(3, 1, 1)
    if im.dtype == torch.uint8:
        im = im.type(torch.float32) / 255.0
    if im.dtype == torch.float64:
        im = im.type(torch.float32)
    im -= im.amin()
    im /= max(float(im.amax()), 0.01)
    return im


@torch.jit.export
def resize_word(image, factor=7.0, quant=2.0, threshold=0.8):
    assert image.dtype == torch.float, image.dtype
    assert float(image.amax()) <= 1.01, image.amax()
    if image.amax() < 0.01:
        return torch.zeros((3, 1, 1))
    image = image / image.amax()
    c, h, w = image.shape
    assert c in [1, 3], c
    assert h > 16 and h < 1000
    assert w > 16 and w < 8000
    assert not torch.isnan(image).any()
    yprof = (image > threshold).sum(0).sum(1)
    if yprof.sum() < 1.0:
        return torch.zeros((3, 1, 1))
    ymean = (torch.linspace(0, h, len(yprof)) * yprof).sum() / yprof.sum()
    ystd = (torch.abs(torch.linspace(0, h, len(yprof)) - ymean) * yprof).sum() / yprof.sum()
    if ystd < 1.0:
        return torch.zeros((3, 1, 1))
    scale = factor / ystd
    assert scale > 0, scale
    scale = quantscale(scale, unit=quant)
    assert scale > 0, scale
    simage = interpolate(
        image.unsqueeze(0), (int(scale * h), int(scale * w)), mode="bilinear", align_corners=False
    )[0]
    return simage


@torch.jit.export
def crop_image(image, threshold=0.8, padding=4):
    c, h, w = image.shape
    if h <= 1 or w <= 1:
        return image
    x0, y0, x1, y1 = findbbox(image > 0.8)
    if x0 < 0:
        return torch.zeros((3, 1, 1))
    d = 5
    x0, y0, x1, y1 = max(x0 - d, 0), max(y0 - d, 0), min(x1 + d, w), min(y1 + d, h)
    simage = image[:, y0:y1, x0:x1]
    simage = pad(simage, [padding] * 4)
    return simage


@torch.jit.export
def stack_images(images):
    assert all(im.ndim == 3 for im in images)
    assert all(im.shape[0] in [1, 3] for im in images)
    assert all(im.shape[1] >= 16 for im in images)
    assert all(im.shape[2] >= 16 for im in images)
    bd, bh, bw = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), bd, bh, bw), dtype=torch.float)
    for i, im in enumerate(images):
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        d, h, w = im.shape
        dy, dx = (bh - h) // 2, (bw - w) // 2
        result[i, :d, dy : dy + h, dx : dx + w] = im
    return result
