import math
from typing import List, Tuple

import torch
import torch.jit
from torch.nn.functional import interpolate, pad


@torch.jit.export
def findbbox1(line: torch.Tensor) -> Tuple[int, int]:
    nz = line.nonzero()
    if nz.size(0) == 0:
        return -1, -1
    return int(nz[0, 0]), int(nz[-1, 0])


@torch.jit.export
def findbbox(image: torch.Tensor) -> Tuple[int, int, int, int]:
    y0, y1 = findbbox1(image.sum(0).sum(1))
    x0, x1 = findbbox1(image.sum(0).sum(0))
    return x0, y0, x1, y1


@torch.jit.export
def quantscale(s: float, unit: float = math.sqrt(2)):
    assert s > 0, s
    assert unit > 0, unit
    return math.exp(math.floor(math.log(s) / math.log(unit) + 0.5) * math.log(unit))


@torch.jit.export
def standardize_image(
    im: torch.Tensor, lo: float = 0.05, hi: float = 0.95
) -> torch.Tensor:
    if im.ndim == 2:
        im = im.unsqueeze(0).repeat(3, 1, 1)
    if im.dtype == torch.uint8:
        im = im.type(torch.float32) / 255.0
    if im.dtype == torch.float64:
        im = im.type(torch.float32)
    im -= torch.quantile(im, lo)
    im /= max(float(torch.quantile(im, hi)), 0.01)
    if torch.quantile(im, 0.5) > 0.5:
        im = 1 - im
    im = im.clamp(0, 1)
    return im


@torch.jit.export
def resize_word(
    image: torch.Tensor, factor: float = 7.0, quant: float = 2.0, threshold: float = 0.8
) -> torch.Tensor:
    assert image.dtype == torch.float, image.dtype
    assert image.min() >= 0 and image.max() <= 1
    if image.amax() < 0.01:
        return torch.zeros((3, 1, 1))
    image = image / image.amax()
    c, h, w = image.shape
    assert c in [1, 3], c
    if h < 16 or h > 1000 or w < 16 or w > 8000:
        return torch.zeros((3, 1, 1))
    assert not torch.isnan(image).any()
    yprof = (image > threshold).sum(0).sum(1)
    if yprof.sum() < 1.0:
        return torch.zeros((3, 1, 1))
    ymean = (torch.linspace(0, h, len(yprof)) * yprof).sum() / yprof.sum()
    ystd = (
        torch.abs(torch.linspace(0, h, len(yprof)) - ymean) * yprof
    ).sum() / yprof.sum()
    if ystd < 1.0:
        return torch.zeros((3, 1, 1))
    scale = factor / ystd
    assert scale > 0, scale
    scale = quantscale(scale, unit=quant)
    assert scale > 0, scale
    assert image.min() >= 0 and image.max() <= 1
    simage = interpolate(
        image.unsqueeze(0),
        (int(scale * h), int(scale * w)),
        mode="bilinear",
        align_corners=False,
    )[0]
    simage = simage.clamp(0, 1)
    return simage


@torch.jit.export
def crop_image(
    image: torch.Tensor, threshold: float = 0.8, padding: int = 4
) -> torch.Tensor:
    assert image.min() >= 0 and image.max() <= 1
    c, h, w = image.shape
    if h <= 1 or w <= 1:
        return image
    x0, y0, x1, y1 = findbbox(image > 0.8)
    if x0 < 0 or y0 < 0:
        return torch.zeros((3, 1, 1))
    d = 5
    x0, y0, x1, y1 = max(x0 - d, 0), max(y0 - d, 0), min(x1 + d, w), min(y1 + d, h)
    simage = image[:, y0:y1, x0:x1]
    simage = pad(simage, [padding] * 4)
    return simage


@torch.jit.export
def auto_resize(im: torch.Tensor) -> torch.Tensor:
    assert im.min() >= 0 and im.max() <= 1
    resized = resize_word(im)
    cropped = crop_image(resized)
    assert im.min() >= 0 and im.max() <= 1
    return cropped


@torch.jit.export
def stack_images(images: List[torch.Tensor]) -> torch.Tensor:
    for im in images:
        assert im.ndim == 3, im.ndim
        assert im.shape[0] in [1, 3]
        assert im.shape[1] >= 16
        assert im.shape[2] >= 16
        assert im.min() >= 0 and im.max() <= 1
    maxima = torch.zeros(3, dtype=torch.int)
    for im in images:
        maxima = torch.max(maxima, torch.tensor(im.shape))
    bd, bh, bw = int(maxima[0]), int(maxima[1]), int(maxima[2])
    result = torch.zeros((len(images), bd, bh, bw), dtype=torch.float)
    for i, im in enumerate(images):
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        d, h, w = im.shape
        dy, dx = (bh - h) // 2, (bw - w) // 2
        result[i, :d, dy : dy + h, dx : dx + w] = im
    return result
