import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import webdataset as wds
import scipy.ndimage

from . import nlbin, utils, utils, loading

app = typer.Typer()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()

# default_jit_model = "http://storage.googleapis.com/ocropus4-models/ruby-sun-22-binarize.pt"


def map_patch(model, patch):
    assert patch.ndim == 3, patch.shape
    c, h, w = patch.shape
    if w < 16 or h < 16:
        return None
    model.eval()
    with torch.no_grad():
        result = model(patch.unsqueeze(0).cuda()).cpu()[0]
    assert result.ndim == 3, result.shape
    # assert result.shape[0] == 2, result.shape
    if result.shape[-2:] != (h, w):
        assert abs(result.shape[1] - h) <= 8, [result.shape, h]
        assert abs(result.shape[2] - w) <= 8, [result.shape, w]
        temp = torch.zeros((1, h, w), dtype=result.dtype)
        oh, ow = min(h, result.shape[1]), min(w, result.shape[2])
        temp[:, :oh, :ow] = result[:, :oh, :ow]
        result = temp
    return result


def patchwise(image: torch.Tensor, f, r=(256, 1024), s=(177, 477), m=(64, 64)):
    assert image.ndim == 3, image.shape
    h, w = image.shape[-2:]
    # result = torch.zeros((h, w), dtype=image.dtype)
    result = None
    counts = torch.zeros((h, w), dtype=torch.int32)
    for y in range(0, h, s[0]):
        for x in range(0, w, s[1]):
            input = image[:, y : y + r[0], x : x + r[1]]
            if input.shape[-2] < m[0] or input.shape[-1] < m[1]:
                continue
            output = f(input)
            if output is None:
                continue
            assert output.ndim == 3
            assert output.shape[-2:] == input.shape[-2:], [output.shape, input.shape]
            if result is None:
                result = torch.zeros((output.shape[0], h, w), dtype=output.dtype)
            result[:, y : y + r[0], x : x + r[1]] += output[:, : min(r[0], h - y), : min(r[1], w - x)]
            counts[y : y + r[0], x : x + r[1]] += 1
    return result / np.maximum(counts, 1.0)[None, :, :]


def norm_none(raw):
    raw = raw.mean(axis=0, keepdims=True)
    return raw


def norm_simple(raw):
    raw = raw.mean(axis=0, keepdims=True)
    lo, hi = np.percentile(raw, 5), np.percentile(raw, 95)
    hi = max(lo + 0.3, hi)
    image = raw - lo
    image /= hi - lo
    image += 1.0 - np.amax(image)
    return image


def norm_oldnlbin(raw):
    return nlbin.nlbin(raw, deskew=False)


def locmax(image, d):
    import torch.nn.functional as F

    temp = F.max_pool2d(image, (d, 1), stride=1, padding=(d // 2, 0))
    result = F.max_pool2d(temp, (1, d), stride=1, padding=(0, d // 2))
    return result


def norm_nlbin(page, zoomed=4.0, hi_r=50, lo_r=30, min_delta=0.5):
    import torch.nn.functional as F
    import kornia.filters as kfilters

    assert page.ndim == 3
    page = 1.0 - page
    page = page.mean(axis=0, keepdims=True)
    page = page.unsqueeze(0)
    page = page.cuda()
    assert torch.min(page) >= 0.0 and torch.max(page) <= 1.0
    est = F.interpolate(page, scale_factor=1.0 / zoomed, mode="bilinear")
    est = kfilters.gaussian_blur2d(est, (5, 5), (2, 2), border_type="replicate")
    hr, lr = int(hi_r / zoomed), int(lo_r / zoomed)
    hr, lr = hr + ((hr + 1) % 2), lr + ((lr + 1) % 2)
    hi = locmax(est, hr)
    hi = kfilters.gaussian_blur2d(hi, (hr, hr), (hr // 3, hr // 3), border_type="replicate")
    hi = F.interpolate(hi, size=page.shape[-2:])
    lo = -locmax(-est, lr)
    lo = kfilters.gaussian_blur2d(lo, (lr, lr), (lr // 3, lr // 3), border_type="replicate")
    lo = F.interpolate(lo, size=page.shape[-2:])
    result = (page - lo) / torch.max(hi - lo, torch.tensor(0.5))
    result = result[0]
    assert result.ndim == 3
    result = 1.0 - result
    result.clamp_(0.0, 1.0)
    return result.cpu()


class Binarizer:
    def __init__(
        self,
        model="",
        mode="nlbin",
        r=(256, 1024),
        s=(177, 477),
        verbose=False,
    ):
        self.normalizer = utils.load_symbol(f"ocropus.preinf.norm_{mode}")
        if model != "":
            self.model = loading.load_jit_model(model)
            self.model.eval()
            self.model.cuda()
        else:
            self.model = None
        self.r, self.s = r, s
        self.verbose = verbose

    def npbinarize(self, image: np.ndarray, zoom=1.0, zoomed=1.0) -> np.ndarray:
        assert image.ndim == 3 and image.shape[2] == 3, image.shape
        assert image.shape[0] >= 16 and image.shape[1] >= 16, image.shape
        temp = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))
        result = self.binarize(temp, zoom=zoom, zoomed=zoomed)
        assert result.ndim == 2, result.shape
        return result.cpu().numpy()

    def binarize(self, image: torch.Tensor, zoom=1.0, zoomed=1.0) -> torch.Tensor:
        import torch.nn.functional as F

        prezoom = zoom * zoomed
        if prezoom != 1.0:
            image = F.interpolate(image.unsqueeze(0), scale_factor=prezoom, mode="bilinear")[0]
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert int(image.shape[0]) in [1, 3]
        if self.verbose:
            print(f"# Binarizing {image.shape}")
        image = self.normalizer(image)
        assert image.min() >= 0 and image.max() <= 1
        assert int(image.shape[0]) in [1, 3]
        if self.model is not None:
            if self.verbose:
                print("# Using model")
            image = image.mean(axis=0, keepdim=True)
            image = patchwise(image, partial(map_patch, self.model), r=self.r, s=self.s)
            assert image.ndim == 3
            assert image.shape[0] == 1
            image = image[0]
            image = image.clip(0, 1)
        else:
            if self.verbose:
                print("# Not using model")
            image = image.mean(axis=0)
        if self.verbose:
            print(f"# Binarized {image.shape}")
        if zoomed != 1.0:
            image = F.interpolate(
                image.unsqueeze(0).unsqueeze(0), scale_factor=1.0 / zoomed, mode="bilinear"
            )[0, 0]
        return image


default_jit_model = "http://storage.googleapis.com/ocropus4-models/effortless-glade-12-binarize.pt"


@app.command()
def binarize(
    fname: str,
    model: str = "",
    output: str = "",
    extensions="jpg;png;jpeg;page.jpg;page.jpeg",
    mode: str = "nlbin",
    deskew: bool = True,
    keep: bool = True,
    patch: str = "400, 400",
    step: str = "300, 300",
    show: float = -1,
    verbose: bool = False,
    zoom: float = 1.0,
    zoomed: float = 1.0,
) -> None:
    """Binarize the images in a WebDataset."""
    if model == "default":
        model = default_jit_model
    assert output != "", "must specify output"
    binarizer = Binarizer(model=model, mode=mode, r=eval(f"({patch})"), s=eval(f"({step})"), verbose=verbose)
    source = wds.WebDataset(fname).decode("rgb").rename(jpeg=extensions)
    sink = wds.TarWriter(output)
    if show >= 0:
        plt.ion()
    for sample in source:
        img = wds.getfirst(sample, extensions)
        if img is None:
            continue
        raw = torch.tensor(img.transpose(2, 0, 1))
        print(sample["__key__"], raw.dtype, raw.shape)
        assert isinstance(raw, torch.Tensor)
        assert raw.ndim == 3
        binarized = binarizer.binarize(raw, zoom=zoom, zoomed=zoomed)
        assert isinstance(binarized, torch.Tensor)
        assert binarized.ndim == 2
        if zoom == 1.0:
            assert binarized.shape[-2:] == raw.shape[-2:]
        binarized = torch.stack([binarized, binarized, binarized], axis=0).numpy().transpose(1, 2, 0)
        if show >= 0:
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(binarized)
            plt.ginput(1, show)
        if keep:
            result = dict(sample)
        else:
            result = dict(__key__=sample["__key__"])
        result.update({"bin.jpg": binarized})
        sink.write(result)
    sink.close()


if __name__ == "__main__":
    app()
