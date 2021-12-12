import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import webdataset as wds

from . import loading, nlbin, slog, utils

app = typer.Typer()

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()

# default_jit_model = "http://storage.googleapis.com/ocropus4-models/ruby-sun-22-binarize.pt"

# In[29]:


def map_patch(model, patch):
    h, w = patch.shape
    if w < 16 or h < 16:
        return None
    a = torch.tensor(patch).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        result = model(a.cuda()).cpu()
    return result[0, 0].detach().cpu().numpy()


def patchwise(image, f, r=(256, 1024), s=(177, 477)):
    h, w = image.shape
    result = np.zeros(image.shape)
    counts = np.zeros(image.shape)
    for y in range(0, h, s[0]):
        for x in range(0, w, s[1]):
            input = image[y : y + r[0], x : x + r[1]]
            output = f(input)
            if output is None:
                continue
            # print(result[y:y+r, x:x+r].shape, output[:r, :r].shape)
            result[y : y + r[0], x : x + r[1]] += output[: min(r[0], h - y), : min(r[1], w - x)]
            counts[y : y + r[0], x : x + r[1]] += 1
    return result / np.maximum(counts, 1.0)


def norm_none(raw):
    return raw


def norm_simple(raw):
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
    return result.cpu()


class Binarizer:
    def __init__(
        self,
        model,
        mode="nlbin",
        r=(256, 1024),
        s=(177, 477),
    ):
        self.normalizer = utils.load_symbol(f"ocropus/ocroimg/norm_{mode}")
        self.model = loading.load_jit_model(model)
        self.model.eval()
        self.r, self.s = r, s

    def binarize(self, image: torch.Tensor) -> torch.Tensor:
        assert isinstance(image, torch.Tensor)
        if image.ndim == 3:
            image = image.mean(0)
        image = self.normalizer(image)
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        result = patchwise(image, partial(map_patch, self.model), r=self.r, s=self.s)
        result = result.clip(0, 1)
        return result


default_jit_model = "http://storage.googleapis.com/ocropus4-models/effortless-glade-12-binarize.pt"


@app.command()
def binarize(
    fname: str,
    model: str = default_jit_model,
    output: str = "",
    extensions="jpg;png;jpeg;page.jpg;page.jpeg",
    mode: str = "nlbin",
    deskew: bool = True,
    keep: bool = True,
    patch: str = "256,1024",
    step: str = "177,477",
    show: float = -1,
) -> None:
    """Binarize the images in a WebDataset."""
    assert model != "", "must specify model"
    assert output != "", "must specify output"
    binarizer = Binarizer(model, mode=mode, r=eval(f"({patch})"), s=eval(f"({step})"))
    source = wds.WebDataset(fname).decode("torchrgb").rename(jpeg=extensions)
    sink = wds.TarWriter(output)
    if show >= 0:
        plt.ion()
    for sample in source:
        raw = wds.getfirst(sample, extensions)
        if raw is None:
            continue
        print(sample["__key__"], raw.dtype, raw.shape)
        binarized = binarizer.binarize(raw)
        if show >= 0:
            plt.subplot(1, 2, 1)
            plt.imshow(raw)
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
