import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import webdataset as wds

from . import loading, nlbin, slog

app = typer.Typer()

logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

app = typer.Typer()

# default_jit_model = "http://storage.googleapis.com/ocropus4-models/ruby-sun-22-binarize.pt"
default_jit_model = (
    "http://storage.googleapis.com/ocropus4-models/effortless-glade-12-binarize.pt"
)

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
            result[y : y + r[0], x : x + r[1]] += output[
                : min(r[0], h - y), : min(r[1], w - x)
            ]
            counts[y : y + r[0], x : x + r[1]] += 1
    return result / np.maximum(counts, 1.0)


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
    model = loading.load_jit_model(model)
    if mode == "nlbin":
        print(
            "# using --mode=nlbin is slow; you can try --mode==normalize or --mode=none",
            file=sys.stderr,
        )
    # model = torch.jit.load(model)
    r = eval(f"({patch})")
    s = eval(f"({step})")
    source = wds.WebDataset(fname).decode("l").rename(jpeg=extensions)
    sink = wds.TarWriter(output)
    if show >= 0:
        plt.ion()
    for sample in source:
        print(sample["__key__"])
        raw = sample["jpeg"]
        print(raw.dtype, raw.shape)
        if mode == "normalize":
            lo, hi = np.percentile(raw, 5), np.percentile(raw, 95)
            hi = max(lo + 0.3, hi)
            image = raw - lo
            image /= hi - lo
            image += 1.0 - np.amax(image)
        elif mode == "nlbin":
            image = nlbin.nlbin(raw, deskew=deskew)
        else:
            image = raw
        binarized = patchwise(image, partial(map_patch, model), r=r, s=s)
        binarized = binarized.clip(0, 1)
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
