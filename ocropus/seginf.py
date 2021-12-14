from itertools import islice

import typer
import numpy as np
import scipy.ndimage as ndi
import webdataset as wds
import torch
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt

from . import utils, loading, preinf

app = typer.Typer()


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = ndi.distance_transform_edt(labels == 0, return_distances=1, return_indices=1)
    indexes = features[0] * labels.shape[1] + features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= distances < maxdist
    return spread


def marker_segmentation(markers, regions, maxdist=100):
    labels, _ = ndi.label(markers)
    spread = spread_labels(labels, maxdist=maxdist)
    segmented = np.where(np.maximum(markers, regions), spread, 0)
    return segmented


class Segmenter:
    def __init__(self, model, device=None, invert=True):
        self.invert = invert
        self.model = model
        self.marker_threshold = 0.3
        self.region_threshold = 0.3
        self.maxdist = 100
        self.patchsize = (512, 512)
        self.overlap = (64, 64)
        self.device = device or utils.device()

    def segment(self, page: torch.Tensor) -> torch.Tensor:
        assert page.ndim == 3
        assert page.shape[0] == 3
        assert page.shape[1] >= 100 and page.shape[0] < 20000, page.shape
        assert page.shape[2] >= 100 and page.shape[1] < 20000, page.shape
        assert page.min() >= 0 and page.max() <= 1
        assert page.mean() > 0.5
        assert page.dtype == torch.float32
        self.page = page
        self.model.eval()
        self.model.to(self.device)
        f = partial(preinf.map_patch, self.model, device=self.device)
        if self.invert:
            page = 1 - page
        self.probs = preinf.patchwise(page, f, r=(400, 400), s=(300, 300)).softmax(0)
        assert self.probs.shape[0] < 10
        self.model.cpu()
        self.gprobs = self.probs.detach().cpu().permute(1, 2, 0).numpy()
        self.segments = marker_segmentation(
            self.gprobs[..., 3] > self.marker_threshold,
            self.gprobs[..., 2] > self.region_threshold,
            self.maxdist,
        )
        return [
            (obj[0].start, obj[0].stop, obj[1].start, obj[1].stop) for obj in ndi.find_objects(self.segments)
        ]


def extract_boxes(page, boxes, pad=5):
    for y0, y1, x0, x1 in boxes:
        h, w = y1 - y0, x1 - x0
        word = ndi.affine_transform(
            page,
            np.eye(2),
            output_shape=(h + 2 * pad, w + 2 * pad),
            offset=(y0 - pad, x0 - pad),
            order=0,
        )
        yield word


default_seg_model = "http://storage.googleapis.com/ocropus4-models/seg.jit.pt"


@app.command()
def segment(
    fname: str,
    model: str = default_seg_model,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "/dev/null",
    display: bool = True,
    limit: int = 999999999,
    device: Optional[str] = None,
    binarize: bool = True,
    binmodel: str = preinf.default_jit_model,
    show: float = -1,
):
    device = device or utils.device(device)
    print(f"# device {device}")

    if binmodel != "" and binmodel != "none":
        binarizer = preinf.Binarizer(model=preinf.default_jit_model)
    else:
        binarizer = None

    segmodel = loading.load_jit_model(model)
    segmenter = Segmenter(segmodel, device=device)

    dataset = wds.WebDataset(fname).decode("torchrgb")

    if show >= 0.0:
        plt.ion()

    # sink = wds.TarWriter(output)

    for sample in islice(dataset, 0, limit):
        image = wds.getfirst(sample, extensions)
        binary = binarizer.binarize(image)
        assert binary.dtype == torch.float32
        rgbbinary = torch.stack([binary] * 3)
        assert rgbbinary.dtype == torch.float32
        result = segmenter.segment(rgbbinary)
        if show >= 0.0:
            plt.clf()
            plt.subplot(122)
            plt.imshow(segmenter.gprobs[..., 1:])
            plt.subplot(121)
            plt.imshow(rgbbinary.numpy().transpose(1, 2, 0))
        for y0, y1, x0, x1 in result:
            if show >= 0.0:
                plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c="red")
        if show >= 0.0:
            plt.ginput(1, show)


if __name__ == "__main__":
    app()
