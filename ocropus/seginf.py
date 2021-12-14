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


def remove_unmarked_regions(markers, regions):
    """Remove regions that are not marked by markers."""
    m = 1000000
    labels, _ = ndi.label(markers)
    rlabels, rn = ndi.label(regions)
    corr = np.unique((rlabels * m + labels).ravel())
    remap = np.zeros(rn + 1, dtype=np.int32)
    for k in corr:
        remap[k // m] = k % m
    return remap[rlabels]


def marker_segmentation(markers, regions, maxdist=100):
    regions = np.maximum(regions, markers)
    labels, _ = ndi.label(markers)
    regions = (remove_unmarked_regions(markers, regions) > 0)
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
        outer = np.maximum(self.gprobs[:, :, 2], self.gprobs[:, :, 3])
        outer = ndi.grey_opening(outer, (4, 4))
        outer = outer > 0.5
        inner = self.gprobs[:, :, 3]
        inner = ndi.grey_closing(inner, (2, 2))
        inner = inner > 0.3
        self.outer, self.inner = outer, inner
        self.segments = marker_segmentation(inner, outer, maxdist=self.maxdist)
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

    binarizer = preinf.Binarizer(model=binmodel, device=device)

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
            plt.subplot(222)
            plt.imshow(segmenter.gprobs[..., 1:])
            plt.subplot(221)
            plt.imshow(rgbbinary.numpy().transpose(1, 2, 0))
        for y0, y1, x0, x1 in result:
            if show >= 0.0:
                plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c="red")
        if show >= 0.0:
            plt.subplot(223)
            plt.imshow(segmenter.inner)
            plt.subplot(224)
            plt.imshow(segmenter.outer)
            plt.ginput(1, show)


if __name__ == "__main__":
    app()
