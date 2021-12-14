from itertools import islice

import typer
import numpy as np
import scipy.ndimage as ndi
import webdataset as wds

from . import utils, patches, loading

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


def smooth_probabilities(probs, smooth):
    if smooth == 0.0:
        return probs
    if isinstance(smooth, (int, float)):
        smooth = (float(smooth), float(smooth), 0)
    else:
        assert len(smooth) == 2
        smooth = list(smooth) + [0]
    maxima = np.amax(np.amax(probs, 0), 0)
    assert maxima.shape == (4,)
    gprobs = ndi.gaussian_filter(probs, smooth)
    gmaxima = np.amax(np.amax(gprobs, 0), 0)
    gprobs /= (maxima / gmaxima)[np.newaxis, np.newaxis, :]
    gprobs = gprobs / gprobs.sum(2)[:, :, np.newaxis]
    return gprobs


class Segmenter:
    def __init__(self, model, scale=0.5, device=None):
        self.smooth = 0.0
        self.model = model
        self.marker_threshold = 0.3
        self.region_threshold = 0.3
        self.maxdist = 100
        self.patchsize = (512, 512)
        self.overlap = (64, 64)
        self.device = utils.device(device)

    def activate(self, yes=True):
        if yes:
            self.model.to(self.device)
        else:
            self.model.cpu()

    def npsegment(self, page):
        assert isinstance(page, np.ndarray)
        assert page.ndim == 2
        assert page.shape[0] >= 100 and page.shape[0] < 20000, page.shape
        assert page.shape[1] >= 100 and page.shape[1] < 20000, page.shape
        self.page = page
        self.activate()
        self.model.eval()
        if page.ndim == 2:
            page = np.expand_dims(page, 2)
        if page.shape[2] == 1:
            page = np.repeat(page, 3, 2)
        probs = patches.patchwise_inference(
            page,
            self.model,
            patchsize=self.patchsize,
            overlap=self.overlap,
        )
        self.probs = probs
        self.gprobs = smooth_probabilities(probs, self.smooth)
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


@app.command()
def segment(
    fname: str,
    model: str,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "",
    display: bool = True,
    limit: int = 999999999,
    device: str = None,
):
    device = utils.device(device)
    if device == "cpu":
        print("segment using CPU")
    model = loading.load_only_model(model)
    segmenter = Segmenter(model, device=device)

    dataset = wds.WebDataset(fname).decode("rgb")

    for sample in islice(dataset, 0, limit):
        image = wds.getfirst(sample, extensions)
        image = np.mean(image, 2)
        segmenter.segment(image)

        pass  # FIXME do something here


if __name__ == "__main__":
    app()
