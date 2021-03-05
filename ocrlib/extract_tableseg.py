import io
import os
import re
import sys

import numpy as np
import webdataset as wds
from lxml import etree
from matplotlib import pylab
import scipy.ndimage as ndi

from . import utils
from .utils import unused
import .patches

import typer

debug = int(os.environ.get("EXTRACT_SEG_DEBUG", "0"))

app = typer.Typer()


def get_text(node):
    """Extract the text from a DOM node."""
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s)


def get_prop(node, name):
    """Extract a property from an hOCR DOM node."""
    title = node.get("title")
    if title is None:
        return title
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


@unused
def get_any(sample, key, default=None):
    """Given a list of keys, return the first dictionary entry matching a key."""
    if isinstance(key, str):
        key = key.split(";")
    for k in key:
        if k in sample:
            return sample[k]
    return default


def center(bbox):
    """Compute the center of a bounding box."""
    x0, y0, x1, y1 = bbox
    return int(np.mean([x0, x1])), int(np.mean([y0, y1]))


def dims(bbox):
    """Compute the w, h of a bounding box."""
    x0, y0, x1, y1 = bbox
    return x1 - x0, y1 - y0


@utils.trace
def bboxes_for_hocr(image, hocr, element="ocrx_table_cell"):
    """
    Generate a segmentation target given an image and hOCR segmentation info.
        :param image: page image
        :param hocr: hOCR format info corresponding to page image
        :param element: hOCR target unit (usually, "ocrx_word" or "ocr_line")
    """
    htmlparser = etree.HTMLParser()
    doc = etree.parse(io.BytesIO(hocr), htmlparser)
    pages = list(doc.xpath('//*[@class="ocr_page"]'))
    assert len(pages) == 1
    page = pages[0]
    h, w = image.shape[:2]
    ocr_bbox = get_prop(page, "bbox")
    if ocr_bbox is not None:
        _, _, w1, h1 = [int(x) for x in ocr_bbox.split()]
        if h1 != h or w1 != w:
            print(
                f"image and page dimensions differ ({h}, {w}) != ({h1}, {w1})",
                file=sys.stderr,
            )
    # print(page.get("title"))
    bboxes = []
    for word in page.xpath(f"//*[@class='{element}']"):
        bbox_str = get_prop(word, "bbox")
        if bbox_str is None:
            continue
        bbox = [int(x) for x in bbox_str.split()]
        bboxes.append(bbox)
    return bboxes


@utils.trace
def table_segmentation_target(
    page, hocr, labels=[1, 2, 3], offsets=[12, 2, -10], element="ocrx_table_cell",
):
    """Extract training patches for segmentation."""
    if page.ndim == 3:
        page = np.mean(page, 2)
    a, b, c = offsets
    bboxes = bboxes_for_hocr(page, hocr, element=element)
    h, w = page.shape[:2]
    target = np.zeros((h, w), dtype="uint8")
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        target[y0 - a : y1 + a, x0 - a : x1 + a] = labels[0]
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        target[y0 - b : y1 + b, x0 - b : x1 + b] = labels[1]
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        cy, cx = max(-(y1 - y0) // 2 + 5, c), max(-(x1 - x0) // 2 + 5, c)
        target[y0 - cy : y1 + cy, x0 - cx : x1 + cx] = labels[2]
    return target


@unused
def augment_segmentation(page, seg, rotation=[0.0, 0.0], scale=[1.0, 1.0], minmark=1):
    if np.sum(seg) <= minmark:
        print(f"no output in segmentation map", file=sys.stderr)
        return
    if scale[0] != 1.0 or scale[1] != 1.0:
        s = np.random.uniform(*scale)
        page = ndi.zoom(page, s, order=1, mode="nearest")
        seg = ndi.zoom(seg, s, order=0, mode="constant", cval=0)
    if rotation[1] != 0.0 or rotation[0] != 0.0:
        alpha = np.random.uniform(*rotation)
        page = ndi.rotate(page, alpha, order=1, mode="nearest")
        seg = ndi.rotate(seg, alpha, order=0, mode="constant", cval=0)
    return page, seg


@utils.trace
def patches_of_segmentation(
    page, seg, label_threshold=2, threshold=10, patchsize=512, n=64
):
    assert isinstance(page, np.ndarray), type(page)
    assert isinstance(seg, np.ndarray), type(seg)
    # assert page.dtype == np.float32, page.dtype
    assert seg.dtype in [np.uint8, np.int32, np.int64], seg.dtype
    assert page.ndim == 2
    assert page.shape == seg.shape
    patchlist = list(
        patches.interesting_patches(
            np.array(seg >= label_threshold, "i"),
            threshold,
            [page, seg],
            r=patchsize,
            n=n,
        )
    )
    print("# interesting patches", len(patches))
    for patch in patchlist:
        yield patch


@app.command()
def tables2seg(
    src: str,
    output: str = "",
    extensions: str = "page.jpg;page.png;png;jpg;jpeg;JPEG;PNG hocr;HOCR;hocr.html",
    maxcount: int = 9999999999,
    subsample: float = 1.0,
    patches_per_image: int = 50,
    show: int = 0,
    patchsize: int = 512,
    scales: str = "0.7, 1.0, 1.3",
    offsets: str = "12, 2, -10",
    rotations: str = "-0.5, 0.5",
    element: str = "ocrx_table_cell",
    skip_missing=True,
    ignore_errors=False,
    debug=False,
    invert: str = "Auto",
    labels: str = "1, 2, 3",
):
    """Extract segmentation patches from src and send them to output."""
    labels = eval(f"[{labels}]")
    offsets = eval(f"[{offsets}]")
    rotations = eval(f"[{rotations}]")
    scales = eval(f"[{scales}]")
    if show > 0:
        pylab.ion()
    assert output != ""
    if isinstance(extensions, str):
        extensions = extensions.split()
    assert len(extensions) == 2
    ds = (
        wds.Dataset(src, handler=wds.warn_and_stop)
        .decode("rgb", handler=wds.warn_and_continue)
        .to_tuple("__key__", *extensions, handler=wds.warn_and_continue)
    )
    count = 0
    with wds.TarWriter(output) as sink:
        for key, page, hocr in ds:
            if skip_missing:
                if page is None:
                    print(key, "page is None", file=sys.stderr)
                    continue
                if hocr is None:
                    print(key, "hocr is None", file=sys.stderr)
                    continue
            assert page is not None, key
            assert hocr is not None, key
            if page.ndim == 3:
                page = np.mean(page, 2)
            page = utils.autoinvert(page, invert)
            print("#", key, "count", count, "shape", page.shape, file=sys.stderr)
            assert isinstance(page, np.ndarray), (key, type(page))
            assert isinstance(hocr, bytes), (key, type(hocr))
            if count >= maxcount:
                print("MAXCOUNT REACHED", file=sys.stderr)
                break
            if np.random.uniform() >= subsample:
                continue
            for scale in scales:
                if count >= maxcount:
                    break
                scale += np.clip(np.random.normal(), -2, 2) * 0.1
                try:
                    pageseg = table_segmentation_target(
                        page, hocr, element=element,  labels=labels, offsets=offsets,
                    )
                    # page, pageseg = augment_segmentation( page, pageseg, rotation=rotations, scale=[scale, scale])
                    patches = patches_of_segmentation(
                        page, pageseg, threshold=2, patchsize=patchsize, n=patches_per_image
                    )
                except ValueError as exn:
                    if ignore_errors:
                        print("===", key, "===")
                        print(exn, file=sys.stderr)
                        continue
                    raise exn
                finished = set()
                for i, (y, x), (img, seg) in patches:
                    if count >= maxcount:
                        break
                    if count % 1000 == 0:
                        print(f"{count}", file=sys.stderr)
                    count += 1
                    assert np.amax(img) < 2.0
                    key_loc = f"{key}@{y},{x}"
                    patch = {
                        "__key__": key_loc,
                        "png": np.clip(img, 0, 1),
                        "seg.png": seg,
                    }
                    if key_loc in finished:
                        print(f"{key}: duplicate rectangle", file=sys.stderr)
                        continue
                    finished.add(key_loc)
                    sink.write(patch)
                    if show > 0 and count % show == 0:
                        pylab.clf()
                        pylab.subplot(121)
                        pylab.imshow(img)
                        pylab.subplot(122)
                        pylab.imshow(seg, vmin=0, vmax=3)
                        pylab.ginput(1, 0.0001)
    print("# wrote", count, "records to", output, file=sys.stderr)


if __name__ == "__main__":
    app()
