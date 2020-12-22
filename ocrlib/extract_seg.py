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

import typer

app = typer.Typer()


def get_text(node):
    """Extract the text from a DOM node."""
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s)


def get_prop(node, name):
    """Extract a property from an hOCR DOM node."""
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def center(bbox):
    """Compute the center of a bounding box."""
    x0, y0, x1, y1 = bbox
    return int(np.mean([x0, x1])), int(np.mean([y0, y1]))


def dims(bbox):
    """Compute the w, h of a bounding box."""
    x0, y0, x1, y1 = bbox
    return x1 - x0, y1 - y0


def acceptable(bbox, minw=10, maxw=1000, minh=10, maxh=100):
    """Determine whether a bounding box has an acceptable size."""
    bw, bh = dims(bbox)
    w0, h0, w1, h1 = map(
        float, os.environ.get("acceptable_bboxes", "10 10 1000 100").split()
    )
    return bw >= w0 and bw <= w1 and bh >= h0 and bh <= h1


def marker_segmentation_target_for_bboxes(image, bboxes, inside=0):
    """
    Generate a segmentation target given an image and hOCR segmentation info.
        :param image: page image
        :param bboxes: list of (x0, y0, x1, y1) bounding boxes for marker generation
    """
    fa, fb, fc, fd = map(
        float, os.environ.get("marker_segmentation", "0.4 0.05 0.2 0.0").split()
    )
    h, w = image.shape[:2]
    target = np.zeros((h, w), dtype="uint8")
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        a = int(bh * fa)
        target[y0 - a : y1 + a, x0 - a : x1 + a] = 1
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        b = int(-bh * fb)
        target[y0 - b : y1 + b, x0 - b : x1 + b] = inside
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        xc, yc = center(bbox)
        bw, bh = dims(bbox)
        c = int(bh * fc)
        d = int(bh * fd)
        target[yc - c : yc + c, x0 + d : x1 - d] = 2
    return target


def marker_segmentation_target_for_hocr(image, hocr, element="ocrx_word"):
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
        bbox = [int(x) for x in get_prop(word, "bbox").split()]
        if not acceptable(bbox):
            continue
        bboxes.append(bbox)
    return marker_segmentation_target_for_bboxes(image, bboxes)


def get_any(sample, key, default=None):
    """Given a list of keys, return the first dictionary entry matching a key."""
    if isinstance(key, str):
        key = key.split(";")
    for k in key:
        if k in sample:
            return sample[k]
    return default


def segmentation_patches(
    page,
    hocr,
    degrade=None,
    n=50,
    patchsize=512,
    threshold=20,
    extra={},
    scale=1.0,
    rotation=(0.0, 0.0),
    element="ocrx_word",
):
    """Extract training patches for segmentation."""
    assert page is not None
    assert hocr is not None
    if page.ndim == 3:
        page = np.mean(page, 2)
    if degrade is not None:
        page = degrade(page)
    seg = marker_segmentation_target_for_hocr(page, hocr, element=element)
    if scale != 1.0:
        page = ndi.zoom(page, scale, order=1, mode="nearest")
        seg = ndi.zoom(seg, scale, order=0, mode="constant", cval=0)
    if rotation[1] != 0.0 or rotation[0] != 0.0:
        alpha = np.random.uniform(*rotation)
        page = ndi.rotate(page, alpha, order=1, mode="nearest")
        seg = ndi.rotate(seg, alpha, order=0, mode="constant", cval=0)
    assert isinstance(page, np.ndarray), type(page)
    assert isinstance(seg, np.ndarray), type(seg)
    assert page.dtype == np.float32, page.dtype
    assert seg.dtype in [np.uint8, np.int32, np.int64], seg.dtype
    assert np.amax(seg) <= 2 and np.amin(seg) >= 0
    assert page.ndim == 2
    assert page.shape == seg.shape
    extra["seg"] = seg
    patches = list(
        utils.interesting_patches(
            np.array(seg >= 2, "i"), threshold, [page, seg], r=patchsize, n=n
        )
    )
    for patch in patches:
        yield patch


@app.command()
def hocr2seg(
    src: str,
    output: str = "",
    extensions: str = "page.jpg;page.png;png;jpg;jpeg;JPEG;PNG hocr;HOCR;hocr.html",
    maxcount: int = 9999999999,
    subsample: float = 1.0,
    patches_per_image: int = 50,
    show: int = 0,
    patchsize: int = 512,
    scales: str = "0.7,1.0,1.4",
    randrot: float = 0.0,
    element: str = "ocrx_word",
    skip_missing=True,
    ignore_errors=True,
    invert=False,
    debug=False,
):
    """Extract segmentation patches from src and send them to output."""
    if show > 0:
        pylab.ion()
    assert output != ""
    scales = list(map(float, scales.split(",")))
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
            if debug:
                print("# starting", key, file=sys.stderr)
            if skip_missing:
                if page is None:
                    print(key, "page is None", file=sys.stderr)
                    continue
                if hocr is None:
                    print(key, "hocr is None", file=sys.stderr)
                    continue
            assert page is not None, key
            assert hocr is not None, key
            if invert:
                page = np.amax(page) - page
            print("#", key, "count", count, "maxcount", maxcount, file=sys.stderr)
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
                    patches = segmentation_patches(
                        page,
                        hocr,
                        n=patches_per_image,
                        element=element,
                        scale=scale,
                        rotation=(-randrot, randrot),
                    )
                except ValueError as exn:
                    if ignore_errors:
                        print("===", key, "===")
                        print(exn, file=sys.stderr)
                        continue
                    raise exn
                for i, (y, x), (img, seg) in patches:
                    if count >= maxcount:
                        break
                    if count % 1000 == 0:
                        print(f"{count}", file=sys.stderr)
                    count += 1
                    assert np.amax(img) < 2.0
                    patch = {
                        "__key__": f"{key}@{y},{x}",
                        "png": np.clip(img, 0, 1),
                        "seg.png": seg,
                    }
                    sink.write(patch)
                    if show > 0 and count % show == 0:
                        pylab.clf()
                        pylab.subplot(121)
                        pylab.imshow(img)
                        pylab.subplot(122)
                        pylab.imshow(seg)
                        pylab.ginput(1, 0.0001)
    print("# wrote", count, "records to", output, file=sys.stderr)


@app.command()
def json2seg():
    raise Exception("unimplemented")


if __name__ == "__main__":
    app()
