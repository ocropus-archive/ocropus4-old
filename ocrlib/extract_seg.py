import io
import re
import sys

import numpy as np
import webdataset as wds
from lxml import etree
from matplotlib import pylab
import scipy.ndimage as ndi

import ocrlib.patches
from . import utils
from .utils import useopt

import typer


# FIXME move this into the function, make it a command line argument
# FIXME ditto for confidence

# debug = int(os.environ.get("EXTRACT_SEG_DEBUG", "0"))
# acceptable_bboxes = list(
#    map(float, os.environ.get("EXTRACT_SEG_ACCEPTABLE", "5 5 8000 8000").split())
# )
# max_aspect = 1.0

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


def get_bbox(node):
    bbox = get_prop(node, "bbox")
    if bbox is not None:
        return [int(x) for x in bbox.split()]
    else:
        return 0, 0, -1, -1


def center(bbox):
    """Compute the center of a bounding box."""
    x0, y0, x1, y1 = bbox
    return int(np.mean([x0, x1])), int(np.mean([y0, y1]))


def dims(bbox):
    """Compute the w, h of a bounding box."""
    x0, y0, x1, y1 = bbox
    return x1 - x0, y1 - y0


def marker_segmentation_target_for_bboxes(
    image,
    bboxes,
    labels=[1, 0, 2],
    sep_margin=0.4,
    region_margin_y=-0.05,
    region_margin_x=-0.1,
    center_margin_y=-0.2,
    center_margin_x=-0.2,
    min_center_y=4,
    min_center_x_margin=6,
):
    """
    Generate a segmentation target given an image and target bounding boxes.
    This generates a central marker and a separator for each bounding box.
        :param image: page image
        :param bboxes: list of (x0, y0, x1, y1) bounding boxes for marker generation
        :param labels: list of labels for marker, inside, separator (default: [1, 0, 2])
    """

    # print(labels); raise Exception()
    h, w = image.shape[:2]
    target = np.zeros((h, w), dtype="uint8")
    bboxes = sorted(bboxes, key=lambda b: -dims(b)[0])
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        a = int(min(bw, bh) * sep_margin)
        target[y0 - a : y1 + a, x0 - a : x1 + a] = labels[0]
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        bw, bh = dims(bbox)
        by = int(min(bw, bh) * region_margin_y)
        bx = int(min(bw, bh) * region_margin_x)
        target[y0 - by : y1 + by, x0 - bx : x1 + bx] = labels[1]
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        _, yc = center(bbox)
        bw, bh = dims(bbox)
        c = int(min(bw, bh) * center_margin_y)
        d = min(int(min(bw, bh) * center_margin_x), -min_center_x_margin)
        assert abs(c) < bh
        assert abs(d) < bw
        # print(c, d); raise Exception()
        target[y0 - c : y0 + c, x0 - c : x1 + c] = labels[2]
        target[yc - min_center_y : yc + min_center_y, x0 - c : x1 + c] = labels[2]
    return target


@useopt
def mask_with_none(image, bboxes):
    return image


@useopt
def mask_with_bbox(image, bboxes, background=0):
    if len(bboxes) == 0:
        return image
    bboxes = np.array(bboxes, dtype=int)
    x0, y0 = np.amin(bboxes[:, :2], 0)
    x1, y1 = np.amax(bboxes[:, 2:], 0)
    mask = np.zeros_like(image)
    mask[y0:y1, x0:x1] = 1
    image = np.where(mask, image, background)
    return image


@useopt
def mask_with_boxes(image, bboxes, background=0):
    mask = np.zeros_like(image)
    for x0, y0, x1, y1 in bboxes:
        mask[y0:y1, x0:x1] = 1
    image = np.where(mask, image, background)
    return image


@useopt
def check_acceptable_none(bbox):
    return True


@useopt
def check_acceptable_word(bbox, minw=10, maxw=500, minh=10, maxh=100, max_aspect=1.0):
    """Determine whether a bounding box has an acceptable size."""
    bw, bh = dims(bbox)
    aspect = bh / max(float(bw), 0.001)
    return (
        aspect <= max_aspect and bw >= minw and bw <= maxw and bh >= minh and bh <= maxh
    )


@useopt
def check_acceptable_line(bbox, minw=10, maxw=3000, minh=10, maxh=200, max_aspect=1.0):
    """Determine whether a bounding box has an acceptable size."""
    bw, bh = dims(bbox)
    aspect = bh / max(float(bw), 0.001)
    return (
        aspect <= max_aspect and bw >= minw and bw <= maxw and bh >= minh and bh <= maxh
    )


@useopt
def check_text_none(b):
    return True


@useopt
def check_text_word(b, char_height=1.0, min_char_height=0.2):
    s = get_text(b)
    if " " in s:
        return False
    x0, y0, x1, y1 = get_bbox(b)
    est_min_width = len(s) * (y1 - y0) * min_char_height
    est_max_width = len(s) * (y1 - y0) * char_height
    actual_width = x1 - x0
    # print(est_max_width, actual_width, repr(s))
    if actual_width < est_min_width or actual_width > est_max_width:
        return False
    return True


@useopt
def check_text_line(b, char_height=1.0, min_char_height=0.2):
    s = get_text(b)
    x0, y0, x1, y1 = get_bbox(b)
    est_min_width = len(s) * (y1 - y0) * min_char_height
    est_max_width = len(s) * (y1 - y0) * char_height
    actual_width = x1 - x0
    # print(est_max_width, actual_width, repr(s))
    if actual_width < est_min_width or actual_width > est_max_width:
        return False
    return True


def bboxes_for_hocr(
    image,
    hocr,
    acceptable_conf=50,
    element="ocrx_word",
    confidence_prop="x_wconf",
    check_acceptable=check_acceptable_word,
    check_text=check_text_word,
    max_bad_text_frac=0.2,
):
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
    bad_conf = 0
    no_conf = 0
    count = 0
    bad_text = 0
    for word in page.xpath(f"//*[@class='{element}']"):
        count += 1
        if not check_text(word):
            bad_text += 1
            continue
        if acceptable_conf >= 0:
            conf = get_prop(word, confidence_prop)
            if conf is not None:
                conf = float(conf)
                if conf < acceptable_conf:
                    bad_conf += 1
                    continue
            else:
                no_conf += 1
        bbox = [int(x) for x in get_prop(word, "bbox").split()]
        if not check_acceptable(bbox):
            continue
        bboxes.append(bbox)
    if bad_text >= max_bad_text_frac * len(bboxes):
        print("# too many bad bounding boxes on this page")
        return []
    print(
        f"# {bad_conf} bad bboxes, {no_conf} no confidence, {len(bboxes)} good, {bad_text} bad text, {count} total"
    )
    return bboxes


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
    minmark=0,
    mask=(lambda image, bboxes: image),
    labels=[1, 2, 3],
    params="",
    fix_boxes=False,
    check_acceptable=check_acceptable_word,
    check_text=check_text_word,
):
    """Extract training patches for segmentation."""
    assert page is not None
    assert hocr is not None
    if page.ndim == 3:
        page = np.mean(page, 2)
    bboxes = bboxes_for_hocr(
        page,
        hocr,
        element=element,
        check_text=check_text,
        check_acceptable=check_acceptable,
    )
    if fix_boxes:
        binary = page > np.mean([np.amax(page), np.amin(page)])
        bboxes = utils.fix_bounding_boxes(binary, bboxes)
    page = mask(page, bboxes)
    if degrade is not None:
        page = degrade(page)
    kw = eval(f"dict({params})")
    seg = marker_segmentation_target_for_bboxes(page, bboxes, labels=labels, **kw)
    if np.sum(seg) <= minmark:
        print(f"didn't get any {element}", file=sys.stderr)
        return
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
    assert np.amax(seg) <= np.amax(labels) and np.amin(seg) >= 0
    assert page.ndim == 2
    assert page.shape == seg.shape
    extra["seg"] = seg
    patches = list(
        ocrlib.patches.interesting_patches(
            np.array(seg >= 2, "i"), threshold, [page, seg], r=patchsize, n=n
        )
    )
    print("# interesting patches", len(patches))
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
    debug=False,
    invert: str = "Auto",
    mask: str = "bbox",
    labels: str = "1, 2, 3",
    fix_boxes: bool = False,
    check: str = "word",
):
    """Extract segmentation patches from src and send them to output."""
    global acceptable_bboxes
    check_acceptable = eval(f"check_acceptable_{check}")
    check_text = eval(f"check_text_{check}")
    labels = eval(f"[{labels}]")
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
            page = utils.autoinvert(page, invert)
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
                        mask=eval(f"mask_with_{mask}"),
                        labels=labels,
                        fix_boxes=fix_boxes,
                        check_text=check_text,
                        check_acceptable=check_acceptable,
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
                        pylab.imshow(seg)
                        pylab.ginput(1, 0.0001)
    print("# wrote", count, "records to", output, file=sys.stderr)


@app.command()
def json2seg():
    raise Exception("unimplemented")


if __name__ == "__main__":
    app()
