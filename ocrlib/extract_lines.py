import sys
import io
import re

import typer
from lxml import html
from matplotlib import pylab
import webdataset as wds
import numpy as np

from . import utils

app = typer.Typer()


def get_text(node):
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s)


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def hocr2images(
    image,
    hocr,
    element="ocrx_word",
    padding=5,
    unicodedammit=False,
    bounds="50,1000,50,200",
    acceptable_size=lambda x: True,
    acceptable_text=lambda x: True,
):
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(hocr, (bytes, str))

    if isinstance(padding, int):
        padding = [padding] * 4
    if unicodedammit:
        from bs4 import UnicodeDammit

        doc = UnicodeDammit(hocr, is_html=True)
        parser = html.HTMLParser(encoding=doc.original_encoding)
        doc = html.document_fromstring(hocr, parser=parser)
    elif isinstance(hocr, bytes):
        doc = html.parse(io.BytesIO(hocr))
    else:
        doc = html.parse(io.StringIO(hocr))

    pages = list(doc.xpath('//*[@class="ocr_page"]'))
    assert len(pages) == 1
    page = pages[0]
    lines = page.xpath("//*[@class='%s']" % element)
    for line in lines:
        bbox = [int(x) for x in get_prop(line, "bbox").split()]
        if padding is not None:
            h, w = image.shape[:2]
            bbox[0] = max(bbox[0] - padding[0], 0)
            bbox[1] = max(bbox[1] - padding[1], 0)
            bbox[2] = min(bbox[2] + padding[2], w)
            bbox[3] = min(bbox[3] + padding[3], h)
        if bbox[0] > bbox[2] or bbox[1] >= bbox[3]:
            continue
        if not acceptable_size(bbox):
            continue
        x0, y0, x1, y1 = bbox
        lineimage = image[y0:y1, x0:x1, ...]
        linetext = get_text(line)
        if acceptable_text is not None and not acceptable_text(linetext):
            continue
        yield lineimage, linetext, bbox


def acceptable_chars(text):
    characters = re.sub(r"\W", r"", text)
    symbols = re.sub(r"\w", r"", text)
    return (
        len(characters) >= 2
        and len(symbols) <= 3
        and len(symbols) <= max(1, len(characters))
    )


def acceptable_words(fname="/usr/share/dict/word"):
    with open(fname) as stream:
        dictionary = set([x.strip().lower() for x in stream.readlines()])

    def f(text):
        core = re.sub(r"\W?(\w+)\W{0,3}", r"\1", text)
        core = core.lower()
        return core in dictionary

    return f


def acceptable_bounds(bounds=(50, 1000, 50, 200), max_aspect=1.1):
    def f(bbox):
        x0, y0, x1, y1 = bbox
        h, w = y1 - y0, x1 - x0
        wlo, whi, hlo, hhi = bounds
        if h < hlo or h > hhi:
            return False
        if w < wlo or w > whi:
            return False
        if h > max_aspect * w:
            return False
        return True

    return f


@app.command()
def hocr2framed(
    src: str,
    output: str = "",
    extensions: str = "page.png;page.jpg;png;jpg;jpeg;JPEG;PNG page.hocr;hocr.html;hocr;HOCR",
    maxcount: int = 9999999999,
    element: str = "ocr_line",
    show: int = 0,
    lightbg: bool = False,
    invert: str = "Auto",
    minelements: int = 2,
):
    """Remove data outside the content frame."""
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
            if page is None:
                print(key, "page is None", file=sys.stderr)
                continue
            if page is None:
                print(key, "hocr is None", file=sys.stderr)
                continue
            page_x0, page_y0, page_x1, page_y1 = 999999999, 999999999, 0, 0
            page = utils.autoinvert(page, invert)
            nelements = 0
            for lineimage, linetext, bbox in hocr2images(
                page,
                hocr,
                element=element,
                padding=5,
            ):
                x0, y0, x1, y1 = bbox
                page_x0 = min(page_x0, x0)
                page_y0 = min(page_y0, y0)
                page_x1 = max(page_x1, x1)
                page_y1 = max(page_y1, y1)
                nelements += 1
            if nelements < minelements:
                print(f"too few instances of {element} found ({nelements})", file=sys.stderr)
                continue
            print("bbox", page_x0, page_y0, page_x1, page_y1, page.shape)
            mask = np.zeros_like(page)
            mask[page_y0:page_y1, page_x0:page_x1, ...] = 1.0
            bgvalue = np.amax(page) if lightbg else np.amin(page)
            npage = np.where(mask, page, bgvalue)
            sink.write({
                "__key__": key,
                "page.jpg": npage,
                "hocr.html": hocr
            })
            if show > 0 and count % show == 0:
                pylab.clf()
                pylab.subplot(121)
                pylab.imshow(page)
                pylab.subplot(122)
                pylab.imshow(npage)
                pylab.ginput(1, 0.0001)
            count += 1
            print("#", key, count, file=sys.stderr)
            if count >= maxcount:
                print("# MAXCOUNT REACHED", file=sys.stderr)
                break
    print("# wrote", count, "records to", output, file=sys.stderr)


@app.command()
def hocr2rec(
    src: str,
    output: str = "",
    extensions: str = "page.png;page.jpg;png;jpg;jpeg;JPEG;PNG page.hocr;hocr.html;hocr;HOCR",
    element="ocrx_word",
    maxcount: int = 9999999999,
    show: int = 0,
    dictionary: str = "NONE",
    bounds: str = "50,1000,50,200",
    invert: str = "Auto",
):
    """Extract recognizable segments from training data.

    The training data is a .tar file containing .page.jpg and .hocr.html files
    (alternative extensions specified by --extensions). For each segment of the
    requested type (usually, ocrx_word or ocr_line), extract the corresponding
    subimage and text. Output a .tar file with each extracted subimages and
    corresponding text. This can be used for training text recognizers.
    """
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
    if dictionary == "NONE":
        acceptable_text = None
    elif dictionary == "":
        acceptable_text = acceptable_chars
    else:
        acceptable_text = acceptable_words(dictionary)
    acceptable_size = acceptable_bounds(list(map(int, bounds.split(","))))
    with wds.TarWriter(output) as sink:
        for key, page, hocr in ds:
            if page is None:
                print(key, "page is None", file=sys.stderr)
                continue
            if page is None:
                print(key, "hocr is None", file=sys.stderr)
                continue
            page = utils.autoinvert(page, invert)
            for lineimage, linetext, bbox in hocr2images(
                page,
                hocr,
                element=element,
                padding=5,
                unicodedammit=False,
                acceptable_text=acceptable_text,
                acceptable_size=acceptable_size,
            ):
                if count >= maxcount:
                    print("MAXCOUNT REACHED", file=sys.stderr)
                    break
                if count >= maxcount:
                    break
                if count % 1000 == 0:
                    print(f"{count}", file=sys.stderr)
                assert np.amax(lineimage) < 2.0
                sample = {
                    "__key__": f"{key}@{count}",
                    "png": np.clip(lineimage, 0, 1),
                    "txt": linetext,
                    "bbox.json": bbox,
                }
                sink.write(sample)
                if show > 0 and count % show == 0:
                    pylab.clf()
                    pylab.imshow(lineimage)
                    pylab.ginput(1, 0.0001)
                count += 1
            print("#", key, count, file=sys.stderr)
            if count >= maxcount:
                print("# MAXCOUNT REACHED", file=sys.stderr)
                break
    print("# wrote", count, "records to", output, file=sys.stderr)


@app.command()
def json2rec():
    raise Exception("unimplemented")


if __name__ == "__main__":
    app()
