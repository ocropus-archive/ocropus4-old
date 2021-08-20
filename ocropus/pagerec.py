import numpy as np
import typer
import sys
import os
import matplotlib.pyplot as plt
from collections import Counter
import functools
import webdataset as wds
import traceback

from . import ocrorec
from . import ocroseg
from . import wordrec
from . import loading
from .utils import public
from lxml import etree
from lxml.builder import E
from .utils import BBox


Charset = ocrorec.Charset

app = typer.Typer()


def reading_order(lines, highlight=None, binary=None, debug=0):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = np.zeros((len(lines), len(lines)), "B")

    def x_overlaps(u, v):
        return u[1].start < v[1].stop and u[1].stop > v[1].start

    def above(u, v):
        return u[0].start < v[0].start

    def left_of(u, v):
        return u[1].stop < v[1].start

    def separates(w, u, v):
        if w[0].stop < min(u[0].start, v[0].start):
            return 0
        if w[0].start > max(u[0].stop, v[0].stop):
            return 0
        if w[1].start < u[1].stop and w[1].stop > v[1].start:
            return 1

    def center(bbox):
        return (
            np.mean(bbox[0].start, bbox[0].stop),
            np.mean(bbox[1].start, bbox[1].stop),
        )

    if highlight is not None:
        plt.clf()
        plt.title("highlight")
        plt.imshow(binary)
        plt.ginput(1, debug)
    for i, u in enumerate(lines):
        for j, v in enumerate(lines):
            if x_overlaps(u, v):
                if above(u, v):
                    order[i, j] = 1
            else:
                if [w for w in lines if separates(w, u, v)] == []:
                    if left_of(u, v):
                        order[i, j] = 1
            if j == highlight and order[i, j]:
                print((i, j), end=" ")
                y0, x0 = center(lines[i])
                y1, x1 = center(lines[j])
                plt.plot([x0, x1 + 200], [y0, y1])
    if highlight is not None:
        print()
        plt.ginput(1, debug)
    return order


def find(condition):
    "Return the indices where ravel(condition) is true"
    (res,) = np.nonzero(np.ravel(condition))
    return res


def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = np.zeros(n)
    L = []

    def visit(k):
        if visited[k]:
            return
        visited[k] = 1
        for l in find(order[:, k]):
            visit(l)
        L.append(k)

    for k in range(n):
        visit(k)
    return L  # [::-1]


def linebbox(words):
    """Compute a bounding box for the words of a text line."""
    if len(words) == []:
        return 0, 0, 0, 0
    boxes = [BBox(*w["box"]) for w in words]
    return functools.reduce(lambda x, y: x.union(y), boxes).coords()


def html_for_words(words, isolated=False, style="", wordscale=0.6):
    for word in words:
        y0, y1, x0, x1 = word["box"]
        s = word["output"]
        if s == "":
            s = "☒"
        bh = wordscale * (y1 - y0) * wordrec.boxfactor(s)
        style = "" + style
        if isolated:
            style += f"position: absolute; top: {y0}px; left: {x0}px;"
            style += f"font-size: {int(bh)+1}px;"
        yield E.span(
            s + " ",
            Class="ocrx_word",
            title=f"bbox {x0} {y0} {x1} {y1}",
            style=style,
        )


def estimate_lineheight(line):
    return np.median([word["box"][1] - word["box"][0] for word in line])


def html_for_lines(lines, image=None, style="", isolated=True, linescale=0.8):
    for line in lines:
        y0, y1, x0, x1 = linebbox(line)
        bh = linescale * estimate_lineheight(line)
        style = "" + style
        if isolated:
            style += f"position: absolute; top: {y0}px; left: {x0}px;"
            style += f"font-size: {int(bh)+1}px;"
        line_html = E.span(
            "", style=style, Class="ocr_line", title=f"bbox {x0} {y0} {x1} {y1}"
        )
        for word_html in html_for_words(line):
            line_html.append(word_html)

        yield line_html


def html_for_page(key, lines, image=None):
    if image is not None:
        pageheight, pagewidth = image.shape[0], image.shape[1]
    else:
        pageheight = np.amax(w[1] for w in lines) + 10
        pagewidth = np.amax(w[0] for w in lines) + 10
    body = E.div(
        style="position: relative; "
        + f"min-height: {pageheight}px; width: {pagewidth}px;"
        + "border: solid 2px black;"
    )
    base = os.path.basename(key)
    if image is not None:
        body.append(
            E.img(
                src=f"{base}.jpg",
                style="position: absolute; opacity: 0.3; top: 0px; left: 0px;",
                alt="",
            )
        )

    for html in html_for_words(lines[0], isolated=True):
        body.append(html)

    for html in html_for_lines(lines[1:]):
        body.append(html)

    return body


@public
class LineGrouper:
    def __init__(self, model):
        self.seg = ocroseg.Segmenter(model)
        self.patchsize = (256, 2000)
        self.seg.marker_threshold = 0.4
        self.seg.region_threshold = 0.4

    def group(self, words, image):
        self.seg.segment(image)
        linemap = self.seg.segments

        lines = {}

        for word in words:
            y0, y1, x0, x1 = word["box"]
            overlap = Counter(linemap[y0:y1, x0:x1].ravel()).most_common()
            most = overlap[0][0]
            lines.setdefault(most, []).append(word)

        extras = []
        if 0 in lines:
            extras = lines[0]
            del lines[0]

        lines = [sorted(line, key=lambda w: w["box"][2]) for line in lines.values()]
        lines = sorted(lines, key=lambda l: l[0]["box"][0])

        return [extras] + lines


@app.command()
def recognize(
    tarfile: str,
    output: str = "",
    conf: str = "./ocropus4.yaml",
    recmodel: str = "",
    segmodel: str = "",
    lgmodel: str = "",
    extensions: str = "jpg;png;page.jpg;page.png;jpeg",
    segment_type: str = "span",
    maxrec: int = 999999999,
    full_html: str = "",
    debug_rec: bool = False,
    debug_seg: bool = False,
    output_html: bool = True,
    output_json: bool = True,
    output_image: bool = True,
):

    assert recmodel != "", "must give --recmodel argument"
    assert segmodel != "", "must give --segmodel argument"

    def show_page(segmenter):
        plt.clf()
        plt.subplot(121)
        plt.imshow(segmenter.page)
        plt.subplot(122)
        print(segmenter.probs.shape)
        print(np.amin(segmenter.probs), np.amax(segmenter.probs))
        plt.imshow(segmenter.probs[:, :, -3:])
        plt.ginput(1, 100)

    def show_segment(segment, text):
        plt.clf()
        plt.imshow(segment)
        plt.title(text)
        plt.ginput(1, 100)

    pr = wordrec.BasicRecognizer(segment_type=segment_type)
    if debug_rec:
        pr.after_recognition_hook = show_segment
    if debug_seg:
        pr.after_segmentation_hook = show_page
    pr.load_recognizer(recmodel)
    pr.load_segmenter(segmodel)
    print("# starting", file=sys.stderr)
    lgmodel = loading.load_only_model(lgmodel)
    lg = LineGrouper(lgmodel)
    ds = wds.WebDataset(tarfile).decode("l").to_tuple(f"__key__ {extensions}")
    sink = None
    if output != "":
        sink = wds.TarWriter(output)
    fullbody = None
    if full_html != "":
        fullbody = E.body()
    for count, (key, image) in enumerate(ds):
        print("\n===", key, "=" * 40, "\n")
        if count >= maxrec:
            break
        assert isinstance(image, np.ndarray), type(image)
        image = wordrec.normalize_image(image)
        try:
            words = pr.recognize_words(image)
        except Exception as exn:
            traceback.print_exc()
            print(exn, file=sys.stderr)

        lines = lg.group(words, image=image)

        page = html_for_page(key, lines, image=image)

        sample = {
            "__key__": key,
        }

        if output_image:
            sample["jpg"] = image

        if output_json:
            sample["lines.json"] = lines

        if output_html:
            singlepage = E.html(E.title(key), E.body(page))
            sample["words.html"] = etree.tostring(singlepage)

        if fullbody is not None:
            fullbody.append(E.h2(key))
            fullbody.append(page)
            fullbody.append(E.p(style="page-break-after: always;"))

        if sink is not None:
            sink.write(sample)

    if full_html != "":
        fulloutput = E.html(E.title(tarfile), fullbody)
        with open(full_html, "wb") as stream:
            stream.write(etree.tostring(fulloutput))


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
