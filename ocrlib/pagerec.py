from typing import List
import sys
import traceback
import re

import os.path
import numpy as np
import PIL
import typer
import scipy.ndimage as ndi
import functools
import matplotlib.pyplot as plt
import yaml
import json
from lxml import etree
from lxml.builder import E

import webdataset as wds

from ocrlib.utils import BBox
from . import ocroline, ocroseg, loading

Charset = ocroline.Charset

app = typer.Typer()


def get_config(conf):
    """Load a config file."""
    if isinstance(conf, str):
        with open(conf) as stream:
            conf = yaml.load(stream, Loader=yaml.FullLoader)
    return conf


def python_to_json(obj):
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (tuple, list)):
        return [python_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: python_to_json(v) for k, v in obj.items()}
    raise ValueError("bad type in fix4json", type(obj), obj)


def sortwords(lst):
    """Sort the words in a text line by x coordinate."""
    return sorted(lst, key=lambda w: w["box"][2])


def linebbox(words):
    """Compute a bounding box for the words of a text line."""
    if len(words) == []:
        return 0, 0, 0, 0
    boxes = [BBox(*w["box"]) for w in words]
    return functools.reduce(lambda x, y: x.union(y), boxes).coords()


def goodsize(segment):
    h, w = segment.shape[:2]
    return (
        h > ocroline.min_h
        and h < ocroline.max_h
        and w > ocroline.min_w
        and w < ocroline.max_w
    )


class BasicRecognizer:
    """A wrapper for the text recognizer and segmenters.

    This performs three steps:

    - segment the page image into words or other small segments
    - recognize each word
    - (optionally) group words into lines using a separate marker segmenter
    - (optionally) perform a topological sort for minimal reading order
    """

    def __init__(self, segment_type="span"):
        self.segmenter = None
        self.lineseg = None
        self.recognizer = None
        self.segment_type = segment_type
        self.after_segmentation_hook = lambda x: None
        self.after_recognition_hook = lambda x, y: None

    def load_config(self, conf):
        conf = get_config(conf)
        models = conf["models"]
        self.load_segmenter(models["segmenter"])
        self.load_recognizer(models["recognizer"])
        # if "lineseg" in models:
        #     self.load_lineseg(models["lineseg"])

    def load_segmenter(self, fname):
        print(f"# loading segmenter {fname}", file=sys.stderr)
        model = loading.load_only_model(fname)
        self.segmenter = ocroseg.Segmenter(model)
        self.segmenter.activate(False)

    def load_recognizer(self, fname):
        print(f"# loading recognizer {fname}", file=sys.stderr)
        model = loading.load_only_model(fname)
        self.recognizer = ocroline.LineRec(model)
        self.recognizer.activate(False)

    def run_recognizers(self, image):
        print("# segmenting", file=sys.stderr)
        self.segmenter.activate(True)
        self.boxes = self.segmenter.segment(image)
        self.segmenter.activate(False)
        self.segments = ocroseg.extract_boxes(image, self.boxes)

        self.after_segmentation_hook(self.segmenter)

        print("# recognizing", file=sys.stderr)
        self.outputs = []
        self.recognizer.activate(True)
        for segment in self.segments:
            if not goodsize(segment):
                self.outputs.append("")
                continue
            text = self.recognizer.recognize(segment)
            self.outputs.append(text)
            self.after_recognition_hook(segment, text)
        self.recognizer.activate(False)

    def recognize_words(self, image, return_segments=False):
        self.run_recognizers(image)
        result = []
        for i in range(len(self.boxes)):
            r = dict(
                type=self.segment_type,
                box=self.boxes[i],
                output=self.outputs[i],
            )
            if return_segments:
                r["image"] = self.segments[i]
            result.append(r)
        return result


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


def trivial_grouper(objects):
    for obj in objects:
        if len(obj["labels"]) == 0:
            obj["group"] = -1
        else:
            obj["group"] = np.amax(list(obj["labels"]))
    groups = {}
    for obj in objects:
        groups.setdefault(obj["group"], []).append(obj)
    extra = groups[-1]
    del groups[-1]
    return list(groups.values()), extra


class PageRecognizer:
    """A full page recognizer. This calls WordRecognizer for a list
    of raw words on the page, then uses a line segmenter just for
    grouping those words into lines. Reading order is determined
    by topologically sorting the lines using some heuristic constraints
    as a partial order. This is just a quick-and-dirty page recognizer
    and will be supplemented later by text block segmentation and boundary
    noise removal.
    """

    def __init__(self):
        self.wordrec = BasicRecognizer()
        self.lineseg = ocroseg.Segmenter(
            smooth=(1, 4), get_targets=lambda classes: classes == 2
        )

    def load_config(self, conf):
        conf = get_config(conf)
        self.wordrec.load_config(conf)
        self.wordrec.lineseg = None  # just for testing

        fname = conf["models"]["lineseg"]
        print(f"# loading line segmenter {fname}", file=sys.stderr)
        self.lineseg.load_from_save(fname)
        self.lineseg.activate(False)
        self.lineseg_type = "line"

        fname = conf["models"].get("blockseg")
        self.bseg = None
        if fname is not None:
            print(f"# loading block segmenter {fname}", file=sys.stderr)
            self.bseg = ocroseg.Segmenter()
            self.bseg.load_from_save(fname)
            self.bseg.activate(False)
            self.bseg_type = "blocK"

    def recognize_page(self, image, return_segments=False, do_topsort=True):

        words = self.wordrec.recognize_words(image, return_segments=return_segments)

        print("# running line segmenter")
        self.lineseg.activate(True)
        self.lineseg.segment(image)
        self.lineseg.activate(False)

        print("# running block segmenter")
        if self.bseg is not None:
            self.bseg.activate(True)
            self.bseg.segment(image)
            self.bseg.activate(False)

        # line segmentation

        labels, n = ndi.label(self.lineseg.segments)
        for word in words:
            y0, y1, x0, x1 = word["box"]
            word["labels"] = set(np.unique(labels[y0:y1, x0:x1])) - set([0])
        lines, extra = trivial_grouper(words)
        extra = [l for l in extra if l["output"] != ""]
        lines = [
            dict(id=k, words=sortwords(line), box=linebbox(line))
            for k, line in enumerate(lines)
        ]
        if do_topsort:
            boxes = [BBox(*line["box"]) for line in lines]
            constraints = reading_order(boxes)
            order = topsort(constraints)
            lines = [lines[i] for i in order]

        result = dict(extra=extra, words=words, lines=lines)
        return result


def load_image(fname):
    image = PIL.Image.open(fname).convert("L")
    image = np.asarray(image)
    assert np.amin(image) > 3
    image = image / 255.0
    return image


def normalize_image(image):
    image -= np.amin(image)
    image = image / float(np.amax(image))
    image = 1.0 - image
    return image


@app.command()
def recognize_image(
    fnames: List[str],
    conf: str = "./ocropus4.yaml",
    extensions: str = "jpg;png;page.jpg;page.png",
    format: str = "text",
):
    """A command for recognizing images given directly on the command line."""
    assert len(fnames) > 0
    with open(conf) as stream:
        conf = yaml.load(stream, Loader=yaml.FullLoader)
    pr = PageRecognizer()
    pr.load_config(conf)

    for fname in fnames:
        image = load_image(fname)
        image = normalize_image(image)
        result = pr.recognize_page(image)
        if format == "json":
            result = python_to_json(result)
            print(json.dumps(result, indent=4))
        else:
            print_raw_text(result)


def print_raw_text(result):
    for line in result.get("lines"):
        text = [w["output"] for w in line["words"]]
        print(" ".join(text))


def descenders(s):
    return len(set("gjpqy").intersection(set(s))) > 0


def ascenders(s):
    return len(set("bdfhijklt").intersection(set(s))) > 0 or re.search(r"[A-Z]", s)


def boxfactor(s):
    if ascenders(s) and descenders(s):
        return 0.75
    elif ascenders(s):
        return 1.0
    elif descenders(s):
        return 1.0
    else:
        return 1.5


def html_for_words(key, words, image=None, wordscale=0.6):
    if image is not None:
        pageheight, pagewidth = image.shape[0], image.shape[1]
    else:
        pageheight = np.amax(w[1] for w in words) + 10
        pagewidth = np.amax(w[0] for w in words) + 10
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
                alt=key,
            )
        )
    for word in words:
        y0, y1, x0, x1 = word["box"]
        s = word["output"]
        if s == "":
            s = "☒"
        bh = wordscale * (y1 - y0) * boxfactor(s)
        style = f"position: absolute; top: {y0}px; left: {x0}px;"
        style += f"font-size: {int(bh)+1}px;"
        style += "font-weight: bold;"
        style += "color: red;"
        body.append(
            E.span(
                s + " ",
                Class="ocrx_word",
                title=f"bbox {x0} {y0} {x1} {y1}",
                style=style,
            )
        )
    return body


@app.command()
def recognize_segments(
    tarfile: str,
    output: str = "",
    conf: str = "./ocropus4.yaml",
    recmodel: str = "",
    segmodel: str = "",
    extensions: str = "jpg;png;page.jpg;page.png",
    segment_type: str = "span",
    maxrec: int = 999999999,
    full_html: str = "",
    debug_rec: bool = False,
    debug_seg: bool = False,
    debug_out: bool = False,
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

    pr = BasicRecognizer(segment_type=segment_type)
    if debug_rec:
        pr.after_recognition_hook = show_segment
    if debug_seg:
        pr.after_segmentation_hook = show_page
    pr.load_recognizer(recmodel)
    pr.load_segmenter(segmodel)
    print("# starting", file=sys.stderr)
    ds = wds.Dataset(tarfile).decode("l").to_tuple(f"__key__ {extensions}")
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
        image = normalize_image(image)
        try:
            result = pr.recognize_words(image)
        except Exception as exn:
            traceback.print_exc()
            print(exn, file=sys.stderr)
        for s in result[:200]:
            print(s["output"], end=" ")
        print("\n")
        sample = {
            "__key__": key,
        }

        if output_image:
            sample["jpg"] = image

        if output_json:
            sample["words.json"] = result

        page = html_for_words(key, result, image=image)

        if output_html:
            singlepage = E.html(E.title(key), E.body(page))
            sample["words.html"] = etree.tostring(singlepage)

        if fullbody is not None:
            fullbody.append(E.h2(key))
            fullbody.append(page)
            fullbody.append(E.p(style="page-break-after: always;"))

        if sink is not None:
            sink.write(sample)

        if debug_out:
            plt.ion()
            plt.clf()
            plt.imshow(image * 0.3 / np.amax(image), vmin=0.0, vmax=1.0)
            ax = plt.gca()
            for word in result:
                y0, y1, x0, x1 = word["box"]
                s = word["output"]
                ax.annotate(
                    s, xy=(x0, y1), xycoords="data", color="red", fontweight="bold"
                )
            plt.ginput(1, 1000)

    if full_html != "":
        fulloutput = E.html(E.title(tarfile), fullbody)
        with open(full_html, "wb") as stream:
            stream.write(etree.tostring(fulloutput))


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
