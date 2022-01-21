from itertools import islice
import sys
import re
import os
import base64
import io

import imageio
import typer
import numpy as np
import scipy.ndimage as ndi
import webdataset as wds
import torch
from functools import partial
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from lxml import etree
from lxml.builder import E

from . import utils, loading, preinf, textinf, seginf, textdata
from dataclasses import dataclass

app = typer.Typer()


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


def cleantext(s):
    s = re.sub(r"[\000-\037]", "~", s)
    return s


def html_for_words(key, words, image=None, wordscale=0.6, max_height=80, count=0, imdir="_images", fontsize=12):
    if image is not None:
        assert isinstance(image, torch.Tensor)
        pageheight, pagewidth = image.shape[-2:]
    else:
        pageheight = np.amax([w["bbox"][3] for w in words]) + 10
        pagewidth = np.amax([w["bbox"][2] for w in words]) + 10
    body = E.div(
        style="position: relative; "
        + f"min-height: {pageheight}px; width: {pagewidth}px;"
        + "border: solid 2px black;"
        + "page-break-after: always;"
    )
    base = os.path.basename(key)
    if image is not None:
        body.append(
            E.img(
                src=f"{os.path.join(imdir, base)}.jpg",
                style="position: absolute; opacity: 0.3; top: 0px; left: 0px;",
                alt=key,
            )
        )
        if imdir != "":
            os.makedirs(imdir, exist_ok=True)
            image = (image.cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            imageio.imsave(f"{os.path.join(imdir, base)}.jpg", image)

    for word in words:
        y0, y1, x0, x1 = word["bbox"]
        s = cleantext(word["text"])
        if s == "":
            s = "☒"
        bh = wordscale * (y1 - y0) * boxfactor(s)
        if bh > max_height:
            bh = max_height
        color = "red"
        fs = fontsize if fontsize > 0 else int(bh) + 1
        style = f"font-size: {fs}px;"
        style += "font-weight: bold;"
        style += f"width: {x1 - x0}px;"
        style += f"height: {y1 - y0}px;"
        style2 = f"position: absolute; top: {y0}px; left: {x0}px;"
        style2 += f"color: {color};"
        style2 += f"border: 1px solid {color};"
        body.append(
            E.div(
                E.div(
                    s + " ",
                    Class="ocrx_word",
                    title=f"bbox {x0} {y0} {x1} {y1}",
                    style=style,
                ),
                style=style2,
            )
        )
    return body


def dhtml_for_words(key, words, image=None, wordscale=0.6, max_height=80, count=0, imdir="_images"):
    body = E.div(style="page-break-after: always;")
    body.append(E.h1(key))
    if image is not None:
        image = (image.clip(0, 1) * 255).type(torch.uint8).numpy().transpose(1, 2, 0)
        with io.BytesIO() as f:
            imageio.imsave(f, image, format="JPEG")
            f.seek(0)
            body.append(
                E.img(
                    src=f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}",
                    # style="position: absolute; opacity: 0.3; top: 0px; left: 0px;",
                    alt=key,
                    width="800px",
                    border="1px solid black",
                ),
            )

    body.append(E.p())

    table = E.table(style="border: solid 2px black;")
    for i in range(0, len(words), 5):
        row = E.tr()
        table.append(row)
        for j in range(5):
            if i + j >= len(words):
                break
            word = words[i+j]
            y0, y1, x0, x1 = word["bbox"]
            scale = min([1.0, 40.0 / (y1 - y0), 200.0 / (x1 - x0)]) 
            width = (x1 - x0) * scale
            height = (y1 - y0) * scale
            image = word["image"]
            assert isinstance(image, torch.Tensor)
            assert image.shape[0] == 3
            assert image.dtype == torch.float32
            image = (255 - image.clip(0, 1) * 255).type(torch.uint8).numpy().transpose(1, 2, 0)
            with io.BytesIO() as f:
                imageio.imsave(f, image, format="JPEG")
                f.seek(0)
                img = E.img(
                    src=f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}",
                    # style="position: absolute; opacity: 0.3; top: 0px; left: 0px;",
                    alt=key,
                    height=f"{height}px",
                    width=f"{width}px",
                    border="1px solid black",
                )
            text = re.sub(r"[\000-\037]", "~", word["text"])
            txt = E.font(E.text(text), size="+1", color="blue")
            row.append(E.td(img, E.br(), txt))
    body.append(table)
    return body


class PageRecognizer:
    def __init__(
        self,
        binmname: str = preinf.default_jit_model,
        segmname: str = seginf.default_seg_model,
        textmname: str = textinf.default_text_model,
        binarize: bool = True,
        padx: int = 3,
        pady: int = 3,
        device: str = None,
    ):
        self.device = device or utils.device(device)
        self.binmname = binmname
        self.segmname = segmname
        self.textmname = textmname
        self.binarize = binarize
        self.padx = padx
        self.pady = pady
        self.binarizer = preinf.Binarizer(model=binmname, device=self.device)
        self.segmenter = seginf.Segmenter(segmname, device=self.device)
        self.recognizer = textinf.TextRecognizer(textmname)
        self.preprocessor = textdata.WordPreprocessor(augment=textdata.augment_none)

    def cpu(self):
        self.recognizer.cpu()

    def recognize(self, image):
        padx, pady = self.padx, self.pady
        self.binary = binary = self.binarizer.binarize(image)
        assert binary.dtype == torch.float32
        rgbbinary = torch.stack([binary] * 3)
        assert rgbbinary.dtype == torch.float32
        result = self.segmenter.segment(rgbbinary)
        self.recognizer.to(self.device)
        h, w = image.shape[-2:]
        for y0, y1, x0, x1 in result:
            if y1 - y0 < 16 or x1 - x0 < 16:
                continue
            if y1 - y0 >= 512 or x1 - x0 >= 1024:
                continue
            y0, y1 = max(0, y0 - pady), min(h, y1 + padx)
            x0, x1 = max(0, x0 - padx), min(w, x1 + pady)
            patch = 1.0 - binary[..., y0:y1, x0:x1]
            assert patch.ndim == 2, patch.shape
            assert patch.dtype == torch.float32, patch.dtype
            # print(">", patch.shape, patch.mean())
            result = self.preprocessor.preprocess((patch, None))
            if result is None:
                continue
            patch, _ = result
            # print("<", patch.shape, patch.mean())
            assert patch.ndim == 3, patch.shape
            assert patch.dtype == torch.float32, patch.dtype
            assert patch.shape[0] == 3, patch.shape
            patch = patch.unsqueeze(0)
            orig = image[..., y0:y1, x0:x1]
            recognized = self.recognizer.recognize(patch)[0]
            yield dict(text=recognized, bbox=(y0, y1, x0, x1), image=patch[0], orig=orig)
        self.recognizer.cpu()


@app.command()
def page2words(
    fname: str,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "/dev/null",
    display: bool = True,
    limit: int = 999999999,
    device: Optional[str] = None,
    binarize: bool = True,
    binmname: str = "none",
    segmname: str = seginf.default_seg_model,
    textmname: str = textinf.default_text_model,
    show: float = -1,
    padx: int = 3,
    pady: int = 3,
    orig: bool = False,
):
    print(f"# device {device}")

    dataset = wds.WebDataset(fname).decode("torchrgb")
    print("# dataset", next(iter(dataset)).keys())

    if show >= 0.0:
        plt.ion()

    if binmname == "default":
        binmname = preinf.default_jit_model

    engine = PageRecognizer(
        binmname=binmname,
        segmname=segmname,
        textmname=textmname,
        binarize=binarize,
        padx=padx,
        pady=pady,
        device=device,
    )

    sink = wds.TarWriter(output)

    for sample in islice(dataset, 0, limit):
        print(f"# {sample['__key__']}")
        image = wds.getfirst(sample, extensions)
        count = 0

        for word in engine.recognize(image):
            count += 1
            if show >= 0.0:
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.imshow(word["orig"].numpy().transpose(1, 2, 0))
                plt.subplot(1, 2, 2)
                plt.imshow(1 - word.image.numpy().transpose(1, 2, 0))
                plt.title(word["text"])
                plt.ginput(1, show)
            key = f"{sample['__key__']}/{count:04d}"
            osample = dict(
                __key__=key,
                png=word.image.numpy().transpose(1, 2, 0),
                text=word["text"],
                json=dict(
                    bbox=word["bbox"],
                    text=word["text"],
                ),
            )
            sink.write(osample)
        engine.cpu()

    sink.close()


@app.command()
def page2html(
    fname: str,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    html: str = "",
    dhtml: str = "",
    output: str = "/dev/null",
    display: bool = True,
    limit: int = 999999999,
    device: Optional[str] = None,
    binarize: bool = True,
    binmname: str = "none",
    segmname: str = seginf.default_seg_model,
    textmname: str = textinf.default_text_model,
    show: float = -1,
    padx: int = 3,
    pady: int = 3,
    orig: bool = False,
    fontsize: int = 12,
):
    print(f"# device {device}")

    dataset = wds.WebDataset(fname).decode("torchrgb")
    print("# dataset", next(iter(dataset)).keys())

    if show >= 0.0:
        plt.ion()

    if binmname == "default":
        binmname = preinf.default_jit_model

    engine = PageRecognizer(
        binmname=binmname,
        segmname=segmname,
        textmname=textmname,
        binarize=binarize,
        padx=padx,
        pady=pady,
        device=device,
    )

    sink = wds.TarWriter(output)

    if html != "":
        fullbody = E.body()
    else:
        fullbody = None

    if dhtml != "":
        debugbody = E.body()
    else:
        debugbody = None

    for count, sample in enumerate(islice(dataset, 0, limit)):
        print(f"# {sample['__key__']}")
        image = wds.getfirst(sample, extensions)

        words = list(engine.recognize(image))
        key = sample["__key__"]
        page = html_for_words(key, words, image=image, count=count, fontsize=fontsize)
        osample = {
            "__key__": key,
            "jpg": torch.stack([engine.binary] * 3).numpy().transpose(1, 2, 0),
            "html": etree.tostring(page),
        }
        sink.write(osample)

        if fullbody is not None:
            # fullbody.append(E.h2(key))
            fullbody.append(page)
            # fullbody.append(E.p(style="page-break-after: always;"))

        if debugbody is not None:
            dpage = dhtml_for_words(key, words, image=image, count=count)
            debugbody.append(dpage)

    sink.close()

    if fullbody is not None:
        fulloutput = E.html(E.title(fname), fullbody)
        with open(html, "wb") as stream:
            stream.write(etree.tostring(fulloutput, pretty_print=True))


    if debugbody is not None:
        debugoutput = E.html(E.title(fname), debugbody)
        with open(dhtml, "wb") as stream:
            stream.write(etree.tostring(debugoutput, pretty_print=True))


if __name__ == "__main__":
    app()
