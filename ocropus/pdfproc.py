#!/usr/bin/env python

import sys
import os
import glob
import re
import tempfile
import typer
import webdataset as wds


app = typer.Typer()


def getnum(s):
    s = os.path.split(s)[1]
    match = re.search("page-([0-9]+)", s)
    return int(match.group(1))


def iterate_pages(pdfdata):
    with tempfile.TemporaryDirectory() as td:
        src = f"{td}/__document__.pdf"
        with open(src, "wb") as stream:
            stream.write(pdfdata)
        cmd = f"cd {td} && pdftoppm -jpeg -r 300 -hide-annotations {src} page"
        print("#", cmd, file=sys.stderr)
        status = os.system(cmd)
        if status != 0:
            return ValueError("pdf command failed")
        fnames = glob.glob(f"{td}/page-*.jpg")
        fnames = sorted(fnames, key=getnum)
        for fname in fnames:
            yield fname


@app.command()
def renderfile(
    pdfname: str,
    output: str = "",
    withpdf: bool = False,
):
    with open(pdfname, "rb") as stream:
        data = stream.read()
    pdfbase = os.path.splitext(pdfname)[0]
    assert output != ""
    sink = wds.TarWriter(output)
    if withpdf:
        sink.write(
            {
                "__key__": pdfbase,
                "pdf": data,
            }
        )
    for fname in iterate_pages(data):
        base = os.path.split(fname)[1]
        base = os.path.splitext(base)[0]
        sample = {"__key__": f"{pdfbase}/{base}", "jpg": open(fname, "rb").read()}
        sink.write(sample)
    sink.close()


@app.command()
def render(
    input: str,
    output: str = "",
    withpdf: bool = False,
    pdfext: str = "pdf",
):
    ds = wds.WebDataset(input)
    assert output != ""
    sink = wds.TarWriter(output)

    for sample in ds:
        key = sample["__key__"]
        print("===", key, file=sys.stderr)
        if pdfext not in sample:
            print(f"no {pdfext} found", list(sample.keys()), file=sys.stderr)
            continue
        data = sample[pdfext]
        if withpdf:
            sink.write(sample)
        count = 0
        for fname in iterate_pages(data):
            base = os.path.split(fname)[1]
            base = os.path.splitext(base)[0]
            sample = {"__key__": f"{key}/{base}", "jpg": open(fname, "rb").read()}
            sink.write(sample)
            count += 1
        print(count, "pages")
    sink.close()


if __name__ == "__main__":
    app()
