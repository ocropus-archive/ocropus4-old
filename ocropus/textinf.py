from itertools import islice

import random
import typer
import numpy as np
import scipy.ndimage as ndi
import webdataset as wds
import torch
from functools import partial
from typing import Optional, List
import matplotlib.pyplot as plt
import editdistance
import re

from . import utils, loading, preinf, textmodels

app = typer.Typer()


class TextRecognizer:
    def __init__(self, mname):
        self.model = loading.load_jit_model(mname)
        self.device = None

    def to(self, device="cpu"):
        self.model.to(device)
        self.device = device

    def cpu(self):
        self.to("cpu")

    def recognize(self, lines: torch.Tensor) -> List[str]:
        assert lines.ndim == 4
        assert lines.shape[1] == 3
        assert lines.min() >= 0 and lines.max() <= 1
        assert lines.shape[2] >= 16 and lines.shape[2] <= 512
        assert lines.shape[3] >= 16 and lines.shape[3] <= 1024
        lines = lines.to(self.device)
        with torch.no_grad():
            self.model.eval()
            result = self.model(lines).softmax(1).cpu()
        seqs = [textmodels.ctc_decode(x) for x in result]
        text = [self.model.decode_str(torch.tensor(x)) for x in seqs]
        return text


default_text_model = "http://storage.googleapis.com/ocropus4-models/text.jit.pt"


@app.command()
def noop():
    pass


def normalize(s):
    return re.sub(r"[^a-zA-Z0-9,.:;/?!]+", "", s).lower()


@app.command()
def recognize(
    fname: str,
    mname: str = default_text_model,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "",
    display: bool = True,
    limit: int = 999999999,
    device: Optional[str] = None,
    binarize: bool = True,
    binmodel: str = preinf.default_jit_model,
    show: float = -1,
    verbose: bool = False,
    correct: float = 1.0,
    wrong: float = 1.0,
    dictionary: str = "",
):
    words = None
    if dictionary:
        words = open(dictionary).read().splitlines()
        words = [w.lower() for w in words]
    device = device or utils.device(device)
    print(f"# device {device}")
    recognizer = TextRecognizer(mname)
    recognizer.to(device)
    dataset = (
        wds.WebDataset(fname)
        .decode("rgb")
        .rename(
            jpg=extensions,
            gt="txt;gt.txt",
        )
    )
    errs = 0
    total = 0
    sink = None
    if output != "":
        sink = wds.TarWriter(output)
    count = 0
    for sample in dataset:
        if count >= limit:
            break
        patch = sample["jpg"]
        patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0)
        text = recognizer.recognize(patch)[0]
        sample["pred.txt"] = text
        err = None
        if "gt" in sample:
            if words is not None:
                w = sample["gt"].strip()
                w = re.sub("[^a-zA-Z0-9']", "", w)
                if len(w) < 3 or w.lower() not in words:
                    continue
            err = editdistance.eval(normalize(text), normalize(sample["gt"]))
            sample["err"] = str(err)
            if err == 0 and random.uniform(0, 1) > correct:
                continue
            elif err > 0 and random.uniform(0, 1) > wrong:
                continue
            errs += err
            total += len(sample["gt"])
        else:
            assert correct >= 1.0 and wrong >= 1.0, "when specifying correct/wrong, gt must be present"
        if verbose:
            print(f"{sample['__key__']} {err} [{sample.get('gt')}] [{text}]")
        if sink:
            sink.write(sample)
        count += 1
    if sink is not None:
        sink.close()
    print(f"{errs/total:.2f} {errs} {total}")


if __name__ == "__main__":
    app()
