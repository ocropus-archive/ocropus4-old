from itertools import islice

import typer
import numpy as np
import scipy.ndimage as ndi
import webdataset as wds
import torch
from functools import partial
from typing import Optional, List
import matplotlib.pyplot as plt

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
def recognize(
    fname: str,
    model: str = default_text_model,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "/dev/null",
    display: bool = True,
    limit: int = 999999999,
    device: Optional[str] = None,
    binarize: bool = True,
    binmodel: str = preinf.default_jit_model,
    show: float = -1,
):
    device = device or utils.device(device)
    print(f"# device {device}")


if __name__ == "__main__":
    app()
