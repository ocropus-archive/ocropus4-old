"""Text recognition."""

import time
import warnings
import random, re
import os.path
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import signal

import pytorch_lightning as pl
import torch
import torch.jit
import webdataset as wds
from scipy import ndimage as ndi
from torch.utils.data import DataLoader
from urllib.parse import urljoin
from dataclasses import dataclass, field

from . import confparse, degrade, jittable, utils

import typer

app = typer.Typer()


def identity(x: Any) -> Any:
    """Identity function."""
    return x


def datawarn(*args):
    if int(os.environ.get("OCROPUS_DATA_WARNINGS", "0")):
        warnings.warn(*args)
    return


min_w, min_h, max_w, max_h = 15, 15, 4000, 200


def collate4ocr(samples: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    """Collate OCR images and text into tensor/list batch.

    Args:
        samples (List[Tuple[Tensor, str]]): [description]

    Returns:
        Tuple[Tensor, List[str]]: batch of images and text
    """
    images, seqs = zip(*samples)
    images = jittable.stack_images(images)
    return images, seqs


@utils.useopt
def augment_none(image: torch.Tensor) -> torch.Tensor:
    """Perform no augmentation.

    Args:
        image (torch.Tensor): input image

    Returns:
        torch.Tensor: unaugmented output
    """
    result = utils.as_torchimage(image)
    if result.shape[0] == 1:
        result = result.expand(3, -1, -1)
    return result


@utils.useopt
def augment_transform(timage: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using geometric transformations and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): input image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: augmented image
    """
    image = utils.as_npimage(timage)
    if image.mean() > 0.5:
        image = 1.0 - image
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.random_transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    result = utils.as_torchimage(image)
    return result


@utils.useopt
def augment_distort(timage: torch.Tensor, p: float = 0.5, maxh=100.0) -> torch.Tensor:
    """Augment image using distortions and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): original image
        p (float, optional): probability of binarization. Defaults to 0.5.
        p (float, optional): probability of inverting the image. Defaults to 0.2.

    Returns:
        [type]: augmented image
    """
    image = utils.as_npimage(timage)
    image = image.mean(axis=2)
    if image.mean() > 0.5:
        image = 1.0 - image
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > maxh:
        image = ndi.zoom(image, float(maxh) / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.random_transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        (image,) = degrade.distort_all(image, sigma=(0.5, 4.0), maxdelta=(0.1, 2.5))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    result = utils.as_torchimage(image)
    return result


###
### Text-Related
###


def fixquotes(s: str) -> str:
    """Replace unicode quotes with ascii ones.

    Args:
        s (str): unicode string

    Returns:
        str: unicode string with ascii quotes
    """
    s = re.sub("[\u201c\u201d]", '"', s)
    s = re.sub("[\u2018\u2019]", "'", s)
    s = re.sub("[\u2014]", "-", s)
    return s


@utils.useopt
def normalize_none(s: str) -> str:
    """String normalization that only fixes quotes.

    Args:
        s (str): input string

    Returns:
        str: normalized string
    """
    s = fixquotes(s)
    return s


@utils.useopt
def normalize_simple(s: str) -> str:
    """Simple text normalization.

    - replaces unicode quotes with ascii ones
    - replaces multiple whitespace with singe space
    - strips space at beginning and end of string

    Args:
        s (str): [description]

    Returns:
        str: [description]
    """
    s = fixquotes(s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', "''", s)
    return s.strip()


@utils.useopt
def normalize_tex(s: str) -> str:
    """Simple text normalization.

    - replaces tex commands (\alpha, \beta, etc) with "~"
    - eliminates "{", "}", "_" and "^" from input (TeX sequences)
    - replaces unicode quotes with ascii ones
    - replaces multiple whitespace with singe space
    - strips space at beginning and end of string

    Args:
        s (str): [description]

    Returns:
        str: [description]
    """
    s = fixquotes(s)
    s = re.sub("\\\\[A-Za-z]+", "~", s)
    s = re.sub("\\\\[_^]+", "", s)
    s = re.sub("[{}]", "", s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', "''", s)
    return s.strip()


def good_text(regex: str, sample: dict) -> bool:
    """Check if a string matches a regular expression."""
    return re.search(regex, sample["txt"])


###
# Data Loading
###


class WordPreprocessor:
    def __init__(self, height=64, max_width=512, augment=augment_distort, max_zoom=4.0):
        self.height = height
        self.max_width = max_width
        self.augment = augment
        self.max_zoom = max_zoom

    def goodsize(self, image):
        h, w = image.shape[-2:]
        if h > 200 or w > 2048:
            datawarn(f"image too large {image.shape}")
            return False
        if h < 16 or w < 16:
            datawarn(f"image too small {image.shape}")
            return False
        return True

    def goodhist(self, image):
        hist, _ = torch.histogram(image, bins=5, range=(0, 1))
        if hist[0] < hist[4]:
            datawarn(f"inverted image {hist}")
            return False
        if hist[1:4].sum() > hist[0]:
            datawarn(f"nonbinary image {hist}")
            return False
        return True

    def goodcc(self, image, text):
        if text is None:
            return True
        l = len(text)
        h, w = image.shape[-2:]
        if w / float(l) < 0.3 * h:
            datawarn(f"too narrow for text {image.shape} for {text}")
            return False
        if w / float(l) > 3.0 * h:
            datawarn(f"too wide for text {image.shape} for {text}")
            return False
        _, n = ndi.label(image > 0.5)
        if n < 0.7 * len(text):
            datawarn(f"too few cc's {n} for {text}")
            return False
        if n > 2.0 * len(text):
            datawarn(f"too many cc's {n} for {text}")
            return False
        return True

    def preprocess(
        self,
        sample: Tuple[torch.Tensor, str],
        train: bool = True,
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """Preprocess training sample."""
        image, label = sample
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        if image.ndim == 3:
            image = image.mean(0)
        image = image - image.min()
        image /= max(float(image.max()), 1e-4)
        if not self.goodsize(image):
            return None
        if not self.goodhist(image):
            return None
        image = image.unsqueeze(0)
        image = jittable.crop_image(image)
        if not self.goodsize(image):
            return None
        if not self.goodcc(image, label):
            return None
        zoom = float(self.height) / image.shape[-2]
        if zoom > self.max_zoom:
            datawarn(f"zoom too large {zoom} for {image.shape}")
            return None
        if train:
            zoom *= random.uniform(0.8, 1.0)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            scale_factor=zoom,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )[0]
        if image.shape[-2] < 16 or image.shape[-1] < 16:
            datawarn(f"image too narrow after rescaling {zoom} {image.shape}")
            return None
        if image.shape[-1] > self.max_width:
            datawarn(f"image too wide after rescaling {zoom} {image.shape}")
            return None
        if train:
            image = self.augment(image)
        image = image.clip(0, 1).type(torch.float32)
        return image, label


class TextDataLoader(pl.LightningDataModule):
    """Lightning Data Module for OCR training."""

    extensions = "line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt"
    shuffle = 5000
    nepoch = 50000

    def __init__(
        self,
        probs: List[float] = [1.0, 1.0, 0.2],
        train_bs: int = 16,
        val_bs: int = 24,
        num_workers: int = 8,
        cache_size: int = -1,
        cache_dir: str = None,
        augment: str = "distort",
        bucket: str = "http://storage.googleapis.com/nvdata-ocropus-words/",
        height: int = 64,
        max_width: int = 512,
        val_bucket: str = "http://storage.googleapis.com/nvdata-ocropus-val/",
        val_shards: str = "http://storage.googleapis.com/nvdata-ocropus-val/val-word-{000000..000007}.tar",
        datamode: str = "default",
        **kw,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.augment = eval(f"augment_{augment}")
        self.preprocessor = WordPreprocessor(height=height, max_width=max_width, augment=self.augment)

    def make_reader(self, url, normalize=normalize_simple, select="[A-Za-z0-9]"):
        bucket = self.hparams.bucket
        return (
            wds.WebDataset(
                bucket + url,
                cache_size=float(self.hparams.cache_size),
                cache_dir=self.hparams.cache_dir,
                verbose=True,
                shardshuffle=50,
                resampled=True,
                handler=wds.warn_and_continue,
            )
            .rename(txt="txt;gt.txt")
            .decode()
            .map_dict(txt=normalize)
            .select(partial(good_text, select))
        )

    def make_mix(self):
        sources = []
        if self.hparams.datamode == "uw3":
            sources.append(self.make_reader("uw3-word-{000000..000021}.tar", select="."))
            probs = [1.0]
        else:
            sources.append(self.make_reader("generated-{000000..000313}.tar", select="."))
            sources.append(self.make_reader("uw3-word-{000000..000022}.tar", normalize=normalize_tex))
            sources.append(self.make_reader("ia1-{000000..000033}.tar"))
            sources.append(self.make_reader("gsub-{000000..000167}.tar"))
            sources.append(self.make_reader("cdipsub-{000000..000092}.tar"))
            sources.append(self.make_reader("bin-gsub-{000000..000167}.tar"))
            sources.append(self.make_reader("bin-ia1-{000000..000033}.tar"))
            sources.append(self.make_reader("italic-{000000..000455}.tar", select="."))
            sources.append(self.make_reader("ascii-{000000..000422}.tar", select="."))
            probs = self.hparams.probs
        n = len(sources)
        assert len(probs) <= n
        probs = probs + [probs[-1]] * (n - len(probs))
        ds = wds.FluidWrapper(wds.RandomMix(sources, probs))
        return ds

    def train_dataloader(self) -> DataLoader:
        bs = self.hparams.train_bs
        ds = self.make_mix()
        ds = ds.shuffle(self.shuffle).decode("torchrgb8", partial=True).to_tuple(self.extensions)
        ds = ds.map(self.preprocessor.preprocess)
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        ).slice(self.nepoch // bs)
        return dl

    def val_dataloader(self) -> DataLoader:
        bs = self.hparams.val_bs
        if self.hparams.val_shards in ["", None]:
            return None
        ds = (
            wds.WebDataset(
                self.hparams.val_shards,
                cache_size=float(self.hparams.cache_size),
                cache_dir=self.hparams.cache_dir,
                verbose=True,
            )
            .rename(txt="txt;gt.txt")
            .decode()
            .map_dict(txt=normalize_simple)
            .select(partial(good_text, "[A-Za-z0-9]"))
        )
        ds = ds.decode("torchrgb8", partial=True).to_tuple(self.extensions)
        ds = ds.map(partial(self.preprocessor.preprocess, train=False))
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        ).slice(self.nepoch // bs)
        return dl


@app.command()
def show(rows: int = 8, cols: int = 4, augment: str = "distort", bs: int = 1, nw: int = 0, val: bool = False):
    """Show a sample of the data."""
    n = rows * cols
    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    if val:
        dl = TextDataLoader(augment="none", val_bs=bs, num_workers=nw).val_dataloader()
    else:
        dl = TextDataLoader(augment=augment, train_bs=bs, num_workers=nw).train_dataloader()
    for i, sample in enumerate(dl):
        if i > 0 and i % n == 0:
            plt.ginput(1, timeout=0)
            plt.clf()
        img, txt = sample
        print(i, txt[0], img.shape)
        img = img[0].permute(1, 2, 0).numpy()
        fig.add_subplot(rows, cols, (i % n) + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        txt = re.sub(r"[\\$]", "~", txt[0])
        plt.title(txt)


@app.command()
def bench(t: float = 60.0, augment: str = "distort", bs: int = 1, nw: int = 0, val: bool = False):
    """Show a sample of the data."""
    if val:
        dl = TextDataLoader(augment="none", val_bs=bs, num_workers=nw).val_dataloader()
    else:
        dl = TextDataLoader(augment=augment, train_bs=bs, num_workers=nw).train_dataloader()
    total = 0
    start = time.time()
    for i, sample in enumerate(dl):
        img, txt = sample
        total += len(img)
        now = time.time()
        if now - start > t:
            break
    print("# samples", total, "samples/s", total / (now - start))


if __name__ == "__main__":
    app()
