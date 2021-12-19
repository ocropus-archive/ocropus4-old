"""Text recognition."""

import random, re
import os.path
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple

import pytorch_lightning as pl
import torch
import torch.jit
import webdataset as wds
from scipy import ndimage as ndi
from torch.utils.data import DataLoader
from urllib.parse import urljoin
from dataclasses import dataclass, field

from . import confparse, degrade, jittable, utils


def identity(x: Any) -> Any:
    """Identity function."""
    return x


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


def goodsize(sample: Dict[Any, Any], max_w: int = max_w, max_h: int = max_h) -> bool:
    """Determine whether the given sample has a good size."""
    image, _ = sample
    h, w = image.shape[-2:]
    good = h > min_h and h < max_h and w > min_w and w < max_w
    if not good:
        print("bad sized image", image.shape)
        return False
    if image.ndim == 3:
        image = image.mean(0)
    if (image > 0.5).sum() < 10.0:
        print("nearly empty image", image.sum())
        return False
    return True


@utils.useopt
def augment_none(image: torch.Tensor) -> torch.Tensor:
    """Perform no augmentation.

    Args:
        image (torch.Tensor): input image

    Returns:
        torch.Tensor: unaugmented output
    """
    return utils.as_torchimage(image)


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
        (image,) = degrade.distort_all(image)
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


def good_text(regex: str, sample: str) -> bool:
    """Check if a string matches a regular expression."""
    image, txt = sample
    return re.search(regex, txt)


###
### Data Loading
###


all_urls = """
http://storage.googleapis.com/nvdata-ocropus-words/generated-{000000..000313}.tar
http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-{000000..000022}.tar
http://storage.googleapis.com/nvdata-ocropus-words/ia1-{000000..000033}.tar
http://storage.googleapis.com/nvdata-ocropus-words/gsub-{000000..000167}.tar
http://storage.googleapis.com/nvdata-ocropus-words/cdipsub-{000000..000092}.tar
http://storage.googleapis.com/nvdata-ocropus-words/bin-gsub-{000000..000167}.tar
http://storage.googleapis.com/nvdata-ocropus-words/bin-ia1-{000000..000033}.tar
""".strip().split(
    "\n"
)


def make_mixed_loader(probs, hparams):
    assert len(probs) <= len(all_urls)
    probs = probs + [probs[-1]] * (len(all_urls) - len(probs))
    sources = []
    for i, url in enumerate(all_urls):
        print(f"adding {url} with weight {probs[i]}")
        ds = wds.WebDataset(
            url,
            cache_size=float(hparams.cache_size),
            cache_dir=hparams.cache_dir,
            verbose=True,
            shardshuffle=50,
            resampled=True,
        )
        sources.append(ds)
    ds = wds.FluidWrapper(wds.RandomMix(sources, probs))
    return ds


class TextDataLoader(pl.LightningDataModule):
    """Lightning Data Module for OCR training."""

    val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-word-{000000..000007}.tar"
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
        **kw,
    ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        bs = self.hparams.train_bs
        ds = make_mixed_loader(self.hparams.probs, self.hparams)
        ds = ds.shuffle(self.shuffle)
        ds = ds.decode("torchrgb8").to_tuple(self.extensions)
        ds = ds.map_tuple(identity, normalize_tex)
        ds = ds.select(partial(good_text, "[A-Za-z0-9]"))
        ds = ds.map_tuple(eval(f"augment_{self.hparams.augment}"), identity)
        ds = ds.map_tuple(jittable.standardize_image, identity)
        ds = ds.select(partial(goodsize, max_w=1000, max_h=100))
        ds = ds.map_tuple(jittable.auto_resize, identity)
        ds = ds.select(partial(goodsize, max_w=1000, max_h=100))
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        ).slice(self.nepoch // bs)
        return dl

    def val_dataloader(self) -> DataLoader:
        if self.hparams.val_shards in ["", None]:
            return None
        ds = wds.WebDataset(
            self.val_shards,
            cache_size=float(self.hparams.cache_size),
            cache_dir=self.hparams.cache_dir,
            verbose=True,
        )
        ds = ds.decode("torchrgb8").to_tuple(self.extensions)
        ds = ds.map_tuple(identity, normalize_simple)
        ds = ds.select(partial(good_text, "[A-Za-z0-9]"))
        ds = ds.map_tuple(jittable.standardize_image, identity)
        ds = ds.select(partial(goodsize, max_w=1000, max_h=100))
        ds = ds.map_tuple(jittable.auto_resize, identity)
        ds = ds.select(partial(goodsize, max_w=1000, max_h=100))
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=self.hparams.val_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return dl
