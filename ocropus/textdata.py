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
""".strip().split("\n")

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

    default_train_shards = "http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-{000000..000022}.tar"
    default_bucket = "http://storage.googleapis.com/nvdata-ocropus-words/"
    default_val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-word-{000000..000007}.tar"

    def __init__(
        self,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        train_bs: int = 16,
        val_bs: int = 24,
        text_select_re: str = "[A-Za-z0-9]",
        nepoch: int = 50000,
        num_workers: int = 8,
        cache_size: int = -1,
        cache_dir: str = None,
        shuffle: int = 5000,
        augment: str = "distort",
        text_normalizer: str = "simple",
        extensions: str = "line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
        max_w: int = 1000,
        max_h: int = 200,
        **kw,
    ):
        """Initialize the TextDataLoader

        Args:
            train_shards (Optional[Union[str, List[str], Dict[Any, Any]]], optional): list of shards to train on. Defaults to None.
            val_shards (Optional[Union[str, List[str]]], optional): list of shards to validate on. Defaults to None.
            train_bs (int, optional): batch size for training. Defaults to 4.
            val_bs (int, optional): batch size for validation. Defaults to 20.
            text_select_re (str, optional): regular expression that selects training samples. Defaults to "[A-Za-z0-9]".
            nepoch (int, optional): number of samples per epoch. Defaults to 5000.
            num_workers (int, optional): number of workers per loader. Defaults to 4.
            cache_size (int, optional): cache size for downloading shards. Defaults to -1.
            cache_dir (str, optional): directory where shards are cached. Defaults to None.
            shuffle (int, optional): size of inline shuffle buffer. Defaults to 5000.
            augment (str, optional): choice of sample augmentation. Defaults to "distort".
            text_normalizer (str, optional): choice of text normalization. Defaults to "simple".
            extensions (str, optional): choice of file name extensions. Defaults to "line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt".
            max_w (int, optional): maximum image width (larger=ignored). Defaults to 1000.
            max_h (int, optional): maximum image height (larger=ignored). Defaults to 200.
        """
        super().__init__()
        train_shards = train_shards or "@small"
        val_shards = val_shards or self.default_val_shards
        self.save_hyperparameters()

    def make_loader(
        self,
        fname: Union[str, List[str], Dict[Any, Any]],
        batch_size: int,
        mode: str = "train",
        augment: str = "distort",
    ) -> DataLoader:
        """Make a data loader for a given collection of shards.

        Args:
            fname (Union[str, List[str], Dict[Any, Any]]): shard spec, shard list, or dataset dict spec
            batch_size (int): desired batch size
            mode (str, optional): mode (val or train). Defaults to "train".
            augment (str, optional): augmentation function to use. Defaults to "distort".

        Returns:
            DataLoader: data loader
        """
        if fname[0] == "[":
            ds = make_mixed_loader(eval(fname), self.hparams)
        else:
            if fname == "@small":
                train_shards = self.default_train_shards
            elif fname == "@bucket":
                train_shards = self.default_bucket
                train_shards = utils.maybe_expand_bucket(train_shards)
            else:
                train_shards = utils.maybe_expand_bucket(fname)
            ds = wds.WebDataset(
                train_shards,
                cache_size=float(self.hparams.cache_size),
                cache_dir=self.hparams.cache_dir,
                verbose=True,
                shardshuffle=50 if mode == "train" else 0,
                resampled=(mode == "train"),
            )
        if mode == "train" and self.hparams.shuffle > 0:
            ds = ds.shuffle(self.hparams.shuffle)
        ds = ds.decode("torchrgb8").to_tuple(self.hparams.extensions)
        text_normalizer = eval(f"normalize_{self.hparams.text_normalizer}")
        ds = ds.map_tuple(identity, text_normalizer)
        if self.hparams.text_select_re != "":
            ds = ds.select(partial(good_text, self.hparams.text_select_re))
        if augment != "":
            f = eval(f"augment_{augment}")
            ds = ds.map_tuple(f, identity)
        ds = ds.map_tuple(jittable.standardize_image, identity)
        ds = ds.select(goodsize)
        ds = ds.map_tuple(jittable.auto_resize, identity)
        ds = ds.select(partial(goodsize, max_w=self.hparams.max_w, max_h=self.hparams.max_h))
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        ).slice(self.hparams.nepoch // batch_size)
        return dl

    def train_dataloader(self) -> DataLoader:
        """Make a data loader for training.

        Returns:
            DataLoader: data loader
        """
        print("using training shards:", self.hparams.train_shards)
        ds = self.hparams.train_shards
        return self.make_loader(
            ds,
            self.hparams.train_bs,
            mode="train",
        )

    def val_dataloader(self) -> DataLoader:
        """Make a data loader for validation.

        Returns:
            DataLoader: data loader
        """
        if self.hparams.val_shards in ["", None]:
            return None
        return self.make_loader(
            self.hparams.val_shards,
            self.hparams.val_bs,
            mode="val",
            augment="",
        )
