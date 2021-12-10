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

from . import confparse, degrade, jittable, utils


def identity(x: Any) -> Any:
    """Identity function."""
    return x

min_w, min_h, max_w, max_h = 15, 15, 4000, 200

def collate4ocr(
    samples: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[str]]:
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
def augment_transform(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using geometric transformations and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): input image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: augmented image
    """
    image = utils.as_npimage(image)
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    image = utils.as_torchimage(image)
    return image


@utils.useopt
def augment_distort(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using distortions and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): original image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        [type]: augmented image
    """
    image = utils.as_npimage(image)
    image = image.mean(axis=2)
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        (image,) = degrade.distort_all(image)
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    image = utils.as_torchimage(image)
    return image


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


class TextDataLoader(pl.LightningDataModule):
    """Lightning Data Module for OCR training."""

    default_bucket = "https://storage.googleapis.com/nvdata-ocropus-words"
    default_shards = "uw3-word-{000000..000022}.tar"
    default_val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-word-{000000..000007}.tar"

    def __init__(
        self,
        train_bucket: Optional[str] = None,
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
            train_bucket (Optional[str]): URL of the bucket containing the training data
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
        train_shards = utils.get_shards(train_bucket, train_shards)
        val_shards = val_shards or self.default_val_shards
        self.train_shards = train_shards
        self.train_bs = train_bs
        self.val_shards = val_shards
        self.val_bs = val_bs
        self.cache_size = cache_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.extensions = extensions
        self.text_normalizer = text_normalizer
        self.text_select_re = text_select_re
        self.num_workers = num_workers
        self.nepoch = nepoch
        self.max_w = max_w
        self.max_h = max_h
        print(f"+++ {self.train_shards}, {self.val_shards}")

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
        ds = wds.WebDataset(
            fname,
            cache_size=float(self.cache_size),
            cache_dir=self.cache_dir,
            verbose=True,
            shardshuffle=50,
            resampled=True,
        )
        if mode == "train" and self.shuffle > 0:
            ds = ds.shuffle(self.shuffle)
        ds = ds.decode("torchrgb8").to_tuple(self.extensions)
        text_normalizer = eval(f"normalize_{self.text_normalizer}")
        ds = ds.map_tuple(identity, text_normalizer)
        if self.text_select_re != "":
            ds = ds.select(partial(good_text, self.text_select_re))
        if augment != "":
            f = eval(f"augment_{augment}")
            ds = ds.map_tuple(f, identity)
        ds = ds.map_tuple(jittable.standardize_image, identity)
        ds = ds.select(goodsize)
        ds = ds.map_tuple(jittable.auto_resize, identity)
        ds = ds.select(partial(goodsize, max_w=self.max_w, max_h=self.max_h))
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ).slice(self.nepoch // batch_size)
        return dl

    def train_dataloader(self) -> DataLoader:
        """Make a data loader for training.

        Returns:
            DataLoader: data loader
        """
        return self.make_loader(
            self.train_shards,
            self.train_bs,
            mode="train",
        )

    def val_dataloader(self) -> DataLoader:
        """Make a data loader for validation.

        Returns:
            DataLoader: data loader
        """
        if self.val_shards in ["", None]:
            return None
        return self.make_loader(
            self.val_shards,
            self.val_bs,
            mode="val",
            augment="",
        )
