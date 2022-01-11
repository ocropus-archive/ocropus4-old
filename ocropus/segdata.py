import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from scipy import ndimage as ndi
from webdataset.filters import default_collation_fn
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import partial
import warnings
import random
import typer


from . import confparse, utils, jittable, degrade


app = typer.Typer()


def collate4seg(samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, segs = zip(*samples)
    images = jittable.stack_images(images)
    segs = [s.unsqueeze(0) for s in segs]
    segs = jittable.stack_images(segs)
    segs = segs[:, 0]
    return images, segs


def simple_bg_fg(binimage, amplitude=0.3, imsigma=1.0, sigma=3.0):
    """Simple noisy grascale image from a binary image."""
    bg = np.random.uniform(size=binimage.shape)
    bg = amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    fg = np.random.uniform(size=binimage.shape)
    fg = 1.0 - amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    mask = utils.normalize_image(ndi.gaussian_filter(binimage, imsigma))
    return mask * fg + (1.0 - mask) * bg


def convert_image_target(sample):
    image, target = sample
    assert image.shape[0] == 3, image.shape
    assert target.shape[0] == 3, target.shape
    image = image.type(torch.float32) / 255.0
    target = target[0].long()
    assert target.max() <= 15, target.max()
    return image, target


@utils.useopt
def augmentation_none(sample):
    image, target = sample
    assert isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    assert image.dtype == torch.float32 and image.max() <= 1.0
    assert target.ndim == 2 and target.dtype == torch.long
    return sample


def masked_norm(image, target):
    a = image.ravel()[target.ravel() > 0]
    lo, hi = np.amin(a), np.amax(a)
    return np.clip((image - lo) / (hi - lo), 0, 1)


@utils.useopt
def augmentation_default(sample, p=0.5, a=2.0):
    image, target = sample
    assert isinstance(image, torch.Tensor), type(image)
    assert isinstance(target, torch.Tensor), type(target)
    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    assert image.dtype == torch.float32 and image.max() <= 1.0
    assert target.ndim == 2 and target.dtype == torch.long
    if random.random() < p:
        assert image.shape[-2:] == target.shape[-2:]
        h, w = image.shape[-2:]
        image = image.mean(0).numpy()
        scale = random.uniform(0.9, 1.1)
        assert image.ndim == 2 and image.dtype == np.float32
        image = ndi.zoom(image, scale, order=1)
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
        assert image.ndim == 3 and image.shape[0] == 3, image.shape
        assert target.ndim == 2 and target.dtype == torch.long
        target = target.numpy()
        target = ndi.zoom(target, scale, order=0)
        target = torch.tensor(target)
    if random.random() < p:
        # Randomly rotate the image
        image = image.mean(0).numpy()
        alpha = random.uniform(-a, a)
        assert image.ndim == 2 and image.dtype == np.float32
        image = ndi.rotate(image, alpha, order=1, mode="mirror")
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
        assert image.ndim == 3 and image.shape[0] == 3, image.shape
        assert target.ndim == 2 and target.dtype == torch.long
        target = target.numpy()
        target = ndi.rotate(target, alpha, order=0, mode="mirror")
        target = torch.tensor(target)
    if random.random() < p:
        # Blur the image
        image = image.mean(0).numpy()
        sigma = random.uniform(0.5, 1.5)
        image = ndi.gaussian_filter(image, sigma=sigma)
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
    if random.random() < p:
        # Add noise
        image = image.mean(0).numpy()
        image = degrade.noisify(image, amp1=0.2, amp2=0.2)
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
    return image, target


def filter_size(sample, maxsize=1e9):
    image, seg = sample
    if np.prod(image.shape) > maxsize:
        warnings.warn(f"batch too large {image.shape}, maxsize is {maxsize}")
        return None
    return sample


def FilterSize(maxsize):
    return partial(filter_size, maxsize=maxsize)


class SegDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_shards=None,
        train_bs=2,
        val_shards=None,
        val_bs=2,
        extensions="image.png;framed.png;ipatch.png;png target.png;lines.png;spatch.png;seg.png",
        scale=0.5,
        augmentation="default",
        shuffle=0,
        num_workers=8,
        invert="False",
        remapper=None,
        nepoch=1000000000,
        maxsize=1e9,
        maxshape=(800, 800),
    ):
        super().__init__()
        assert train_shards is not None
        self.save_hyperparameters()

    def limit_size(self, sample):
        image, seg = sample
        if image.shape[-2:] > self.hparams.maxshape:
            scale = min(self.hparams.maxshape[0] / image.shape[-2], self.hparams.maxshape[1] / image.shape[-1])
            assert isinstance(image, torch.Tensor) and isinstance(seg, torch.Tensor)
            assert image.ndim == 3 and image.shape[0] == 3, image.shape
            assert seg.ndim == 2 and seg.dtype == torch.long, seg.shape
            image = image.mean(0).numpy()
            image = ndi.zoom(image, scale, order=1)
            image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
            seg = seg.numpy()
            seg = ndi.zoom(seg, scale, order=0)
            seg = torch.tensor(seg)
        return image, seg

    def make_loader(self, urls, batch_size, mode):
        training = wds.WebDataset(urls, handler=wds.warn_and_continue)
        training = training.shuffle(
            self.hparams.shuffle,
            handler=wds.warn_and_continue,
        )
        training = training.decode("torchrgb8")
        training = training.to_tuple(self.hparams.extensions, handler=wds.warn_and_continue)
        training = training.map(convert_image_target)
        if self.hparams.remapper is not None:
            training = training.map_tuple(None, self.hparams.remapper)
        training = training.map(self.limit_size)
        if mode == "train":
            augmentation = eval(f"augmentation_{self.hparams.augmentation}")
            training = training.map(augmentation)
        return (
            wds.WebLoader(
                training,
                batch_size=batch_size,
                collate_fn=collate4seg,
                num_workers=self.hparams.num_workers,
            )
            .map(FilterSize(self.hparams.maxsize))
            .slice(self.hparams.nepoch // batch_size)
        )

    def train_dataloader(self):
        return self.make_loader(
            self.hparams.train_shards,
            self.hparams.train_bs,
            "train",
        )

    def val_dataloader(self):
        if self.val_shards is None:
            return None
        return self.make_loader(
            self.hparams.val_shards,
            self.hparams.val_bs,
            "val",
        )


class WordSegDataLoader(SegDataLoader):

    train_shards = "http://storage.googleapis.com/nvdata-ocropus-wseg/uw3-wseg-{000000..000117}.tar"
    val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-wseg-000000.tar"

    def __init__(self, train_shards=None, val_shards=None, **kw):
        train_shards = train_shards or self.train_shards
        val_shards = val_shards or self.val_shards
        super().__init__(train_shards=self.train_shards, val_shards=self.val_shards, **kw)


class PageSegDataLoader(SegDataLoader):

    train_shards = (
        "http://storage.googleapis.com/nvdata-publaynet-seg/publaynet-train-{000000..000340}-mseg2.tar"
    )
    val_shards = "http://storage.googleapis.com/nvdata-publaynet-seg/publaynet-val-{000000..000011}-mseg2.tar"

    def __init__(self, train_shards=None, val_shards=None, **kw):
        train_shards = train_shards or self.train_shards
        val_shards = val_shards or self.val_shards
        super().__init__(train_shards=train_shards, val_shards=val_shards, **kw)


@app.command()
def words(bs: int = 1, nw: int = 0, val: bool = False):
    """Show a sample of the data."""
    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    if val:
        dl = WordSegDataLoader(val_bs=bs, num_workers=nw).val_dataloader()
    else:
        dl = WordSegDataLoader(train_bs=bs, num_workers=nw).train_dataloader()
    plt.ion()
    for i, sample in enumerate(dl):
        plt.clf()
        img, seg = sample
        img = img[0].permute(1, 2, 0).numpy()
        seg = seg[0].numpy()
        fig.add_subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.title(f"{img.shape} {np.prod(img.shape)}")
        fig.add_subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(seg, vmin=0, vmax=5, cmap="gist_rainbow")
        plt.title(f"{seg.shape} {np.prod(seg.shape)}")
        plt.ginput(1, timeout=0)


@app.command()
def pages(bs: int = 1, nw: int = 0, val: bool = False):
    """Show a sample of the data."""
    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    if val:
        dl = PageSegDataLoader(val_bs=bs, num_workers=nw).val_dataloader()
    else:
        dl = PageSegDataLoader(train_bs=bs, num_workers=nw).train_dataloader()
    plt.ion()
    for i, sample in enumerate(dl):
        plt.clf()
        img, seg = sample
        img = img[0].permute(1, 2, 0).numpy()
        seg = seg[0].numpy()
        fig.add_subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(1 - img)
        plt.title(f"{img.shape} {np.prod(img.shape)}")
        fig.add_subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(seg, vmin=0, vmax=5, cmap="gist_rainbow")
        plt.title(f"{seg.shape} {np.prod(seg.shape)}")
        plt.ginput(1, timeout=0)


if __name__ == "__main__":
    app()
