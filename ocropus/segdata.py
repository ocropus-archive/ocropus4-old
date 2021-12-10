import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from scipy import ndimage as ndi
from webdataset.filters import default_collation_fn

from . import confparse, utils


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
    assert target.ndim == 2 and target.dtype == torch.uint8
    return sample


def masked_norm(image, target):
    a = image.ravel()[target.ravel() > 0]
    lo, hi = np.amin(a), np.amax(a)
    return np.clip((image - lo) / (hi - lo), 0, 1)


@utils.useopt
def augmentation_default(sample):
    image, target = sample
    assert isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    assert image.dtype == torch.float32 and image.max() <= 1.0
    assert target.ndim == 2 and target.dtype == torch.long
    return sample


class SegDataLoader(pl.LightningDataModule):
    default_train_shards = "http://storage.googleapis.com/nvdata-ocropus-wseg/uw3-wseg-{000000..000117}.tar"
    default_val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-wseg-000000.tar"

    def __init__(
        self,
        train_shards=None,
        train_bs=2,
        val_shards=None,
        val_bs=2,
        extensions="image.png;framed.png;ipatch.png;png target.png;lines.png;spatch.png;seg.png",
        scale=0.5,
        augmentation="none",
        shuffle=0,
        num_workers=8,
        invert="False",
        remapper=None,
        nepoch=1000000000,
    ):
        super().__init__()
        train_shards = train_shards or self.default_train_shards
        train_shards = utils.maybe_expand_bucket(train_shards)
        val_shards = val_shards or self.default_val_shards
        self.save_hyperparameters()

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
        if mode == "train":
            augmentation = eval(f"augmentation_{self.hparams.augmentation}")
            training = training.map(augmentation)
        return wds.WebLoader(training, batch_size=batch_size, num_workers=self.hparams.num_workers).slice(
            self.hparams.nepoch // batch_size
        )

    def train_dataloader(self):
        return self.make_loader(
            self.hparams.train_shards,
            self.hparams.train_bs,
            "train",
        )

    def val_dataloader(self):
        return self.make_loader(
            self.hparams.val_shards,
            self.hparams.val_bs,
            "val",
        )
