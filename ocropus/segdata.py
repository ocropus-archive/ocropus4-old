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


def isimage(im):
    assert isinstance(im, torch.Tensor)
    assert im.ndim == 3, im.shape
    assert im.shape[0] == 3, im.shape
    assert im.dtype in [torch.uint8, torch.float32], im.dtype
    return True

def ismask(im):
    assert isinstance(im, torch.Tensor)
    assert im.ndim == 2, im.shape
    return True


def samedims(*args):
    for arg in args[1:]:
        assert arg.shape[-2:] == args[0].shape[-2:], [arg.shape for arg in args]
    return True


def printkv(d):
    for k, v in d.items():
        print(f"{k}: {repr(v)[:60]}")


def decode_image(data):
    return torch.tensor(imread(BytesIO(data))).permute(2, 0, 1)


def decode_mask(data):
    image = decode_image(data)
    return image[0]


def collate4seg(samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, segs, masks = zip(*samples)
    images = jittable.stack_images(images)
    segs = [s.unsqueeze(0) for s in segs]
    segs = jittable.stack_images(segs)
    segs = segs[:, 0]
    masks = [m.unsqueeze(0) for m in masks]
    masks = jittable.stack_images(masks)
    masks = masks[:, 0]
    return images, segs, masks


def simple_bg_fg(binimage, amplitude=0.3, imsigma=1.0, sigma=3.0):
    """Simple noisy grascale image from a binary image."""
    bg = np.random.uniform(size=binimage.shape)
    bg = amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    fg = np.random.uniform(size=binimage.shape)
    fg = 1.0 - amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    mask = utils.normalize_image(ndi.gaussian_filter(binimage, imsigma))
    return mask * fg + (1.0 - mask) * bg


def convert_image_target(sample):
    image, target, mask = sample
    assert isimage(image) and isimage(target)
    target = target[0].long()
    assert target.max() <= 15, target.max()
    return image, target, mask


def make_weight_mask(targets, weightmask=1, bordermask=0):
    assert ismask(targets)
    if weightmask < 0 and bordermask < 0:
        return targets
    mask = targets.detach().cpu().numpy()
    mask = (mask >= 0.5).astype(float)
    if weightmask > 0:
        mask = ndi.maximum_filter(mask, (weightmask, weightmask), mode="constant")
    if bordermask > 0:
        d = bordermask
        mask[:d, :] = 0
        mask[-d:, :] = 0
        mask[:, :d] = 0
        mask[:, -d:] = 0
    mask = torch.tensor(mask, device=targets.device)
    assert ismask(mask)
    return mask


@utils.useopt
def augmentation_none(sample):
    image, target, mask = sample
    assert isimage(image) and ismask(target) and ismask(mask)
    assert samedims(image, target, mask)
    return sample


def masked_norm(image, target):
    a = image.ravel()[target.ravel() > 0]
    lo, hi = np.amin(a), np.amax(a)
    return np.clip((image - lo) / (hi - lo), 0, 1)


@utils.useopt
def augmentation_default(sample, p=0.5, a=2.0):
    image, target, mask = sample
    assert isimage(image) 
    assert ismask(target) 
    assert ismask(mask)
    assert samedims(image, target, mask)
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
        mask = mask.numpy()
        mask = ndi.zoom(mask, scale, order=0)
        mask = torch.tensor(mask)
    if random.random() < p:
        # Randomly rotate the image
        image = image.mean(0).numpy()
        alpha = random.uniform(-a, a)
        assert image.ndim == 2 and image.dtype == np.float32
        image = ndi.rotate(image, alpha, order=1, mode="constant", cval=0)
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
        assert image.ndim == 3 and image.shape[0] == 3, image.shape
        assert target.ndim == 2 and target.dtype == torch.long
        target = target.numpy()
        target = ndi.rotate(target, alpha, order=0, mode="constant", cval=0)
        target = torch.tensor(target)
        mask = mask.numpy()
        mask = ndi.rotate(mask, alpha, order=0, mode="constant", cval=0)
        mask = torch.tensor(mask)
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
    assert isimage(image) and ismask(target) and ismask(mask)
    assert samedims(image, target, mask)
    return image, target, mask


def filter_size(sample, maxsize=1e9):
    image = sample[0]
    if np.prod(image.shape) > maxsize:
        warnings.warn(f"batch too large {image.shape}, maxsize is {maxsize}")
        return None
    return sample


def FilterSize(maxsize):
    return partial(filter_size, maxsize=maxsize)


image_extensions = "image.png;framed.png;ipatch.png;png;jpg;jpeg"
seg_extensions = "target.png;lines.png;spatch.png;seg.png"
mask_extensions = "mask.png;mask.jpg;mask.jpeg"


class SegDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_bs=2,
        val_bs=2,
        scale=0.5,
        augmentation="default",
        shuffle=0,
        num_workers=8,
        invert="False",
        remapper=None,
        nepoch=1000000000,
        maxsize=1e9,
        maxshape=(800, 800),
        synthval=-1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def limit_size(self, sample):
        image, seg, mask = sample
        if image.shape[-2:] > self.hparams.maxshape:
            scale = min(
                self.hparams.maxshape[0] / image.shape[-2], self.hparams.maxshape[1] / image.shape[-1]
            )
            assert isinstance(image, torch.Tensor) and isinstance(seg, torch.Tensor)
            assert image.ndim == 3 and image.shape[0] == 3, image.shape
            assert seg.ndim == 2 and seg.dtype == torch.long, seg.shape
            image = image.mean(0).numpy()
            image = ndi.zoom(image, scale, order=1)
            image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)
            seg = seg.numpy()
            seg = ndi.zoom(seg, scale, order=0)
            seg = torch.tensor(seg)
            mask = mask .numpy()
            mask = ndi.zoom(mask, scale, order=0)
            mask = torch.tensor(mask)
        return image, seg, mask

    def make_sources(self, mode="train"):
        raise NotImplementedError

    def fixup(self, sample):
        return sample

    def process(self, sample):
        image, seg = sample["image"], sample["seg"]
        assert image.dtype == torch.uint8, image.dtype
        assert isimage(image)
        assert samedims(image, seg)
        sample["image"] = image.float() / 255.0
        sample["seg"] = seg = seg[0].long()
        if "synthfigs" in sample["__url__"]:
            if self.hparams.synthval != -1:
                seg[...] = self.hparams.synthval
            mask = torch.ones_like(seg, dtype=torch.float32)
        else:
            mask = make_weight_mask(seg).type(torch.float32)
        sample["mask"] = mask
        self.fixup(sample)
        return sample

    def make_loader(self, batch_size, mode):
        extensions = image_extensions + " " + seg_extensions
        training = self.make_sources(mode=mode)
        training = training.shuffle(
            self.hparams.shuffle,
            handler=wds.warn_and_continue,
        )
        training = training.decode("torchrgb8")
        training = training.rename(image=image_extensions, seg=seg_extensions, handler=wds.warn_and_continue)
        training = training.map(self.process)
        training = training.to_tuple("image", "seg", "mask", handler=wds.warn_and_continue)
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
        return self.make_loader(self.hparams.train_bs, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.hparams.val_bs, mode="val")


class WordSegDataLoader(SegDataLoader):

    train_shards = "http://storage.googleapis.com/nvdata-ocropus-wseg/uw3-wseg-{000000..000117}.tar"
    val_shards = "http://storage.googleapis.com/nvdata-ocropus-val/val-wseg-000000.tar"
    synthfigs = "http://storage.googleapis.com/nvdata-synthfigs/openimages-train-{000000..000143}.tar"

    def fixup(self, sample):
        image = sample["image"]
        seg = sample["seg"]
        mask = sample["mask"]
        assert isimage(image) and ismask(mask) and ismask(seg)
        h, w = image.shape[-2:]
        if h > 512 or w > 512:
            sample["image"] = image[:, :512, :512]
            sample["seg"] = seg[:512, :512]
            sample["mask"] = mask[:512, :512]
        return sample

    def make_sources(self, mode="train"):
        if mode == "train":
            main = wds.WebDataset(self.train_shards, handler=wds.warn_and_continue)
            imgs = wds.WebDataset(self.synthfigs, handler=wds.warn_and_continue)
            sources = [main, imgs]
            probs = [0.7, 0.3]
            return wds.FluidWrapper(wds.RandomMix(sources, probs))
        elif mode == "val":
            return wds.WebDataset(self.val_shards, handler=wds.warn_and_continue)
        else:
            raise ValueError(mode)

    def __init__(self, **kw):
        super().__init__(**kw)


class PageSegDataLoader(SegDataLoader):

    train_shards = (
        "http://storage.googleapis.com/nvdata-publaynet-seg/publaynet-train-{000000..000340}-mseg2.tar"
    )
    val_shards = "http://storage.googleapis.com/nvdata-publaynet-seg/publaynet-val-{000000..000011}-mseg2.tar"

    def make_sources(self, mode="train"):
        if mode == "train":
            return wds.WebDataset(self.train_shards, handler=wds.warn_and_continue)
        elif mode == "val":
            return wds.WebDataset(self.val_shards, handler=wds.warn_and_continue)
        else:
            raise ValueError(mode)

    def __init__(self, **kw):
        super().__init__(**kw)


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
        img, seg, mask = sample
        img = img[0].permute(1, 2, 0).numpy()
        seg = seg[0].numpy()
        mask = mask[0].numpy()
        fig.add_subplot(2, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.title(f"{img.shape} {np.prod(img.shape)}")
        fig.add_subplot(2, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(seg, vmin=0, vmax=5, cmap="gist_rainbow")
        plt.title(f"{seg.shape} {np.prod(seg.shape)}")
        fig.add_subplot(2, 2, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mask, vmin=0.0, vmax=1.0, cmap="gray")
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
