import os
import sys
import time
import math

import random
import typer
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import amin, median, mean
from scipy import ndimage as ndi
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds
from torchmore import layers
import traceback
import skimage
import skimage.filters
from functools import partial
from itertools import islice

from .utils import Schedule, repeatedly
from . import slog
from . import utils
from . import loading
from . import patches
from . import slices as sl
from .utils import useopt, junk
from . import degrade


logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


app = typer.Typer()


###
# Loading and Preprocessing
###


def simple_bg_fg(binimage, amplitude=0.3, imsigma=1.0, sigma=3.0):
    """Simple noisy grascale image from a binary image."""
    bg = np.random.uniform(size=binimage.shape)
    bg = amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    fg = np.random.uniform(size=binimage.shape)
    fg = 1.0 - amplitude * utils.normalize_image(ndi.gaussian_filter(bg, sigma))
    mask = utils.normalize_image(ndi.gaussian_filter(binimage, imsigma))
    return mask * fg + (1.0 - mask) * bg


@useopt
def augmentation_none(sample):
    image, target = sample
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(target, np.ndarray), type(image)
    print(image.dtype, image.shape, target.dtype, target.shape)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if target.ndim == 3:
        target = target[..., 0]
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64), target.dtype
    assert image.shape[:2] == target.shape[:2]
    return image, target


def masked_norm(image, target):
    a = image.ravel()[target.ravel() > 0]
    lo, hi = np.amin(a), np.amax(a)
    return np.clip((image - lo) / (hi - lo), 0, 1)


@useopt
def augmentation_default(sample):
    image, target = sample
    assert isinstance(image, np.ndarray), type(image)
    assert isinstance(target, np.ndarray), type(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 3:
        image = np.mean(image, 2)
    if target.ndim == 3:
        target = target[..., 0]
    #print(image.dtype, image.shape, target.dtype, target.shape)
    x = random.uniform(0, 1)
    if x < 0.3:
        image = masked_norm(image, target)
    else:
        image = image - np.amin(image)
        image /= max(np.amax(image), 0.001)
    if random.uniform(0.0, 1.0) < 0.3:
        image, target = degrade.transform_all(image, target, order=[1, 0])
    if False and random.uniform(0.0, 1.0) < 0.5:
        # FIXME this generates bad masks somehow
        image, target = degrade.distort_all(image, target, order=[1, 0], sigma=(3.0, 10.0), maxdelta=(0.1, 5.0))
    if random.uniform(0.0, 1.0) < 0.5:
        image = degrade.noisify(image)
    image = np.clip(image, 0.0, 1.0)
    #print(image.dtype, image.shape, target.dtype, target.shape)
    image = np.array([image, image, image], dtype=np.float32).transpose(1, 2, 0)
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64), target.dtype
    assert image.shape[:2] == target.shape[:2]
    return image, target


def np2tensor(sample):
    image, target = sample
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64)
    assert image.shape[:2] == target.shape[:2]
    image = np.mean(image, 2)
    image = torch.tensor(image).float().unsqueeze(0)
    target = torch.tensor(target).long()
    return image, target

def checktypes(message, sample):
    image, target = sample
    assert isinstance(image, np.ndarray), (message, type(image))
    assert isinstance(target, np.ndarray), (message, type(image))
    return sample

def make_loader(
    urls,
    batch_size=2,
    extensions="image.png;framed.png;ipatch.png target.png;lines.png;spatch.png",
    scale=0.5,
    augmentation=augmentation_none,
    shuffle=0,
    num_workers=1,
    invert="False",
    remapper=None,
):
    def autoinvert(x):
        return utils.autoinvert(x, invert)

    def remap(y):
        if remapper is None:
            return y
        return remapper[y]

    training = wds.WebDataset(urls).shuffle(shuffle).decode("rgb8")
    training = training.to_tuple(extensions, handler=wds.warn_and_continue)
    training = training.map_tuple(autoinvert, remap)
    training = training.map(augmentation)
    training = training.map(np2tensor)
    if "LOGKEYS" in os.environ:
        training = training.log_keys(os.environ["LOGKEYS"])
    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


###
# Logging
###

mem_every = utils.Every(60)


@junk
def memsum(trainer):
    """Output current memory usage every minute. (Callback)"""
    if not mem_every():
        return
    print(torch.cuda.memory_summary())


log_progress_every = utils.Every(60)


def log_progress(trainer):
    """Log training progress every minute. (Callback)"""
    if not log_progress_every():
        return
    avgloss = mean(trainer.losses[-100:]) if len(trainer.losses) > 0 else 0.0
    logger.scalar("train/loss", avgloss, step=trainer.nsamples, json=dict(lr=trainer.last_lr))
    logger.flush()


print_progress_every = utils.Every(60)


def print_progress(self):
    if not print_progress_every():
        return
    print(
        f"# {len(self.losses)} {np.median(self.losses[-10:])}", file=sys.stderr, flush=True,
    )


def display_progress(self):
    import matplotlib.pyplot as plt

    cmap = plt.cm.nipy_spectral

    if int(os.environ.get("noreport", 0)):
        return
    if time.time() - self.last_display < self.every:
        return
    self.last_display = time.time()
    inputs, targets, outputs = self.last_batch
    plt.ion()
    fig = plt.gcf()
    fig.clf()
    for i in range(6):
        fig.add_subplot(2, 3, i + 1)
    ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
    ax1.set_title(f"{len(self.losses)}")
    doc = inputs[0, 0].detach().cpu().numpy()
    mask = getattr(self, "last_mask")
    if mask is not None:
        mask = mask[0].detach().numpy()
        combined = np.array([doc, doc, mask]).transpose(1, 2, 0)
        ax1.imshow(combined)
    else:
        ax1.imshow(doc, cmap="gray")
    p = outputs.detach().cpu().softmax(1)
    assert not torch.isnan(inputs).any()
    assert not torch.isnan(outputs).any()
    b, d, h, w = outputs.size()
    result = p.numpy()[0].transpose(1, 2, 0)
    if result.shape[2] > 3:
        result = result[..., 1:4]
    else:
        result = result[..., :3]
    ax2.imshow(result, vmin=0, vmax=1)
    m = result.shape[1] // 2
    ax2.plot([m, m], [0, h], color="white", alpha=0.5)
    if len(self.losses) >= 100:
        atsamples = self.atsamples[::10]
        losses = self.losses
        losses = ndi.gaussian_filter(losses, 10.0)
        losses = losses[::10]
        losses = ndi.gaussian_filter(losses, 10.0)
        ax4.plot(atsamples, losses)
        ax4.set_ylim((0.9 * amin(losses), median(losses) * 3))
    colors = [cmap(x) for x in np.linspace(0, 1, p.shape[1])]
    for i in range(0, d):
        ax5.plot(p[0, i, :, m], color=colors[i % len(colors)])
    if p.shape[1] <= 4:
        t = targets[0].detach().numpy()
        t = np.array([t == 1, t == 2, t == 3]).astype(float).transpose(1, 2, 0)
        ax3.imshow(t)
    else:
        ax3.imshow(p.argmax(1)[0], vmin=0, vmax=p.shape[1], cmap=cmap)
    ax6.imshow(targets[0].detach().numpy(), vmin=0, vmax=p.shape[1], cmap=cmap)
    plt.ginput(1, 0.001)


###
# Training
###


class SegTrainer:
    def __init__(
        self,
        model,
        *,
        lr=1e-4,
        every=3.0,
        device=None,
        maxgrad=10.0,
        margin=16,
        weightmask=0,
        bordermask=16,
        **kw,
    ):
        super().__init__()
        self.device = utils.device(device)
        if self.device.type == "cpu":
            print("SegTrain using CPU")
        self.model = model.to(self.device)
        self.every = every
        self.atsamples = []
        self.losses = []
        self.last_lr = None
        self.lr_schedule = None
        self.clip_gradient = maxgrad
        self.charset = None
        self.margin = margin
        self.last_display = 0
        self.nsamples = 0
        self.nbatches = 0
        self.weightmask = weightmask
        self.bordermask = bordermask
        self.weightlayer = layers.weighted_grad
        self.last_mask = None
        self.set_lr(lr)
        self.old_interpolate = False

    def set_lr(self, lr, momentum=0.9):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if lr != self.last_lr:
            print(f"# learning rate {lr} @ {self.nsamples}", file=sys.stderr)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            self.last_lr = lr

    def set_lr_schedule(self, f):
        assert callable(f)
        self.lr_schedule = f

    def train_batch(self, inputs, targets):
        """All the steps necessary for training a batch.

        Stores the last batch in self.last_batch.
        Adds the loss to self.losses.
        Clips the gradient if self.clip_gradient is not None.
        """
        if self.lr_schedule:
            self.set_lr(self.lr_schedule(self.nsamples))
        self.model.train()
        self.optimizer.zero_grad()
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = self.model.forward(inputs)
        if outputs.shape != inputs.shape:
            assert outputs.shape[0] == inputs.shape[0]
            assert outputs.ndim == 4
            bs, h, w = targets.shape
            if self.old_interpolate:
                # this is wrong when used with ModPad(16)
                outputs = F.interpolate(outputs, size=(h, w))
            else:
                outputs = outputs[:, :, :h, :w]
        if self.weightmask >= 0:
            mask = targets.detach().numpy()
            assert mask.ndim == 3
            mask = (mask >= 0.5).astype(float)
            if self.weightmask > 0:
                w = self.weightmask
                mask = ndi.maximum_filter(mask, (0, w, w), mode="constant")
            if self.bordermask > 0:
                d = self.bordermask
                mask[:, :d, :] = 0
                mask[:, -d:, :] = 0
                mask[:, :, :d] = 0
                mask[:, :, -d:] = 0
            mask = torch.tensor(mask)
            self.last_mask = mask
            # umask = mask.unsqueeze(1).expand(-1, outputs.size(1), -1, -1)
            # outputs = self.weightlayer(outputs, umask.to(outputs.device))
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets, mask=mask)
        if math.isnan(float(loss)):
            print("got NaN loss", file=sys.stderr)
            return 999.0
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (
            inputs.detach().cpu(),
            targets.detach().cpu(),
            outputs.detach().cpu(),
        )
        self.nsamples += len(inputs)
        self.nbatches += 1
        loss = loss.detach().item()
        self.atsamples.append(self.nsamples)
        self.losses.append(loss)
        return loss

    def compute_loss(self, outputs, targets, mask=None):
        """Compute loss taking a margin into account."""
        b, d, h, w = outputs.shape
        b1, h1, w1 = targets.shape
        assert h <= h1 and w <= w1 and h1 - h < 5 and w1 - w < 5, (
            outputs.shape,
            targets.shape,
        )
        targets = targets[:, :h, :w]
        # lsm = outputs.log_softmax(1)
        if self.margin > 0:
            m = self.margin
            outputs = outputs[:, :, m:-m, m:-m]
            targets = targets[:, m:-m, m:-m]
            if mask is not None:
                mask = mask[:, m:-m, m:-m]
        if mask is None:
            loss = nn.CrossEntropyLoss()(outputs, targets.to(outputs.device))
        else:
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets.to(outputs.device))
            loss = torch.sum(loss * mask.to(loss.device)) / (0.1 + mask.sum())
        return loss

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def predict_batch(self, inputs):
        """Compute probability outputs for the batch."""
        probs = self.probs_batch(inputs)
        return probs.argmax(1)

    def err_batch(self, inputs, targets):
        pred = self.predict_batch(inputs)
        assert pred.size() == targets.size(), (pred.size(), targets.size())
        errors = (pred != targets).sum()
        return float(errors) / float(targets.nelement())

    def errors(self, loader, ntest=999999999):
        total = 0
        errors = []
        for inputs, targets in loader:
            if total >= ntest:
                break
            err = self.err_batch(inputs, targets)
            errors.append(err)
            total += len(inputs)
        if len(errors) < 1:
            return 1.0
        return np.mean(errors)


###
# Inference
###


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = ndi.distance_transform_edt(labels == 0, return_distances=1, return_indices=1)
    indexes = features[0] * labels.shape[1] + features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= distances < maxdist
    return spread


def marker_segmentation(markers, regions, maxdist=100):
    labels, _ = ndi.label(markers)
    spread = spread_labels(labels, maxdist=maxdist)
    segmented = np.where(np.maximum(markers, regions), spread, 0)
    return segmented


def smooth_probabilities(probs, smooth):
    if smooth == 0.0:
        return probs
    if isinstance(smooth, (int, float)):
        smooth = (float(smooth), float(smooth), 0)
    else:
        assert len(smooth) == 2
        smooth = list(smooth) + [0]
    maxima = np.amax(np.amax(probs, 0), 0)
    assert maxima.shape == (4,)
    gprobs = ndi.gaussian_filter(probs, smooth)
    gmaxima = np.amax(np.amax(gprobs, 0), 0)
    gprobs /= (maxima / gmaxima)[np.newaxis, np.newaxis, :]
    gprobs = gprobs / gprobs.sum(2)[:, :, np.newaxis]
    return gprobs


class Segmenter:
    def __init__(self, model, scale=0.5, device=None):
        self.smooth = 0.0
        self.model = model
        self.marker_threshold = 0.3
        self.region_threshold = 0.3
        self.maxdist = 100
        self.patchsize = (512, 512)
        self.overlap = (64, 64)
        self.device = device or utils.device(device)

    def to(self, device):
        self.model.to(device)

    def segment(self, page):
        assert isinstance(page, np.ndarray)
        assert page.ndim == 2
        assert page.shape[0] >= 100 and page.shape[0] < 20000, page.shape
        assert page.shape[1] >= 100 and page.shape[1] < 20000, page.shape
        self.page = page
        if page.ndim == 2:
            page = np.expand_dims(page, 2)
        self.model.to(self.device)
        try:
            self.model.eval()
            probs = patches.patchwise_inference(page, self.model, patchsize=self.patchsize, overlap=self.overlap)
        finally:
            self.model.to("cpu")
        self.probs = probs
        self.gprobs = smooth_probabilities(probs, self.smooth)
        self.segments = marker_segmentation(
            self.gprobs[..., 3] > self.marker_threshold,
            self.gprobs[..., 2] > self.region_threshold,
            self.maxdist,
        )
        return [
            (obj[0].start, obj[0].stop, obj[1].start, obj[1].stop) for obj in ndi.find_objects(self.segments)
        ]


# class LineSegmenter(Segmenter):
#     def __init__(self, smooth=(1, 4)):
#         super().__init__(smooth=smooth)

#     def get_targets(self, classes):
#         return classes == 2


def extract_boxes(page, boxes, pad=5):
    for y0, y1, x0, x1 in boxes:
        h, w = y1 - y0, x1 - x0
        word = ndi.affine_transform(
            page, np.eye(2), output_shape=(h + 2 * pad, w + 2 * pad), offset=(y0 - pad, x0 - pad), order=0,
        )
        yield word


def save_model(logger, trainer, test_dl, ntest=999999):
    model = trainer.model
    if test_dl is not None:
        print("# testing", file=sys.stderr)
        err = trainer.errors(test_dl, ntest=ntest)
        logger.scalar("val/err", err, step=trainer.nsamples)
    elif len(trainer.losses) < 10:
        err = 999.0
    else:
        err = np.mean(trainer.losses[-100:])
    print(f"# saving {trainer.nsamples}", file=sys.stderr)
    print(model)
    logger.save_ocrmodel(model, loss=err, step=trainer.nsamples)


###
# Command Line
###


@app.command()
def train(
    training,
    training_args: str = "",
    training_bs: int = 2,
    display: float = -1.0,
    shuffle: int = 1000,
    model: str = "segmentation_model_210429",
    test: str = None,
    test_bs: int = 2,
    test_args: str = "",
    ntrain: int = int(1e12),
    ntest: int = int(1e12),
    schedule: str = "1e-3 * (0.9 ** (n//100000))",
    augmentation: str = "default",
    extensions: str = "png;image.png;framed.png;ipatch.png seg.png;target.png;lines.png;spatch.png",
    prefix: str = "ocroseg",
    weightmask: int = 0,
    bordermask: int = 16,
    num_workers: int = 1,
    log_to: str = "",
    parallel: bool = False,
    save_interval: float = 30 * 60,
    noutput: int = 4,
    invert: str = False,
    remap: str = "",
    device: str = None,
):
    global logger

    if log_to == "":
        log_to = None
    logger = slog.Logger(fname=log_to, prefix=prefix)
    logger.sysinfo()
    logger.save_config(
        dict(
            model=model,
            training=training,
            training_args=training_args,
            training_bs=training_bs,
            schedule=schedule,
        ),
    )
    logger.flush()

    if remap == "":
        remapper = np.zeros(noutput, dtype=np.uint8)
        for i in range(noutput):
            remapper[i] = i
    else:
        translate = eval("{" + remap + "}")
        noutput = np.max(list(translate.values())) + 1
        nclasses = np.max(list(translate.keys())) + 1
        print(f"noutput {noutput} nclasses {nclasses}")
        remapper = np.zeros(nclasses, dtype=np.uint8)
        for k, v in translate.items():
            remapper[k] = v

    kw = eval(f"dict({training_args})")
    training_dl = make_loader(
        training,
        batch_size=training_bs,
        extensions=extensions,
        shuffle=shuffle,
        augmentation=eval(f"augmentation_{augmentation}"),
        num_workers=num_workers,
        invert=invert,
        remapper=remapper,
        **kw,
    )
    (images, targets,) = next(iter(training_dl))
    if test is not None:
        kw = eval(f"dict({test_args})")
        test_dl = make_loader(test, batch_size=test_bs, **kw)
    else:
        test_dl = None

    model = loading.load_or_construct_model(model, noutput=noutput)
    if parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(model)

    trainer = SegTrainer(model, weightmask=weightmask, bordermask=bordermask, device=device)
    trainer.set_lr_schedule(eval(f"lambda n: {schedule}"))

    schedule = Schedule()

    for images, targets in repeatedly(training_dl):
        if trainer.nsamples >= ntrain:
            break
        assert float(targets.min()) >= 0
        assert float(targets.max()) <= noutput
        trainer.train_batch(images, targets)
        if schedule("progress", 60, initial=True):
            print_progress(trainer)
        if schedule("log", 600):
            log_progress(trainer)
        if schedule("save", save_interval, initial=True):
            save_model(logger, trainer, test_dl)
        if display > 0 and schedule("display", display):
            try:
                display_progress(trainer)
            except:
                traceback.print_exc()

    save_model(logger, trainer, test_dl)


@app.command()
def predict(
    fname: str,
    model: str,
    extensions: str = "png;image.png;framed.png;ipatch.png seg.png;target.png;lines.png;spatch.png",
    output: str = "",
    display: bool = True,
    limit: int = 999999999,
):
    model = loading.load_only_model(model)
    segmenter = Segmenter(model)

    pass # FIXME do something here

@app.command()
def segment(
    fname: str,
    model: str,
    extensions: str = "png;image.png;framed.png;ipatch.png;jpg;jpeg;JPEG",
    output: str = "",
    display: bool = True,
    limit: int = 999999999,
    device: str = None,
):
    device = utils.device(device)
    if device.type == "cpu":
        print("segment using CPU")
    model = loading.load_only_model(model)
    segmenter = Segmenter(model, device=device)

    dataset = wds.WebDataset(fname).decode("rgb")

    for sample in islice(dataset, 0, limit):
        image = wds.getfirst(sample, extensions)
        image = np.mean(image, 2)
        result = segmenter.segment(image)

        pass # FIXME do something here


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
