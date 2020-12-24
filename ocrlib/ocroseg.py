import os
import sys
import time

import typer
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import amin, median, mean
from scipy import ndimage as ndi
from torch import nn, optim
from torch.utils.data import DataLoader
from webdataset import Dataset
import torchmore.layers

from . import slog
from . import utils


logger = slog.NoLogger()

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


###
# Helpers
###


app = typer.Typer()


class Every(object):
    """Do something every few seconds/minutes/hours."""

    def __init__(self, interval):
        self.interval = interval
        self.last = 0

    def __call__(self):
        now = time.time()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False


def nothing(*args, **kw):
    """Do nothing (for callbacks)."""
    return None


def model_device(model):
    """Find the device of a model."""
    return next(model.parameters()).device


def scale_to(a, shape, order=0):
    """Scale a numpy array to a given target size."""
    scales = np.array(a.shape, "f") / np.array(shape, "f")
    result = ndi.affine_transform(a, np.diag(scales), output_shape=shape, order=order)
    return result


def cpu(t):
    return t.detach().cpu()


def modimage(image, mod):
    h, w = image.shape[-2:]
    hm, wm = (h // mod) * mod, (w // mod) * mod
    return image[..., :hm, :wm]


###
# Loading and Preprocessing
###


def preproc(scale, extra_target_scale=1.0, mod=16):
    def f(pair):
        image, seg = pair
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        assert np.amax(image) <= 1.0
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        assert np.amax(seg) < 16, "max # classes for segmentation is set to 16"
        image = ndi.zoom(image, scale, order=1)
        seg = ndi.zoom(seg, scale * extra_target_scale, order=0)
        image = torch.tensor(image).unsqueeze(0)
        seg = torch.tensor(seg).long()
        return modimage(image, mod), modimage(seg, mod)

    return f


augmentation_none = None


def augmentation_gray(sample):
    image, target = sample
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if target.ndim == 3:
        target = target[..., 0]
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64)
    assert image.shape[:2] == target.shape[:2]
    if np.random.uniform() > 0.5:
        image = utils.simple_bg_fg(
            image,
            amplitude=np.random.uniform(0.01, 0.3),
            imsigma=np.random.uniform(0.01, 2.0),
            sigma=np.random.uniform(0.5, 10.0),
        )
    return image, target


def augmentation_page(sample, max_distortion=0.05):
    image, target = sample
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if target.ndim == 3:
        target = target[..., 0]
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64)
    assert image.shape[:2] == target.shape[:2]
    d = min(image.shape[0], image.shape[1]) * max_distortion
    src = [
        np.random.uniform(-d, d),
        image.shape[0] + np.random.uniform(-d, d),
        np.random.uniform(-d, d),
        image.shape[1] + np.random.uniform(-d, d),
    ]
    image = utils.get_affine_patch(image, image.shape[:2], src, order=1)
    if np.random.uniform() > 0.5:
        image = utils.simple_bg_fg(
            image,
            amplitude=np.random.uniform(0.01, 0.3),
            imsigma=np.random.uniform(0.01, 2.0),
            sigma=np.random.uniform(0.5, 10.0),
        )
    target = utils.get_affine_patch(target, target.shape[:2], src, order=0)
    return image, target


def make_loader(
    urls,
    batch_size=2,
    extensions="image.png;framed.png;ipatch.png target.png;lines.png;spatch.png",
    scale=0.5,
    extra_target_scale=1.0,
    augmentation=None,
    output_scale=None,
    shuffle=0,
    mod=16,
    num_workers=1,
):
    training = Dataset(urls).shuffle(shuffle).decode("rgb8").to_tuple(extensions)
    if augmentation is not None:
        training.map(augmentation)
    training.map(preproc(scale, extra_target_scale=extra_target_scale, mod=mod))

    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


def load_model(fname):
    assert fname is not None, "provide model with --mdef or --load"
    assert os.path.exists(fname), f"{fname} does not exist"
    assert fname.endswith(".py"), f"{fname} must be a .py file"
    src = open(fname).read()
    mod = slog.load_module("mmod", src)
    assert "make_model" in dir(
        mod
    ), f"{fname} source does not define make_model function"
    return mod, src


###
# Logging
###

mem_every = Every(60)


def memsum(trainer):
    """Output current memory usage every minute. (Callback)"""
    if not mem_every():
        return
    print(torch.cuda.memory_summary())


log_progress_every = Every(60)


def log_progress(trainer):
    """Log training progress every minute. (Callback)"""
    if not log_progress_every():
        return
    avgloss = mean(trainer.losses[-100:]) if len(trainer.losses) > 0 else 0.0
    logger.scalar(
        "train/loss", avgloss, step=trainer.nsamples, json=dict(lr=trainer.last_lr)
    )
    logger.flush()


print_progress_every = Every(60)


def print_progress(self):
    if not print_progress_every():
        return
    print(
        f"# {len(self.losses)} {np.median(self.losses[-10:])}",
        file=sys.stderr,
        flush=True,
    )


def display_progress(self):
    import matplotlib.pyplot as plt

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
        ax1.imshow(mask * 0.3 + doc * 0.7, cmap="gray")
    else:
        ax1.imshow(doc, cmap="gray")
    p = outputs.detach().cpu().softmax(1)
    b, d, h, w = outputs.size()
    result = p.numpy()[0].transpose(1, 2, 0)
    if result.shape[2] > 3:
        result = result[..., 1:4]
    else:
        result = result[..., :3]
    ax2.imshow(result, vmin=0, vmax=1)
    m = result.shape[1] // 2
    ax2.plot([m, m], [0, h], color="white", alpha=0.5)
    t = targets[0].detach().numpy()
    t = np.array([t == 1, t == 2, t == 3]).astype(float).transpose(1, 2, 0)
    ax3.imshow(t)
    ax6.imshow(targets[0].detach().numpy(), vmin=0, vmax=5, cmap=plt.cm.viridis)
    if len(self.losses) >= 100:
        losses = ndi.gaussian_filter(self.losses, 10.0)
        losses = losses[::10]
        losses = ndi.gaussian_filter(losses, 10.0)
        ax4.plot(losses)
        ax4.set_ylim((0.9 * amin(losses), median(losses) * 3))
    colors = "black r g b yellow".split()
    for i in range(0, d):
        ax5.plot(p[0, i, :, m], color=colors[i])
    plt.ginput(1, 0.001)


###
# Training
###


class SegTrainer:
    def __init__(
        self,
        model,
        *,
        lossfn=None,
        lr=1e-4,
        every=3.0,
        device=None,
        savedir=True,
        maxgrad=10.0,
        margin=16,
        weightmask=-1,
        **kw,
    ):
        super().__init__()
        self.model = model
        self.device = None
        # self.lossfn = nn.CTCLoss()
        self.lossfn = nn.CrossEntropyLoss()
        self.every = every
        self.losses = []
        self.last_lr = None
        self.set_lr(lr)
        self.clip_gradient = maxgrad
        self.charset = None
        self.margin = margin
        self.last_display = 0
        self.every_batch = []
        self.nsamples = 0
        self.nbatches = 0
        self.weightmask = weightmask
        self.weightlayer = torchmore.layers.WeightedGrad()
        self.last_mask = None

    def set_lr(self, lr, momentum=0.9):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if lr != self.last_lr:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum
            )
            self.last_lr = lr

    def train_batch(self, inputs, targets):
        """All the steps necessary for training a batch.

        Stores the last batch in self.last_batch.
        Adds the loss to self.losses.
        Clips the gradient if self.clip_gradient is not None.
        """
        self.model.train()
        self.optimizer.zero_grad()
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = self.model.forward(inputs)
        if self.weightmask >= 0:
            mask = targets.detach().numpy()
            assert mask.ndim == 3
            mask = (mask >= 0.5).astype(float)
            if self.weightmask > 0:
                w = self.weightmask
                mask = ndi.maximum_filter(mask, (0, w, w), mode="constant")
            mask = torch.tensor(mask)
            self.last_mask = mask
            outputs = self.weightlayer.forward(outputs, mask)
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (cpu(inputs), cpu(targets), cpu(outputs))
        self.nsamples += len(inputs)
        self.nbatches += 1
        return loss.detach().item()

    def compute_loss(self, outputs, targets):
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
        loss = self.lossfn(outputs, targets.to(outputs.device))
        return loss

    def train_epoch(self, loader, ntrain=int(1e12)):
        """Train over a dataloader for the given number of epochs."""
        start = self.nsamples
        for sample in loader:
            images, targets = sample
            loss = self.train_batch(images, targets)
            self.losses.append(float(loss))
            for f in self.every_batch:
                f(self)
            if self.nsamples - start >= ntrain:
                break

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
        return np.mean(errors)


###
# Inference
###


class Segmenter:
    def __init__(self, smooth=1.0, scale=0.5, preproc=None):
        if isinstance(smooth, (int, float)):
            self.smooth = (float(smooth), float(smooth), 0)
        else:
            assert len(smooth) == 2
            self.smooth = list(smooth) + [0]
        self.zoom = scale
        self.preproc = preproc or torchmore.layers.ModPad(8)
        self.preproc.eval()

    def load_from_save(
        self, fname, args={},
    ):
        result = torch.load(fname)
        mod = slog.load_module("model", result["msrc"])
        model = mod.make_model(**args)
        model.load_state_dict(result["mstate"])
        model.eval()
        self.model = model
        self.activate()

    def load_from_log(self, fname, args={}):
        log = slog.Logger(fname)
        result = log.load_last()
        mod = slog.load_module("model", result["msrc"])
        model = mod.make_model(**args)
        model.load_state_dict(result["mstate"])
        model.eval()
        self.model = model
        self.activate()

    def activate(self, yes=True):
        if yes:
            self.model.cuda()
            self.preproc.cuda()
        else:
            self.model.cpu()
            self.model.cpu()

    def get_targets(self, classes):
        return classes >= 2

    def segment(self, page, nocheck=False, unzoom=True):
        assert page.ndim == 2
        assert page.shape[0] >= 100 and page.shape[0] < 20000
        assert page.shape[1] >= 100 and page.shape[1] < 20000
        self.page = page
        if not nocheck:
            assert np.median(page) < 0.5, "text should be white on black"
        self.activate()
        zoomed = ndi.zoom(page, self.zoom, order=1)
        batch = torch.FloatTensor(zoomed).unsqueeze(0).unsqueeze(0)
        batch = batch.cuda()
        batch = self.preproc(batch)
        with torch.no_grad():
            probs = (
                self.model(batch)
                .softmax(1)
                .detach()
                .cpu()
                .numpy()[0]
                .transpose(1, 2, 0)
            )
        if unzoom:
            probs = ndi.zoom(probs, (1.0 / self.zoom, 1.0 / self.zoom, 1.0), order=1)
        self.probs = probs
        maxima = np.amax(np.amax(probs, 0), 0)
        assert maxima.shape == (4,)
        gprobs = ndi.gaussian_filter(probs, self.smooth)
        gmaxima = np.amax(np.amax(gprobs, 0), 0)
        gprobs /= (maxima / gmaxima)[np.newaxis, np.newaxis, :]
        gprobs = gprobs / gprobs.sum(2)[:, :, np.newaxis]
        self.gprobs = gprobs
        classes = np.argmax(gprobs, 2)
        self.segments = self.get_targets(classes)
        labels, n = ndi.label(self.segments)
        return [
            (obj[0].start, obj[0].stop, obj[1].start, obj[1].stop)
            for obj in ndi.find_objects(labels)
        ]


class LineSegmenter(Segmenter):
    def __init__(self, smooth=(1, 4)):
        super().__init__(smooth=smooth)

    def get_targets(self, classes):
        return classes == 2


def extract_boxes(page, boxes, pad=5):
    for y0, y1, x0, x1 in boxes:
        h, w = y1 - y0, x1 - x0
        word = ndi.affine_transform(
            page,
            np.eye(2),
            output_shape=(h + 2 * pad, w + 2 * pad),
            offset=(y0 - pad, x0 - pad),
            order=0,
        )
        yield word


###
# Command Line
###


@app.command()
def train(
    training,
    training_args: str = "",
    training_bs: int = 2,
    epochs: int = 200,
    display: bool = False,
    shuffle: int = 1000,
    mdef: str = None,
    test: str = None,
    test_bs: int = 2,
    test_args: str = "",
    ntrain: int = int(1e12),
    ntest: int = int(1e12),
    schedule: str = "1e-3 * (0.9 ** (n//100000))",
    zoom: float = 0.5,
    augmentation: str = "none",
    extensions: str = "png;image.png;framed.png;ipatch.png seg.png;target.png;lines.png;spatch.png",
    prefix: str = "ocroseg",
    weightmask: int = -1,
    num_workers: int = 1,
    log_to: str = "",
):
    global logger

    mmod, msrc = load_model(mdef)

    if log_to == "":
        log_to = None
    logger = slog.Logger(fname=log_to, prefix=prefix)
    logger.sysinfo()
    logger.json(
        "args",
        dict(
            epochs=epochs,
            mdef=mdef,
            msrc=msrc,
            training=training,
            training_args=training_args,
            training_bs=training_bs,
            schedule=schedule,
        ),
    )
    logger.flush()

    kw = eval(f"dict({training_args})")
    training_dl = make_loader(
        training,
        batch_size=training_bs,
        extensions=extensions,
        shuffle=shuffle,
        augmentation=eval(f"augmentation_{augmentation}"),
        num_workers=num_workers,
        **kw,
    )
    images, targets, = next(iter(training_dl))
    if test is not None:
        kw = eval(f"dict({test_args})")
        test_dl = make_loader(test, batch_size=test_bs, **kw)

    model = mmod.make_model()
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(model)

    trainer = SegTrainer(model, zoom=zoom, weightmask=weightmask)
    trainer.schedule = eval(f"lambda n: {schedule}")
    trainer.every_batch.append(log_progress)
    trainer.every_batch.append(print_progress)
    # trainer.every_batch.append(memsum)
    if display:
        trainer.every_batch.append(display_progress)
    for epoch in range(epochs):
        print("=== training")
        trainer.train_epoch(training_dl, ntrain=ntrain)
        if test is not None:
            print("=== testing")
            err = trainer.errors(test_dl, ntest=ntest)
            logger.scalar("val/err", err, step=trainer.nsamples)
        else:
            err = np.mean(trainer.losses[-100:])
        state = dict(
            mdef=mdef,
            msrc=msrc,
            mstate=model.state_dict(),
            ostate=trainer.optimizer.state_dict(),
        )
        logger.save("model", state, scalar=err, step=trainer.nsamples)
        print(
            "epoch",
            epoch,
            "nsamples",
            trainer.nsamples,
            "err",
            err,
            "lr",
            trainer.last_lr,
        )


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
