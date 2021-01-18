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

import ocrlib.patches
from ocrlib.utils import Schedule, repeatedly
from . import slog
from . import utils
from . import loading
import ocrlib.utils


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


def preproc(scale, extra_target_scale=1.0, mod=16):
    def f(pair):
        image, seg = pair
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.float32:
            pass
        else:
            raise ValueError(f"{image.dtype}: unknown dtype")
        assert np.amax(image) <= 1.0
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        assert np.amax(seg) < 16, "max # classes for segmentation is set to 16"
        image = ndi.zoom(image, scale, order=1)
        seg = ndi.zoom(seg, scale * extra_target_scale, order=0)
        image = torch.tensor(image).unsqueeze(0)
        seg = torch.tensor(seg).long()
        # return modimage(image, mod), modimage(seg, mod)
        return image, seg

    return f


def augmentation_none(sample):
    image, target = sample
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 4:
        image = np.mean(image, 3)
    if target.ndim == 3:
        target = target[..., 0]
    assert image.ndim == 3
    assert target.ndim == 2
    assert image.dtype == np.float32
    assert target.dtype in (np.uint8, np.int16, np.int32, np.int64)
    assert image.shape[:2] == target.shape[:2]
    return image, target


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
        image = simple_bg_fg(
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
    image = ocrlib.patches.get_affine_patch(image, image.shape[:2], src, order=1)
    if np.random.uniform() > 0.5:
        image = simple_bg_fg(
            image,
            amplitude=np.random.uniform(0.01, 0.3),
            imsigma=np.random.uniform(0.01, 2.0),
            sigma=np.random.uniform(0.5, 10.0),
        )
    target = ocrlib.patches.get_affine_patch(target, target.shape[:2], src, order=0)
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


def make_loader(
    urls,
    batch_size=2,
    extensions="image.png;framed.png;ipatch.png target.png;lines.png;spatch.png",
    scale=0.5,
    extra_target_scale=1.0,
    augmentation=augmentation_none,
    output_scale=None,
    shuffle=0,
    mod=16,
    num_workers=1,
):
    training = Dataset(urls).shuffle(shuffle).decode("rgb8").to_tuple(extensions)
    training.map(augmentation)
    training.map(np2tensor)
    return DataLoader(training, batch_size=batch_size, num_workers=num_workers)


###
# Logging
###

mem_every = ocrlib.utils.Every(60)


def memsum(trainer):
    """Output current memory usage every minute. (Callback)"""
    if not mem_every():
        return
    print(torch.cuda.memory_summary())


log_progress_every = ocrlib.utils.Every(60)


def log_progress(trainer):
    """Log training progress every minute. (Callback)"""
    if not log_progress_every():
        return
    avgloss = mean(trainer.losses[-100:]) if len(trainer.losses) > 0 else 0.0
    logger.scalar(
        "train/loss", avgloss, step=trainer.nsamples, json=dict(lr=trainer.last_lr)
    )
    logger.flush()


print_progress_every = ocrlib.utils.Every(60)


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
        self.lr_schedule = None
        self.clip_gradient = maxgrad
        self.charset = None
        self.margin = margin
        self.last_display = 0
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
        self.last_batch = (
            inputs.detach().cpu(),
            targets.detach().cpu(),
            outputs.detach().cpu(),
        )
        self.nsamples += len(inputs)
        self.nbatches += 1
        loss = loss.detach().item()
        self.losses.append(loss)
        return loss

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


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = ndi.distance_transform_edt(
        labels == 0, return_distances=1, return_indices=1
    )
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


def full_inference(batch, model, *args, **kw):
    with torch.no_grad():
        return model(batch).detach().softmax(1).cpu()


def patchwise_inference(batch, model, size=(400, 1600), overlap=0):
    if np.prod(batch.shape[-2:]) <= np.prod(size):
        return full_inference(batch, model)
    assert overlap == 0, "nonzero overlap not implemented yet"
    h, w = batch.shape[-2:]
    result = None
    for y in range(0, h, size[0]):
        for x in range(0, w, size[1]):
            patch = batch[..., y : y + size[0], x : x + size[1]]
            print("patch", patch.shape)
            probs = full_inference(patch, model)
            if result is None:
                nshape = list(probs.shape[:-2]) + list(batch.shape[-2:])
                print("*", nshape)
                result = torch.zeros(nshape)
            print("probs", probs.shape)
            print(result.shape, y, x)
            result[..., y : y + size[0], x : x + size[1]] = probs.detach().cpu()
    return result


class Segmenter:
    def __init__(self, model, scale=0.5, preproc=None):
        self.smooth = 0.0
        self.zoom = scale
        self.preproc = preproc or torchmore.layers.ModPad(8)
        self.preproc.eval()
        self.model = model
        self.marker_threshold = 0.3
        self.region_threshold = 0.3
        self.maxdist = 100
        self.activate()

    def activate(self, yes=True):
        if yes:
            self.model.cuda()
            self.preproc.cuda()
        else:
            self.model.cpu()
            self.model.cpu()

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
        probs = patchwise_inference(batch, self.model)
        probs = probs.numpy()[0].transpose(1, 2, 0)
        if unzoom:
            probs = ndi.zoom(probs, (1.0 / self.zoom, 1.0 / self.zoom, 1.0), order=1)
        self.probs = probs
        self.gprobs = smooth_probabilities(probs, self.smooth)
        self.segments = marker_segmentation(
            self.gprobs[..., 3] > self.marker_threshold,
            self.gprobs[..., 2] > self.region_threshold,
            self.maxdist,
        )
        return [
            (obj[0].start, obj[0].stop, obj[1].start, obj[1].stop)
            for obj in ndi.find_objects(self.segments)
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
            page,
            np.eye(2),
            output_shape=(h + 2 * pad, w + 2 * pad),
            offset=(y0 - pad, x0 - pad),
            order=0,
        )
        yield word


def save_model(logger, trainer, test_dl, ntest=999999):
    model = trainer.model
    if test_dl is not None:
        print("# testing", file=sys.stderr)
        err = trainer.errors(test_dl, ntest=ntest)
        logger.scalar("val/err", err, step=trainer.nsamples)
    else:
        err = np.mean(trainer.losses[-100:])
    print(f"# saving {trainer.nsamples}", file=sys.stderr)
    print(model)
    loading.log_model(logger, model, loss=err, step=trainer.nsamples)


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
    model: str = "segmentation_model_210117",
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
    parallel: bool = False,
    save_interval: float = 30 * 60,
):
    global logger

    if log_to == "":
        log_to = None
    logger = slog.Logger(fname=log_to, prefix=prefix)
    logger.sysinfo()
    logger.json(
        "args",
        dict(
            epochs=epochs,
            model=model,
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
    (
        images,
        targets,
    ) = next(iter(training_dl))
    if test is not None:
        kw = eval(f"dict({test_args})")
        test_dl = make_loader(test, batch_size=test_bs, **kw)
    else:
        test_dl = None

    model = loading.load_or_construct_model(model)
    model.cuda()
    if parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(model)

    trainer = SegTrainer(model, zoom=zoom, weightmask=weightmask)
    trainer.set_lr_schedule(eval(f"lambda n: {schedule}"))

    schedule = Schedule()

    for images, targets in repeatedly(training_dl):
        if trainer.nsamples >= ntrain:
            break
        trainer.train_batch(images, targets)
        if schedule("progress", 60, initial=True):
            print_progress(trainer)
        if schedule("log", 600):
            log_progress(trainer)
        if schedule("save", save_interval, initial=True):
            save_model(logger, trainer, test_dl)
        if display and schedule("display", 15):
            display_progress(trainer)

    save_model(logger, trainer, test_dl)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
