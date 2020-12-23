import itertools as itt
import os
import random
import sys
import re

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from numpy import amax, arange, mean, newaxis, tile
from scipy import ndimage as ndi
from torch import nn, optim
from torch.utils.data import DataLoader

import webdataset as wds
from torchmore import layers

from . import lineest, linemodels, slog
from .utils import Every, Charset

_ = linemodels


logger = slog.NoLogger()


app = typer.Typer()


min_w, min_h, max_w, max_h = 15, 15, 4000, 200


def goodsize(sample):
    image, txt = sample
    h, w = image.shape[-2:]
    good = h > min_h and h < max_h and w > min_w and w < max_w
    if not good:
        print("rejecting", image.shape)
    return good


plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


def model_device(model):
    """Find the device of a model."""
    return next(model.parameters()).device


def asnp(a):
    """Convert to numpy."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    else:
        assert isinstance(a, np.ndarray)
        return a


def nothing(*args, **kw):
    pass


def ctc_decode(probs, sigma=1.0, threshold=0.7, kind=None, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    probs = asnp(probs.T)
    assert (
        abs(probs.sum(1) - 1) < 1e-4
    ).all(), "input not normalized; did you apply .softmax()?"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, newaxis]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = tile(labels[:, newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


def pack_for_ctc(seqs):
    """Pack a list of sequences for nn.CTCLoss."""
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def collate4ocr(samples):
    """Collate image+sequence samples into batches.

    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    bh, bw, bd = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), bh, bw, bd), dtype=torch.float)
    for i, im in enumerate(images):
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        h, w, d = im.shape
        dy, dx = random.randint(0, bh - h), random.randint(0, bw - w)
        result[i, dy : dy + h, dx : dx + w, :d] = im
    return result, seqs


class LineTrainer:
    def __init__(self, model, *, lr=1e-4, device=None, savedir=True, maxgrad=10.0):
        super().__init__()
        self.model = model
        self.device = None
        self.losses = []
        self.last_lr = None
        self.set_lr(lr)
        self.clip_gradient = maxgrad
        self.nbatches = 0
        self.nsamples = 0
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.every_batch = []
        self.schedule = lambda n: None
        self.charset = None

    def set_lr(self, lr, momentum=0.9):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if lr is None:
            return
        if lr != self.last_lr:
            print(f"setting learning rate to {lr:4.1e}", file=sys.stderr)
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
        self.set_lr(self.schedule(self.nsamples))
        self.nbatches += 1
        self.nsamples += len(inputs)
        self.optimizer.zero_grad()
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = self.model.forward(inputs)
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (inputs.detach().cpu(), targets, outputs.detach().cpu())
        return loss.detach().item()

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch. Uses `probfn`."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def train(self, loader, epochs=1, learning_rates=None):
        """Train over a dataloader for the given number of epochs."""
        if learning_rates is None:
            learning_rates = [self.last_lr] * epochs
        for epoch, lr in enumerate(learning_rates):
            self.set_lr(lr)
            self.epoch = epoch
            for sample in loader:
                images, targets = sample
                loss = self.train_batch(images, targets)
                self.losses.append(float(loss))
                for f in self.every_batch:
                    f(self)

    def errors(self, loader, ntest=999999999):
        """Compute OCR errors using edit distance."""
        total = 0
        errors = 0
        for inputs, targets in loader:
            targets, tlens = pack_for_ctc(targets)
            predictions = self.predict_batch(inputs)
            start = 0
            for p, l in zip(predictions, tlens):
                t = targets[start : start + l].tolist()
                errors += editdistance.distance(p, t)
                total += len(t)
                start += l
            if total > ntest:
                break
        return errors, total

    def compute_loss(self, outputs, targets):
        assert len(targets) == len(outputs)
        targets, tlens = pack_for_ctc(targets)
        layers.check_order(outputs, "BDL")
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

    def predict_batch(self, inputs, **kw):
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result


class LineRec:
    def __init__(self, *, charset=Charset()):
        self.charset = charset

    def load_from_save(self, fname):
        result = torch.load(fname)
        print(f"# loaded {fname}", file=sys.stderr)
        self.charset = result.get("charset", self.charset)
        self.dewarp_to = result.get("dewarp_to", -1)
        if self.dewarp_to > 0:
            self.dewarper = lineest.CenterNormalizer(target_height=self.dewarp_to)
        else:
            self.dewarper = None
        mod = slog.load_module("model", result["msrc"])
        model = mod.make_model(96)
        model.load_state_dict(result["mstate"])
        model.eval()
        self.mod = mod
        self.model = model
        model.cuda()

    def activate(self, active):
        if active:
            self.model.cuda()
        else:
            self.model.cpu()

    def recognize(self, image, full=False):
        if image.dtype == np.uint8:
            image = image.astype(float) / 255.0
        assert np.amin(image) >= 0 and np.amax(image) <= 1
        if self.dewarper is not None:
            image = self.dewarper.measure_and_normalize(image)
            if image.shape[0] < 5 or image.shape[1] < 5:
                return None
        self.last_image = image
        batch = torch.FloatTensor(image.reshape(1, 1, *image.shape))
        self.probs = self.model(batch.cuda()).softmax(1)
        if not full:
            decoded = ctc_decode(self.probs[0])
            return self.charset.decode_str(decoded)
        else:
            h, w = image.shape
            decoded = [(r / float(w), self.charset.decode_chr(c), p) for r, c, p in ctc_decode(self.probs[0], full=True)]
            return decoded


def identity(x):
    return x


def invert_image(a):
    return 1.0 - a


def normalize_image(a):
    a = a - a.min()
    a /= a.max()
    return a


def normalize_none(s):
    return s


def normalize_simple(s):
    s = re.sub("\\\\[A-Za-z]+", "~", s)
    s = re.sub("\\\\[_^]+", "", s)
    s = re.sub("[{}]", "", s)
    s = re.sub(" +", " ", s)
    s = re.sub("\"", "''", s)
    return s.strip()


def make_loader(
    fname,
    batch_size=5,
    shuffle=1000,
    invert=False,
    normalize_intensity=False,
    ntrain=-1,
    mode="train",
    charset=Charset(),
    dewarp_to=-1,
    text_normalizer="simple",
    extensions="line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
    **kw,
):
    training = wds.Dataset(fname)
    if mode == "train" and shuffle > 0:
        training.shuffle(shuffle)
    training.decode("l8").to_tuple(extensions)
    text_normalizer = eval(f"normalize_{text_normalizer}")
    training.map_tuple(identity, text_normalizer)
    training.map_tuple(lambda a: a.astype(float) / 255.0, charset.preptargets)
    if invert:
        training.map_tuple(invert_image, identity)
    if normalize_intensity:
        training.map_tuple(normalize_image, identity)
    if dewarp_to > 0:
        dewarper = lineest.CenterNormalizer(target_height=dewarp_to)
        training.map_tuple(dewarper.measure_and_normalize, identity)
    training.map_tuple(lambda x: torch.tensor(x).unsqueeze(0), identity)
    training.select(goodsize)
    if ntrain > 0:
        print(ntrain)
        training = wds.ResizedDataset(training, length=ntrain)
    training_dl = DataLoader(
        training, collate_fn=collate4ocr, batch_size=batch_size, **kw
    )
    return training_dl


def print_progress(trainer):
    if trainer.nbatches % 50 != 0:
        return
    avgloss = mean(trainer.losses[-100:]) if len(trainer.losses) > 0 else 0.0
    print(
        f"{trainer.epoch:3d} {trainer.nbatches:9d} {avgloss:10.4f}",
        file=sys.stderr,
        flush=True,
    )


log_progress_every = Every(60)


def log_progress(trainer):
    if log_progress_every():
        avgloss = mean(trainer.losses[-100:]) if len(trainer.losses) > 0 else 0.0
        logger.scalar(
            "train/loss", avgloss, step=trainer.nsamples, json=dict(lr=trainer.last_lr)
        )
        logger.flush()


def display_progress(trainer):
    if trainer.nbatches % 50 != 0:
        return
    import matplotlib.pyplot as plt

    inputs, targets, outputs = trainer.last_batch
    inputs = inputs.numpy()[0, 0]
    outputs = outputs.softmax(1).numpy()[0]
    decoded = ctc_decode(outputs)
    plt.ion()
    plt.clf()
    plt.subplot(221)
    s = trainer.charset.decode_str(targets[0].numpy())
    plt.title(s)
    plt.imshow(inputs)
    plt.subplot(222)
    plt.imshow(outputs, vmin=0, vmax=1)
    plt.subplot(223)
    losses = np.array(trainer.losses)
    if len(losses) >= 20:
        losses = ndi.gaussian_filter(losses, 10.0)
    plt.ylim(np.amin(losses), np.median(losses) * 4)
    plt.plot(losses)
    plt.subplot(224)
    s = trainer.charset.decode_str(decoded)
    plt.title(s)
    for row in outputs:
        plt.plot(row)
    plt.ginput(1, 0.001)


def load_model(fname):
    assert fname is not None, "must provide file name to load model from"
    assert os.path.exists(fname), f"{fname} does not exist"
    assert fname.endswith(".py"), f"{fname} must be a .py file"
    src = open(fname).read()
    mod = slog.load_module("mmod", src)
    assert "make_model" in dir(
        mod
    ), f"{fname} source does not define make_model function"
    return mod, src


@app.command()
def train(
    training: str,
    training_bs: int = 3,
    epochs: int = 200,
    display: bool = False,
    invert: bool = False,
    normalize_intensity: bool = False,
    mdef: str = None,
    test: str = None,
    test_bs: int = 20,
    ntest: int = int(1e12),
    lr: float = 1e-3,
    schedule: str = "1e-3 * (0.9**((n//100000)**.5))",
    ntrain: int = -1,
    checkerr: float = 1e12,
    charset_file: str = None,
    dewarp_to: int = -1,
):

    charset = Charset(chardef=charset_file)

    mmod, msrc = load_model(mdef)

    logger = slog.Logger(prefix="ocroline")
    logger.sysinfo()
    logger.json(
        "args",
        dict(
            epochs=epochs,
            mdef=mdef,
            msrc=msrc,
            training=training,
            training_bs=training_bs,
            invert=invert,
            normalize_intensity=normalize_intensity,
            schedule=schedule,
        ),
    )
    logger.flush()

    training_dl = make_loader(
        training,
        batch_size=training_bs,
        invert=invert,
        normalize_intensity=normalize_intensity,
        ntrain=ntrain,
        dewarp_to=dewarp_to,
    )
    print(next(iter(training_dl))[0].size())
    if test is not None:
        test_dl = make_loader(
            test,
            batch_size=training_bs,
            invert=invert,
            normalize_intensity=normalize_intensity,
            dewarp_to=dewarp_to,
            mode="test",
        )

    model = mmod.make_model(len(charset))
    model.cuda()
    print(model)

    trainer = LineTrainer(model)
    trainer.charset = charset
    if schedule != "":
        trainer.schedule = eval(f"lambda n: {schedule}")
    else:
        trainer.set_lr(lr)
    trainer.every_batch.append(log_progress)
    trainer.every_batch.append(display_progress if display else print_progress)
    print("starting training")
    for epoch in range(epochs):
        trainer.train(training_dl)
        if test is not None:
            errors, total = trainer.errors(test_dl, ntest=ntest)
            err = float(errors) / total
            logger.scalar("val/err", err, step=trainer.nsamples)
            logger.save(
                "model",
                dict(
                    n=trainer.nsamples,
                    dewarp_to=dewarp_to,
                    mdef=mdef,
                    msrc=msrc,
                    charset=charset,
                    mstate=model.state_dict(),
                    ostate=trainer.optimizer.state_dict(),
                ),
                step=trainer.nsamples,
            )
            logger.flush()
            print("test set:", err, errors, total)
            assert err < checkerr


@app.command()
def recognize(
    model: str,
    fname: str,
    extensions: str = "png;jpg;line.png;line.jpg",
    invert: bool = True,
    normalize: bool = True,
):
    linerec = LineRec()
    linerec.load_from_save(model)
    linerec.invert = invert
    linerec.normalize = normalize
    dataset = wds.Dataset(fname)
    dataset.decode("l8").rename(image=extensions)
    plt.ion()
    for sample in dataset:
        image = sample["image"]
        if invert:
            image = invert_image(image)
        if normalize:
            image = normalize_image(image)
        if image.shape[0] < 10 or image.shape[1] < 10:
            print(sample.get("__key__", image.shape))
            continue
        result = linerec.recognize(image)
        plt.clf()
        plt.imshow(linerec.last_image)
        plt.title(result)
        plt.ginput(1, 1.0)
        print(sample.get("__key__"), image.shape, result)


@app.command()
def check(
    training: str = "data/words-simple-training.tar",
    training_bs: int = 5,
    epochs: int = 1,
    nbatches: int = 3000,
    display: bool = False,
    invert: bool = False,
    normalize_intensity: bool = False,
    model: str = "lstm2",
    check: str = "data/words-simple-test.tar",
    checkerr: float = 0.05,
    checking_bs: int = 20,
    ncheck: int = 1000,
):
    charset = Charset()
    training_dl = make_loader(
        training,
        batch_size=training_bs,
        invert=invert,
        normalize_intensity=normalize_intensity,
    )
    print(next(iter(training_dl))[0].size())

    model = eval(f"linemodels.make_{model}")(len(charset))
    model.cuda()
    print(model)

    trainer = LineTrainer(model)
    trainer.every_batch.append(display_progress if display else print_progress)
    print("starting training")
    trainer.train(itt.islice(iter(training_dl), 0, nbatches))

    if checkerr >= 0.0:
        checking_dl = make_loader(
            check,
            batch_size=checking_bs,
            invert=invert,
            normalize_intensity=normalize_intensity,
        )
        errors, total = trainer.errors(checking_dl, ntest=ncheck)
        err = float(errors) / total
        print(err, errors, total)
        assert err <= checkerr, "error rate too high"


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
