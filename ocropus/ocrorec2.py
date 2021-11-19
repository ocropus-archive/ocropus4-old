"""Text recognition."""

import io
import random
import re
from functools import partial
from itertools import islice

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import typer
import webdataset as wds
from numpy import amax, arange, newaxis, tile
from scipy import ndimage as ndi
from torch import nn
from torch.utils.data import DataLoader
from torchmore import layers

from . import degrade, lineest, linemodels, loading, slog, utils
from .utils import Charset, useopt

_ = linemodels


app = typer.Typer()


min_w, min_h, max_w, max_h = 15, 15, 4000, 200


def identity(x):
    return x


def goodsize(sample):
    """Determine whether the given sample has a good size."""
    image, _ = sample
    h, w = image.shape[-2:]
    good = h > min_h and h < max_h and w > min_w and w < max_w
    if not good:
        print("rejecting", image.shape)
    return good


class TextModel(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self, image):
        pass


plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")


def ctc_decode(probs, sigma=1.0, threshold=0.7, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    if not isinstance(probs, np.ndarray):
        probs = probs.detach().cpu().numpy()
    probs = probs.T
    delta = np.amax(abs(probs.sum(1) - 1))
    assert delta < 1e-4, f"input not normalized ({delta}); did you apply .softmax()?"
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


def log_matplotlib_figure(tb, fig, index, key="image"):
    """Log the given matplotlib figure to tensorboard logger tb."""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = image.convert("RGB")
    image = image.resize((600, 600))
    image = np.array(image)
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)
    tb.experiment.add_image(key, image, index)


class TextLightning(pl.LightningModule):
    """A class encapsulating the logic for training text line recognizers."""

    def __init__(self, model, *, lr=3e-4, device=None, maxgrad=10.0):
        """A class encapsulating line training logic.

        :param model: the model to be trained
        :param lr: learning rate, defaults to 1e-4
        :param device: GPU used for training, defaults to None
        :param maxgrad: gradient clipping, defaults to 10.0
        """
        super().__init__()
        self.lr = lr
        self.model = model
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.charset = None
        self.schedule = utils.Schedule()

    def training_step(self, batch, index):
        inputs, targets = batch
        outputs = self.model.forward(inputs)
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        self.log("train_loss", loss, on_step=True)
        if index % 100 == 0:
            self.log_ocr_result(index, inputs, targets, outputs)
        return loss

    def log_ocr_result(self, index, inputs, targets, outputs):
        """Log the given inputs, targets, and outputs to the logger."""
        inputs = inputs.detach().cpu().numpy()[0]
        outputs = outputs.detach().softmax(1).cpu().numpy()[0]
        decoded = ctc_decode(outputs)
        decode_str = Charset().decode_str
        t = decode_str(targets[0].cpu().numpy())
        s = decode_str(decoded)
        figure = plt.figure(figsize=(10, 10))
        # log the OCR result for the first image in the batch
        plt.clf()
        plt.imshow(inputs[0], cmap=plt.cm.gray)
        plt.title(f"{t} : {s}", size=48)
        log_matplotlib_figure(self.logger, figure, index)
        # plot the posterior probabilities for the first image in the batch
        plt.clf()
        for row in outputs:
            plt.plot(row)
        log_matplotlib_figure(self.logger, figure, index, key="probs")
        plt.close(figure)

    def compute_loss(self, outputs, targets):
        assert len(targets) == len(outputs)
        targets, tlens = pack_for_ctc(targets)
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

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

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def predict_batch(self, inputs, **kw):
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result


def invert_image(a):
    return 1.0 - a


def normalize_image(a):
    a = a - a.min()
    a = a / float(a.max())
    return a


@useopt
def normalize_none(s):
    return s


@useopt
def normalize_simple(s):
    s = re.sub("\\\\[A-Za-z]+", "~", s)
    s = re.sub("\\\\[_^]+", "", s)
    s = re.sub("[{}]", "", s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', "''", s)
    return s.strip()


def good_text(regex, sample):
    image, txt = sample
    return re.search(regex, txt)


def augment_transform(image, p=0.5):
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    return image


def augment_distort(image, p=0.5):
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
    return image


def make_loader(
    fname,
    batch_size=5,
    shuffle=5000,
    invert=False,
    normalize_intensity=False,
    ntrain=-1,
    mode="train",
    charset=Charset(),
    augment="distort",
    text_normalizer="simple",
    text_select_re="[0-9A-Za-z]",
    extensions="line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
    **kw,
):
    training = wds.WebDataset(fname, caching=True, verbose=True, shardshuffle=50)
    if mode == "train" and shuffle > 0:
        training = training.shuffle(shuffle)
    training = training.decode("l8").to_tuple(extensions)
    text_normalizer = eval(f"normalize_{text_normalizer}")
    training = training.map_tuple(identity, text_normalizer)
    if text_select_re != "":
        training = training.select(partial(good_text, text_select_re))
    training = training.map_tuple(lambda a: a.astype(float) / 255.0, charset.preptargets)
    if augment != "":
        f = eval(f"augment_{augment}")
        training = training.map_tuple(f, identity)
    training = training.map_tuple(lambda x: torch.tensor(x).unsqueeze(0), identity)
    training = training.select(goodsize)
    if ntrain > 0:
        print(ntrain)
        training = training.with_epoch(ntrain)
    training_dl = DataLoader(training, collate_fn=collate4ocr, batch_size=batch_size, **kw)
    return training_dl


class TextModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        b, c, h, w = images.shape
        assert h > 15 and h < 4000 and w > 15 and h < 4000
        for image in images:
            image -= image.amin()
            image /= torch.max(image.amax(), torch.tensor([0.01], device=image.device))
            if image.mean() > 0.5:
                image[...] = 1.0 - image[...]
        return self.model.forward(images)


default_training_urls = (
    "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{00..22}.tar"
)


@app.command()
def train(
    training: str = default_training_urls,
    training_bs: int = 4,
    invert: bool = False,
    normalize_intensity: bool = False,
    model: str = "text_model_210910",
    test: str = None,
    test_bs: int = 20,
    ntest: int = int(1e12),
    schedule: str = "3e-4 * (0.9**((n//200000)**.5))",
    text_select_re: str = "[A-Za-z0-9]",
    # lr: float = 1e-3,
    # checkerr: float = 1e12,
    charset_file: str = None,
    log_to: str = "",
    ntrain: int = (1 << 31),
    num_workers: int = 4,
    data_parallel: str = "",
    shuffle: int = 20000,
    device: str = None,
):

    device = utils.device(device)
    charset = Charset(chardef=charset_file)

    if log_to == "":
        log_to = None
    logger = slog.Logger(fname=log_to, prefix="ocrorec")
    logger.save_config(
        dict(
            mdef=model,
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
        text_select_re=text_select_re,
        num_workers=4,
        shuffle=shuffle,
    )
    print(next(iter(training_dl))[0].size())
    if test is not None:
        test_dl = make_loader(
            test,
            batch_size=training_bs,
            invert=invert,
            normalize_intensity=normalize_intensity,
            mode="test",
        )
    else:
        test_dl = None

    model = loading.load_or_construct_model(model, len(charset))
    model = TextModel(model)

    lmodel = TextLightning(model)
    callbacks = []
    trainer = pl.Trainer(
        default_root_dir="_logs",
        gpus=1,
        max_epochs=1000,
        callbacks=callbacks,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(lmodel, training_dl)


@app.command()
def recognize(
    fname: str,
    extensions: str = "png;jpg;line.png;line.jpg",
    model: str = "",
    invert: str = "Auto",
    limit: int = 999999999,
    normalize: bool = True,
    display: bool = True,
    device: str = None,
):
    model = loading.load_only_model(model)
    textrec = TextRec(model, device=device)
    textrec.invert = invert
    textrec.normalize = normalize
    dataset = wds.WebDataset(fname)
    dataset = dataset.decode("l8").rename(image=extensions)
    plt.ion()
    for sample in islice(dataset, limit):
        image = sample["image"]
        if invert:
            image = utils.autoinvert(image, invert)
        if normalize:
            image = normalize_image(image)
        if image.shape[0] < 10 or image.shape[1] < 10:
            print(sample.get("__key__", image.shape))
            continue
        result = textrec.recognize(image)
        if display:
            plt.clf()
            plt.imshow(textrec.last_image)
            plt.title(result)
            plt.ginput(1, 1.0)
            print(sample.get("__key__"), image.shape, result)


@app.command()
def toscript(
    model: str,
):
    import torch.jit

    model = loading.load_only_model(model)
    scripted = torch.jit.script(model)
    print(scripted)


@app.command()
def noop():
    pass


if __name__ == "__main__":
    app()
