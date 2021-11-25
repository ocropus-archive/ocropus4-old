"""Text recognition."""

import sys
import os
import io
import random
import re
from functools import partial
from itertools import islice
from io import StringIO
import yaml

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import LambdaLR
import torch
import typer
import webdataset as wds
from numpy import amax, arange, newaxis, tile
from scipy import ndimage as ndi
from torch import nn
from torch.utils.data import DataLoader
from torchmore import layers

from . import degrade, lineest, linemodels, loading, slog, utils
from .utils import useopt
import torch.jit

_ = linemodels


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


@useopt
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


@useopt
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


class TextDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_shards = None,
        val_shards = None,
        train_bs: int = 4,
        val_bs: int = 20,
        text_select_re: str = "[A-Za-z0-9]",
        nepoch: int = 5000,
        num_workers: int = 4,
        cache_size: int = -1,
        cache_dir: str = "./_cache",
        **kw,
    ):
        super().__init__()
        assert train_shards is not None
        if val_shards == "":
            val_shards = None
        self.params = locals()
        self.cache_size = cache_size
        self.cache_dir = cache_dir

    def make_loader(
        self,
        fname,
        batch_size=5,
        shuffle=5000,
        nepoch=5000,
        mode="train",
        augment="distort",
        text_normalizer="simple",
        text_select_re="[0-9A-Za-z]",
        extensions="line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
        cache=True,
        **kw,
    ):
        ds = wds.WebDataset(
            fname,
            cache_size=float(self.cache_size),
            cache_dir=self.cache_dir,
            verbose=True,
            shardshuffle=50,
            resampled=True,
        )
        if mode == "train" and shuffle > 0:
            ds = ds.shuffle(shuffle)
        ds = ds.decode("l8").to_tuple(extensions)
        text_normalizer = eval(f"normalize_{text_normalizer}")
        ds = ds.map_tuple(identity, text_normalizer)
        if text_select_re != "":
            ds = ds.select(partial(good_text, text_select_re))
        ds = ds.map_tuple(lambda a: a.astype(float) / 255.0, None)
        if augment != "":
            f = eval(f"augment_{augment}")
            ds = ds.map_tuple(f, identity)
        ds = ds.map_tuple(lambda x: torch.tensor(x).unsqueeze(0), identity)
        ds = ds.select(goodsize)
        if nepoch > 0:
            ds = ds.with_epoch(nepoch)
        dl = DataLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=batch_size,
            shuffle=False,
            **kw,
        )
        return dl

    def train_dataloader(self):
        return self.make_loader(
            self.params["train_shards"],
            batch_size=self.params["train_bs"],
            mode="train",
        )

    def val_dataloader(self):
        if self.params.get("val_shards") is None:
            return None
        return self.make_loader(
            self.params["val_shards"],
            batch_size=self.params["val_bs"],
            mode="val",
            augment="",
        )


class TextModel(nn.Module):
    @torch.jit.export
    @staticmethod
    def charset_size():
        return 128

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        result = torch.tensor([ord(c) for c in s], dtype=torch.long)
        result[result >= self.charset_size()] = 127
        return result

    @torch.jit.export
    def decode_str(self, a: torch.Tensor) -> str:
        return "".join([chr(int(c)) for c in a])

    @torch.jit.export
    def forward(self, images):
        b, c, h, w = images.shape
        assert c == 1 or c == 3
        if c == 1:
            images = images.repeat(1, 3, 1, 1)
            b, c, h, w = images.shape
        assert h > 15 and h < 4000 and w > 15 and h < 4000
        for image in images:
            image -= image.amin()
            image /= torch.max(image.amax(), torch.tensor([0.01], device=image.device))
            if image.mean() > 0.5:
                image[:, :, :] = 1.0 - image[:, :, :]
        return self.model.forward(images)


class TextLightning(pl.LightningModule):
    """A class encapsulating the logic for training text line recognizers."""

    def __init__(
        self,
        model,
        *,
        display_freq=1000,
        lr=3e-4,
        lr_halflife=1000,
        config={},
    ):
        super().__init__()
        self.display_freq = display_freq
        self.config = config
        self.model = model
        self.lr = lr
        self.lr_halflife = lr_halflife
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.total = 0

    def on_train_start(self):
        print(self.config)
        self.logger.log_hyperparams(self.config)

    def forward(self, inputs):
        return self.model.forward(inputs)

    def training_step(self, batch, index):
        inputs, text_targets = batch
        outputs = self.forward(inputs)
        targets = [self.model.encode_str(s) for s in text_targets]
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        self.log("train_loss", loss)
        err = self.compute_error(outputs, targets)
        self.log("train_err", err, prog_bar=True)
        if index % self.display_freq == 0:
            self.log_results(index, inputs, targets, outputs)
        self.total += len(inputs)
        return loss

    def validation_step(self, batch, index):
        inputs, text_targets = batch
        outputs = self.forward(inputs)
        targets = [self.model.encode_str(s) for s in text_targets]
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        self.log("val_loss", loss)
        err = self.compute_error(outputs, targets)
        self.log("val_err", err)
        return loss

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

    def compute_error(self, outputs, targets):
        probs = outputs.detach().cpu().softmax(1)
        targets = [[int(x) for x in t] for t in targets]
        total = sum(len(t) for t in targets)
        predicted = [ctc_decode(p) for p in probs]
        errs = [editdistance.distance(p, t) for p, t in zip(predicted, targets)]
        return sum(errs) / float(total)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, self.schedule)
        return [optimizer], [scheduler]

    def schedule(self, epoch):
        return 0.5 ** (epoch // self.lr_halflife)

    def log_results(self, index, inputs, targets, outputs):
        self.show_ocr_results(index, inputs, targets, outputs)
        decoded = ctc_decode(outputs.detach().cpu().softmax(1).numpy()[0])
        t = self.model.decode_str(targets[0].cpu().numpy())
        s = self.model.decode_str(decoded)
        print(f"\n{t} : {s}")

    def show_ocr_results(self, index, inputs, targets, outputs):
        """Log the given inputs, targets, and outputs to the logger."""
        inputs = inputs.detach().cpu().numpy()[0]
        outputs = outputs.detach().softmax(1).cpu().numpy()[0]
        decoded = ctc_decode(outputs)
        decode_str = self.model.decode_str
        t = decode_str(targets[0].cpu().numpy())
        s = decode_str(decoded)
        figure = plt.figure(figsize=(10, 10))
        # log the OCR result for the first image in the batch
        plt.clf()
        plt.imshow(inputs[0], cmap=plt.cm.gray)
        plt.title(f"'{s}' @{index}", size=24)
        self.log_matplotlib_figure(figure, self.total)
        # plot the posterior probabilities for the first image in the batch
        plt.clf()
        for row in outputs:
            plt.plot(row)
        self.log_matplotlib_figure(figure, self.total, key="probs")
        plt.close(figure)

    def log_matplotlib_figure(self, fig, index, key="image"):
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
        exp = self.logger.experiment
        if hasattr(exp, "add_image"):
            exp.add_image(key, image, index)
        else:
            import wandb

            exp.log({key: [wandb.Image(image, caption=key)]})

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


default_config = """
data:
    train_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{00..21}.tar"
    train_bs: 12
    val_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{22..22}.tar"
    val_bs: 24
    nepoch: 20000
#logging:
#    wandb:
#        project: ocrorec2
#        log_model: all
checkpoint:
    every_n_epochs: 10
model:
    mname: text_model_210910
    lr: 3e-4
    halflife: 80
    display_freq: 1000
trainer:
    max_epochs: 10000
    gpus: 1
    progress_bar_refresh_rate: 2
    default_root_dir: ./_logs
"""

default_config = yaml.safe_load(StringIO(default_config))


def update_config(config, updates, path=None):
    path = path or []
    if isinstance(config, dict) and isinstance(updates, dict):
        for k, v in updates.items():
            if isinstance(config.get(k), dict):
                update_config(config.get(k), v, path=path+[k])
            else:
                config[k] = v
    else:
        raise ValueError(f"updates don't conform with config at {path}")


def scalar_convert(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s


def flatten_yaml(d, result=None, prefix=""):
    result = {} if result is None else result
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_yaml(v, result=result, prefix=prefix + k + ".")
        else:
            result[prefix + k] = v
    return result


def set_config(config, key, value):
    path = key.split(".")
    for k in path[:-1]:
        config = config.setdefault(k, {})
    config[path[-1]] = scalar_convert(value)


def parse_args(argv):
    config = dict(default_config)
    if len(argv) < 1:
        return config
    if argv[0].startswith("config="):
        _, fname = argv[0].split("=", 1)
        with open(fname, "r") as stream:
            updates = yaml.safe_load(stream)
        update_config(config, updates)
        argv = argv[1:]
    for arg in argv:
        assert "=" in arg, arg
        key, value = arg.split("=", 1)
        set_config(config, key, value)
    return config


def cmd_defaults(argv):
    yaml.dump(default_config, sys.stdout)


def cmd_train(argv):
    config = parse_args(argv)
    yaml.dump(config, sys.stdout)

    data = TextDataLoader(**config["data"])
    print("# checking training batch size", next(iter(data.train_dataloader()))[0].size())

    model = loading.load_or_construct_model(
        config["model"]["mname"],
        TextModel.charset_size(),
    )

    model = TextModel(model)
    _ = torch.jit.script(model)

    flattened_config = flatten_yaml(config)
    lmodel = TextLightning(model, config=flattened_config)

    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval="step"),
    )

    cpconfig = config["checkpoint"]
    mcheckpoint = ModelCheckpoint(**cpconfig)
    callbacks.append(mcheckpoint)

    tconfig = config["trainer"].copy()

    if "logging" in config and "wandb" in config["logging"]:
        print(f"# logging to {config['logging']['wandb']}")
        from pytorch_lightning.loggers import WandbLogger

        tconfig["logger"] = WandbLogger(**config["logging"]["wandb"])
    else:
        print(f"# logging locally")

    trainer = pl.Trainer(
        callbacks=callbacks,
        **tconfig,
    )
    print("mcheckpoint.dirpath", mcheckpoint.dirpath)
    trainer.fit(lmodel, data)


if __name__ == "__main__":
    eval(f"cmd_{sys.argv[1]}")(sys.argv[2:])
