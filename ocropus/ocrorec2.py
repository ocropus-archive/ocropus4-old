"""Text recognition."""

import io
import json
import os
import random
import re
import sys
from typing import List
from functools import partial
from io import StringIO
from itertools import islice

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

from . import degrade, lineest, linemodels, loading, slog, utils, jittable
from .utils import useopt
import torch.jit

_ = linemodels


min_w, min_h, max_w, max_h = 15, 15, 4000, 200


def identity(x):
    return x


class Params:
    def __init__(self, d):
        assert isinstance(d, dict)
        self.__the_dict__ = d

    def get(self, *args):
        return self.__the_dict__.get(*args)

    def __getitem__(self, name):
        return self.__the_dict__[name]

    def __setitem__(self, name, value):
        self.__the_dict__[name] = value

    def __getattr__(self, name):
        value = self.__the_dict__[name]
        if isinstance(value, dict):
            return Params(value)
        else:
            return value

    def __setattr__(self, name, value):
        if name[0] == "_":
            object.__setattr__(self, name, value)
        else:
            self.__the_dict__[name] = value

    def __getstate__(self):
        return self.__the_dict__

    def __setstate__(self, state):
        self.__the_dict__ = state


def goodsize(sample, max_w=max_w, max_h=max_h):
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
    images, seqs = zip(*samples)
    images = jittable.stack_images(images)
    return images, seqs


def collate4ocr_old(samples):
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


def as_npimage(a):
    assert a.ndim == 3
    if isinstance(a, torch.Tensor):
        assert int(a.shape[0]) in [1, 3]
        a = a.detach().cpu().permute(1, 2, 0).numpy()
    assert isinstance(a, np.ndarray)
    assert a.shape[2] in [1, 3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return a


def as_torchimage(a):
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            a = np.stack((a,) * 3, axis=-1)
        assert int(a.shape[2]) in [1, 3]
        a = torch.tensor(a.transpose(2, 0, 1))
    assert a.ndim == 3
    assert isinstance(a, torch.Tensor)
    assert a.shape[0] in [1, 3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return a


@useopt
def augment_none(image):
    return as_torchimage(image)


@useopt
def augment_transform(image, p=0.5):
    image = as_npimage(image)
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    image = as_torchimage(image)
    return image


@useopt
def augment_distort(image, p=0.5):
    image = as_npimage(image)
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
    image = as_torchimage(image)
    return image


def fixquotes(s):
    s = re.sub("[\u201c\u201d]", '"', s)
    s = re.sub("[\u2018\u2019]", "'", s)
    s = re.sub("[\u2014]", "-", s)
    return s


@useopt
def normalize_none(s):
    s = fixquotes(s)
    return s


@useopt
def normalize_simple(s):
    s = fixquotes(s)
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
        train_shards=None,
        val_shards=None,
        train_bs: int = 4,
        val_bs: int = 20,
        text_select_re: str = "[A-Za-z0-9]",
        nepoch: int = 5000,
        num_workers: int = 4,
        cache_size: int = -1,
        cache_dir: str = None,
        batch_size=5,
        shuffle=5000,
        augment="distort",
        text_normalizer="simple",
        extensions="line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
        cache=True,
        max_w=1000,
        max_h=200,
        **kw,
    ):
        super().__init__()
        print(locals())
        assert train_shards is not None
        if val_shards == "":
            val_shards = None
        self.params = Params(locals())
        self.cache_size = cache_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers

    def make_loader(
        self,
        fname,
        batch_size,
        mode="train",
        augment="distort",
    ):
        params = self.params
        ds = wds.WebDataset(
            fname,
            cache_size=float(self.cache_size),
            cache_dir=self.cache_dir,
            verbose=True,
            shardshuffle=50,
            resampled=True,
        )
        if mode == "train" and params.shuffle > 0:
            ds = ds.shuffle(params.shuffle)
        ds = ds.decode("torchrgb8").to_tuple(params.extensions)
        text_normalizer = eval(f"normalize_{params.text_normalizer}")
        ds = ds.map_tuple(identity, text_normalizer)
        if params.text_select_re != "":
            ds = ds.select(partial(good_text, params.text_select_re))
        if augment != "":
            f = eval(f"augment_{augment}")
            ds = ds.map_tuple(f, identity)
        ds = ds.map_tuple(jittable.standardize_image, identity)
        ds = ds.select(goodsize)
        ds = ds.map_tuple(jittable.auto_resize, identity)
        ds = ds.select(partial(goodsize, max_w=params.max_w, max_h=params.max_h))
        if params.nepoch > 0:
            ds = ds.with_epoch(params.nepoch)
        dl = DataLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        )
        return dl

    def train_dataloader(self):
        return self.make_loader(
            self.params["train_shards"],
            self.params["train_bs"],
            mode="train",
        )

    def val_dataloader(self):
        if self.params.get("val_shards") is None:
            return None
        return self.make_loader(
            self.params["val_shards"],
            self.params["val_bs"],
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
        assert c == torch.tensor(1) or c == torch.tensor(3)
        if c == torch.tensor(1):
            images = images.repeat(1, 3, 1, 1)
            b, c, h, w = images.shape
        assert h > 15 and h < 4000 and w > 15 and h < 4000
        for image in images:
            image -= image.min() # was: amin()
            image /= torch.max(image.amax(), torch.tensor([0.01], device=image.device))
            if image.mean() > 0.5:
                image[:, :, :] = 1.0 - image[:, :, :]
        return self.model.forward(images)

    @torch.jit.export
    def standardize(self, im: torch.Tensor) -> torch.Tensor:
        return jittable.standardize_image(im)

    @torch.jit.export
    def auto_resize(self, im: torch.Tensor) -> torch.Tensor:
        return jittable.auto_resize(im)

    @torch.jit.export
    def make_batch(self, images: List[torch.Tensor]) -> torch.Tensor:
        batch = jittable.stack_images(images)
        return batch



class TextLightning(pl.LightningModule):
    """A class encapsulating the logic for training text line recognizers."""

    def __init__(
        self,
        model,
        *,
        display_freq=1000,
        lr=3e-4,
        lr_halflife=1000,
        config="{}",
    ):
        super().__init__()
        self.display_freq = display_freq
        self.model = model
        self.lr = lr
        self.lr_halflife = lr_halflife
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.total = 0
        self.hparams.config = json.dumps(config)
        self.save_hyperparameters()

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
    train_bs: 8
    val_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{22..22}.tar"
    val_bs: 24
    nepoch: 20000
    num_workers: 8
checkpoint:
    every_n_epochs: 10
model:
    mname: ctext_model_211124
    lr: 0.03
    halflife: 2
    display_freq: 1000
trainer:
    max_epochs: 10000
    gpus: 1
    default_root_dir: ./_logs
"""

default_config = yaml.safe_load(StringIO(default_config))


def update_config(config, updates, path=None):
    path = path or []
    if isinstance(config, dict) and isinstance(updates, dict):
        for k, v in updates.items():
            if isinstance(config.get(k), dict):
                update_config(config.get(k), v, path=path + [k])
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


def cmd_dumpjit(argv):
    assert len(argv) == 2, argv
    src, dest = argv
    print(f"loading {src}")
    ckpt = torch.load(open(src, "rb"))
    model = ckpt["hyper_parameters"]["model"]
    model.cpu()
    script = torch.jit.script(model)
    print(f"dumping {dest}")
    assert not os.path.exists(dest)
    torch.jit.save(script, dest)


def cmd_dumponnx(argv):
    assert len(argv) == 2, argv
    src, dest = argv
    print(f"loading {src}")
    ckpt = torch.load(open(src, "rb"))
    model = ckpt["hyper_parameters"]["model"]
    model.cpu()
    script = torch.jit.script(model)
    print(f"dumping {dest}")
    assert not os.path.exists(dest)
    torch.onnx.export(script, (torch.rand(1, 3, 48, 200),), dest, opset_version=11)


def cmd_train(argv):
    config = parse_args(argv)
    yaml.dump(config, sys.stdout)

    data = TextDataLoader(**config["data"])
    print(
        "# checking training batch size", next(iter(data.train_dataloader()))[0].size()
    )

    model = loading.load_or_construct_model(
        config["model"]["mname"],
        TextModel.charset_size(),
    )

    model = TextModel(model)

    # make sure the model is actually convertible to JIT and ONNX
    script = torch.jit.script(model)
    #torch.onnx.export(script, (torch.rand(1, 3, 48, 200),), "/dev/null", opset_version=11)
    print("# model is JIT-able")

    lmodel = TextLightning(model, config=json.dumps(config))

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
