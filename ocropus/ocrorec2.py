"""Text recognition."""

import io
import json
import os
import random
import re
import sys
from typing import List, Optional, Dict, Any, Tuple, Union
from functools import partial
from io import StringIO
from itertools import islice

import yaml

import typer
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
from . import confparse
import torch.jit

_ = linemodels


app = typer.Typer()


min_w, min_h, max_w, max_h = 15, 15, 4000, 200


def identity(x: Any) -> Any:
    """Identity function."""
    return x


def goodsize(sample: Dict[Any, Any], max_w: int = max_w, max_h: int = max_h) -> bool:
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


def ctc_decode(
    probs: torch.Tensor, sigma: float = 1.0, threshold: float = 0.7, full: bool = False
) -> Union[List[int], Tuple[List[int], List[float]]]:
    """Perform simple CTC decoding of the probability outputs from an LSTM or similar model.

    Args:
        probs (torch.Tensor): probabilities in BDL format.
        sigma (float, optional): smoothing. Defaults to 1.0.
        threshold (float, optional): thresholding of output probabilities. Defaults to 0.7.
        full (bool, optional): return both classes and probabilities. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], List[float]]]: [description]
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


def pack_for_ctc(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a list of tensors into tensors in the format required by CTC.

    Args:
        seqs (List[torch.Tensor]): list of tensors (integer sequences)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: packed tensor
    """
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def collate4ocr(
    samples: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[str]]:
    """Collate OCR images and text into tensor/list batch.

    Args:
        samples (List[Tuple[Tensor, str]]): [description]

    Returns:
        Tuple[Tensor, List[str]]: batch of images and text
    """
    images, seqs = zip(*samples)
    images = jittable.stack_images(images)
    return images, seqs


def as_npimage(a: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a numpy image .

    Args:
        a (Union[torch.Tensor, np.ndarray]): some image

    Returns:
        np.ndarray: rank 3 floating point image
    """
    assert a.ndim == 3
    if isinstance(a, torch.Tensor):
        assert int(a.shape[0]) in [1, 3]
        a = a.detach().cpu().permute(1, 2, 0).numpy()
    assert isinstance(a, np.ndarray)
    assert a.shape[2] in [1, 3]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return a


def as_torchimage(a: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert a numpy array into a torch . Image .

    Args:
        a (Union[torch.Tensor, np.ndarray]): some image in numpy or torch format

    Returns:
        torch.Tensor: floating point tensor representing an image
    """
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
def augment_none(image: torch.Tensor) -> torch.Tensor:
    """Perform no augmentation.

    Args:
        image (torch.Tensor): input image

    Returns:
        torch.Tensor: unaugmented output
    """
    return as_torchimage(image)


@useopt
def augment_transform(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using geometric transformations and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): input image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: augmented image
    """
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
def augment_distort(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using distortions and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): original image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        [type]: augmented image
    """
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


def fixquotes(s: str) -> str:
    """Replace unicode quotes with ascii ones.

    Args:
        s (str): unicode string

    Returns:
        str: unicode string with ascii quotes
    """
    s = re.sub("[\u201c\u201d]", '"', s)
    s = re.sub("[\u2018\u2019]", "'", s)
    s = re.sub("[\u2014]", "-", s)
    return s


@useopt
def normalize_none(s: str) -> str:
    """String normalization that only fixes quotes.

    Args:
        s (str): input string

    Returns:
        str: normalized string
    """
    s = fixquotes(s)
    return s


@useopt
def normalize_simple(s: str) -> str:
    """Simple text normalization.

    - replaces tex commands (\alpha, \beta, etc) with "~"
    - eliminates "{", "}", "_" and "^" from input (TeX sequences)
    - replaces unicode quotes with ascii ones
    - replaces multiple whitespace with singe space
    - strips space at beginning and end of string

    Args:
        s (str): [description]

    Returns:
        str: [description]
    """
    s = fixquotes(s)
    s = re.sub("\\\\[A-Za-z]+", "~", s)
    s = re.sub("\\\\[_^]+", "", s)
    s = re.sub("[{}]", "", s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', "''", s)
    return s.strip()


def good_text(regex: str, sample: str) -> bool:
    """Check if a string matches a regular expression."""
    image, txt = sample
    return re.search(regex, txt)


class TextDataLoader(pl.LightningDataModule):
    """Lightning Data Module for OCR training."""

    def __init__(
        self,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        train_bs: int = 4,
        val_bs: int = 20,
        text_select_re: str = "[A-Za-z0-9]",
        nepoch: int = 5000,
        num_workers: int = 4,
        cache_size: int = -1,
        cache_dir: str = None,
        shuffle: int = 5000,
        augment: str = "distort",
        text_normalizer: str = "simple",
        extensions: str = "line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt",
        max_w: int = 1000,
        max_h: int = 200,
        **kw,
    ):
        """Initialize the TextDataLoader

        Args:
            train_shards (Optional[Union[str, List[str], Dict[Any, Any]]], optional): list of shards to train on. Defaults to None.
            val_shards (Optional[Union[str, List[str]]], optional): list of shards to validate on. Defaults to None.
            train_bs (int, optional): batch size for training. Defaults to 4.
            val_bs (int, optional): batch size for validation. Defaults to 20.
            text_select_re (str, optional): regular expression that selects training samples. Defaults to "[A-Za-z0-9]".
            nepoch (int, optional): number of samples per epoch. Defaults to 5000.
            num_workers (int, optional): number of workers per loader. Defaults to 4.
            cache_size (int, optional): cache size for downloading shards. Defaults to -1.
            cache_dir (str, optional): directory where shards are cached. Defaults to None.
            shuffle (int, optional): size of inline shuffle buffer. Defaults to 5000.
            augment (str, optional): choice of sample augmentation. Defaults to "distort".
            text_normalizer (str, optional): choice of text normalization. Defaults to "simple".
            extensions (str, optional): choice of file name extensions. Defaults to "line.png;line.jpg;word.png;word.jpg;jpg;jpeg;ppm;png txt;gt.txt".
            max_w (int, optional): maximum image width (larger=ignored). Defaults to 1000.
            max_h (int, optional): maximum image height (larger=ignored). Defaults to 200.
        """
        super().__init__()
        print(locals())
        assert train_shards is not None
        if val_shards == "":
            val_shards = None
        self.params = confparse.Params(locals())
        self.cache_size = cache_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers

    def make_loader(
        self,
        fname: Union[str, List[str], Dict[Any, Any]],
        batch_size: int,
        mode: str = "train",
        augment: str = "distort",
    ) -> DataLoader:
        """Make a data loader for a given collection of shards.

        Args:
            fname (Union[str, List[str], Dict[Any, Any]]): shard spec, shard list, or dataset dict spec
            batch_size (int): desired batch size
            mode (str, optional): mode (val or train). Defaults to "train".
            augment (str, optional): augmentation function to use. Defaults to "distort".

        Returns:
            DataLoader: data loader
        """
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

    def train_dataloader(self) -> DataLoader:
        """Make a data loader for training.

        Returns:
            DataLoader: data loader
        """
        return self.make_loader(
            self.params["train_shards"],
            self.params["train_bs"],
            mode="train",
        )

    def val_dataloader(self) -> DataLoader:
        """Make a data loader for validation.

        Returns:
            DataLoader: data loader
        """
        if self.params.get("val_shards") is None:
            return None
        return self.make_loader(
            self.params["val_shards"],
            self.params["val_bs"],
            mode="val",
            augment="",
        )


class TextModel(nn.Module):
    """Word-level text model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.jit.export
    def forward(self, images):
        b, c, h, w = images.shape
        assert c == torch.tensor(1) or c == torch.tensor(3)
        if c == torch.tensor(1):
            images = images.repeat(1, 3, 1, 1)
            b, c, h, w = images.shape
        assert h > 15 and h < 4000 and w > 15 and h < 4000
        for image in images:
            image -= image.min()  # was: amin()
            image /= torch.max(image.amax(), torch.tensor([0.01], device=image.device))
            if image.mean() > 0.5:
                image[:, :, :] = 1.0 - image[:, :, :]
        return self.model.forward(images)

    # the following methods are here so that they can be exported to torchscript

    @torch.jit.export
    def standardize(self, im: torch.Tensor) -> torch.Tensor:
        """Standardize the intensity levels in an image."""
        return jittable.standardize_image(im)

    @torch.jit.export
    def auto_resize(self, im: torch.Tensor) -> torch.Tensor:
        """Auto-resize an image to a standard size."""
        return jittable.auto_resize(im)

    @torch.jit.export
    def make_batch(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Make a batch of images by resizing them and centering their contents."""
        batch = jittable.stack_images(images)
        return batch

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        """Encode a string as a tensor."""
        result = torch.tensor([ord(c) for c in s], dtype=torch.long)
        result[result >= self.charset_size()] = 127
        return result

    @torch.jit.export
    def decode_str(self, a: torch.Tensor) -> str:
        """Decode a tensor as a string."""
        return "".join([chr(int(c)) for c in a])

    @torch.jit.export
    @staticmethod
    def charset_size():
        return 128


class TextLightning(pl.LightningModule):
    """A class encapsulating the logic for training text line recognizers.

    Models consist of three nested parts:
    - TextLightning wrapper -- the Lightning interface
        - TextModel -- a wrapper whose forward method takes care of normalization
            - basemodel -- the actual deep learning model
    """

    def __init__(
        self,
        mname: Optional[str] = None,
        display_freq: int = 1000,
        lr: float = 3e-4,
        lr_halflife: int = 1000,
        config: Dict[Any, Any] = {},
        textmodel: Dict[Any, Any] = {},
        basemodel: Dict[Any, Any] = {},
    ):
        super().__init__()
        self.display_freq = display_freq
        self.lr = lr
        self.lr_halflife = lr_halflife
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.total = 0
        self.hparams.config = json.dumps(config)
        self.save_hyperparameters()
        if mname is not None:
            basemodel = loading.load_or_construct_model(mname, **basemodel)
            self.model = TextModel(basemodel, **textmodel)
            # make sure the model is JIT-able
            self.get_jit_model()
            print("model created and is JIT-able")

    def get_jit_model(self):
        """Get the JIT-able version of the model."""
        script = torch.jit.script(self.model)
        # torch.onnx.export(script, (torch.rand(1, 3, 48, 200),), "/dev/null", opset_version=11)
        return script

    def forward(self, inputs):
        """Perform the forward step. This just calls the forward method of the model."""
        return self.model.forward(inputs)

    def training_step(self, batch:torch.Tensor, index:int) -> torch.Tensor:
        """Perform a training step."""
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

    def validation_step(self, batch: torch.Tensor, index:int) -> torch.Tensor:
        """Perform a validation step."""
        inputs, text_targets = batch
        outputs = self.forward(inputs)
        targets = [self.model.encode_str(s) for s in text_targets]
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        self.log("val_loss", loss)
        err = self.compute_error(outputs, targets)
        self.log("val_err", err)
        return loss

    def compute_loss(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        assert len(targets) == len(outputs)
        targets, tlens = pack_for_ctc(targets)
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

    def compute_error(self, outputs:torch.Tensor, targets: List[torch.Tensor]) -> float:
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

    def schedule(self, epoch:int):
        return 0.5 ** (epoch // self.lr_halflife)

    def log_results(self, index:int, inputs:torch.Tensor, targets:torch.Tensor, outputs:torch.Tensor) -> None:
        self.show_ocr_results(index, inputs, targets, outputs)
        decoded = ctc_decode(outputs.detach().cpu().softmax(1).numpy()[0])
        t = self.model.decode_str(targets[0].cpu().numpy())
        s = self.model.decode_str(decoded)
        print(f"\n{t} : {s}")

    def show_ocr_results(self, index:int, inputs:torch.Tensor, targets:torch.Tensor, outputs:torch.Tensor) -> None:
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

    def probs_batch(self, inputs:torch.Tensor) -> torch.Tensor:
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def predict_batch(self, inputs:torch.Tensor, **kw) -> List[str]:
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
lightning:
    mname: ctext_model_211124
    lr: 0.03
    lr_halflife: 2
    display_freq: 1000
trainer:
    max_epochs: 10000
    gpus: 1
    default_root_dir: ./_logs
"""

default_config = yaml.safe_load(StringIO(default_config))


@app.command()
def defaults():
    yaml.dump(default_config, sys.stdout)


@app.command()
def dumpjit(src: str, dst: str):
    print(f"loading {src}")
    ckpt = torch.load(open(src, "rb"))
    model = ckpt["hyper_parameters"]["model"]
    model.cpu()
    script = torch.jit.script(model)
    print(f"dumping {dest}")
    assert not os.path.exists(dest)
    torch.jit.save(script, dest)


@app.command()
def train(argv: Optional[List[str]] = typer.Argument(None)):
    argv = argv or []
    config = confparse.parse_args(argv, default_config)
    yaml.dump(config, sys.stdout)

    data = TextDataLoader(**config["data"])
    print(
        "# checking training batch size", next(iter(data.train_dataloader()))[0].size()
    )

    lmodel = TextLightning(**config["lightning"])

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
    app()
