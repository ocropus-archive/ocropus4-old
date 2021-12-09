"""Text recognition."""

import io
import json
import random
import re
import sys
from functools import partial
from io import StringIO
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torch.jit
import webdataset as wds
import yaml
from matplotlib import gridspec
from numpy import amax, arange, newaxis, tile
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy import ndimage as ndi
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmore import layers

from . import confparse, degrade, jittable, linemodels, textmodels, utils

_ = linemodels


default_config = """
data:
    train_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{00..21}.tar"
    train_bs: 16
    val_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-0000{22..22}.tar"
    val_bs: 24
    nepoch: 20000
    num_workers: 8
    augment: distort
    normalize: simple
checkpoint:
    every_n_epochs: 10
lightning:
    mname: ctext_model_211124
    lr: 0.03
    lr_halflife: 10
    display_freq: 1000
    textmodel:
        charset: ascii
        config:
            noutput: 128
trainer:
    max_epochs: 10000
    gpus: 1
    default_root_dir: ./_logs
logging: {}
"""

default_config = yaml.safe_load(StringIO(default_config))


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


###
### Augmentations
###


@utils.useopt
def augment_none(image: torch.Tensor) -> torch.Tensor:
    """Perform no augmentation.

    Args:
        image (torch.Tensor): input image

    Returns:
        torch.Tensor: unaugmented output
    """
    return utils.as_torchimage(image)


@utils.useopt
def augment_transform(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using geometric transformations and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): input image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: augmented image
    """
    image = utils.as_npimage(image)
    if random.uniform(0, 1) < p:
        image = degrade.normalize(image)
        image = 1.0 * (image > 0.5)
    if image.shape[0] > 80.0:
        image = ndi.zoom(image, 80.0 / image.shape[0], order=1)
    if random.uniform(0, 1) < p:
        (image,) = degrade.transform_all(image, scale=(-0.3, 0))
    if random.uniform(0, 1) < p:
        image = degrade.noisify(image)
    image = utils.as_torchimage(image)
    return image


@utils.useopt
def augment_distort(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Augment image using distortions and noise.

    Also binarizes some images.

    Args:
        image (torch.Tensor): original image
        p (float, optional): probability of binarization. Defaults to 0.5.

    Returns:
        [type]: augmented image
    """
    image = utils.as_npimage(image)
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
    image = utils.as_torchimage(image)
    return image


###
### Text-Related
###


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


@utils.useopt
def normalize_none(s: str) -> str:
    """String normalization that only fixes quotes.

    Args:
        s (str): input string

    Returns:
        str: normalized string
    """
    s = fixquotes(s)
    return s


@utils.useopt
def normalize_simple(s: str) -> str:
    """Simple text normalization.

    - replaces unicode quotes with ascii ones
    - replaces multiple whitespace with singe space
    - strips space at beginning and end of string

    Args:
        s (str): [description]

    Returns:
        str: [description]
    """
    s = fixquotes(s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', "''", s)
    return s.strip()


@utils.useopt
def normalize_tex(s: str) -> str:
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


###
### Data Loading
###


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
        dl = wds.WebLoader(
            ds,
            collate_fn=collate4ocr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        ).slice(self.params.nepoch // batch_size)
        # Would like to shuffle here, but need to reorganize
        # batching logic to do so.
        # if mode == "train" and params.shuffle > 0:
        #     dl = dl.unbatched().shuffle(params.shuffle).batched(batch_size)
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


###
### Text Recognition Models
###


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
        self.save_hyperparameters(ignore="display_freq".split())
        self.model = textmodels.TextModel(mname, **textmodel)
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

    def training_step(self, batch: torch.Tensor, index: int) -> torch.Tensor:
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

    def validation_step(self, batch: torch.Tensor, index: int) -> torch.Tensor:
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

    def compute_loss(
        self, outputs: torch.Tensor, targets: List[torch.Tensor]
    ) -> torch.Tensor:
        assert len(targets) == len(outputs)
        targets, tlens = pack_for_ctc(targets)
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

    def compute_error(
        self, outputs: torch.Tensor, targets: List[torch.Tensor]
    ) -> float:
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

    def schedule(self, epoch: int):
        return 0.5 ** (epoch // self.lr_halflife)

    def log_results(
        self,
        index: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        self.show_ocr_results(index, inputs, targets, outputs)
        decoded = ctc_decode(outputs.detach().cpu().softmax(1).numpy()[0])
        t = self.model.decode_str(targets[0].cpu().numpy())
        s = self.model.decode_str(decoded)
        print(f"\n{t} : {s}")

    def show_ocr_results(
        self,
        index: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        """Log the given inputs, targets, and outputs to the logger."""
        inputs = inputs.detach().cpu().numpy()[0]
        outputs = outputs.detach().softmax(1).cpu().numpy()[0]
        decoded = ctc_decode(outputs)
        decode_str = self.model.decode_str
        decode_str(targets[0].cpu().numpy())
        s = decode_str(decoded)
        # log the OCR result for the first image in the batch
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(inputs[0], cmap=plt.cm.gray, interpolation="nearest")
        ax.set_title(f"'{s}' @{self.total}", size=24)
        ax = fig.add_subplot(gs[0, 1])
        for row in outputs:
            ax.plot(row)
        self.log_matplotlib_figure(fig, self.total, key="output", size=(1200, 600))
        plt.close(fig)

    def log_matplotlib_figure(self, fig, index, key="image", size=(600, 600)):
        """Log the given matplotlib figure to tensorboard logger tb."""
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = image.convert("RGB")
        image = image.resize(size)
        image = np.array(image)
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        exp = self.logger.experiment
        if hasattr(exp, "add_image"):
            exp.add_image(key, image, index)
        else:
            import wandb

            exp.log({key: [wandb.Image(image, caption=key)]})

    def probs_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def predict_batch(self, inputs: torch.Tensor, **kw) -> List[str]:
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result


###
##3 Top-Level Commands
###


def train(argv: List[str]):
    argv = argv or []
    config = confparse.parse_args(argv, default_config)
    yaml.dump(config, sys.stdout)

    dataconfig = config["data"]
    data = TextDataLoader(**dataconfig)
    print(
        "# checking training batch size", next(iter(data.train_dataloader()))[0].size()
    )

    lmodel = TextLightning(**config["lightning"])
    lmodel.hparams.train_bs = dataconfig["train_bs"]
    lmodel.hparams.augment = dataconfig["augment"]
    lmodel.hparams.normalize = dataconfig["normalize"]

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

    if config.get("dumpjit"):
        assert (
            "resume_from_checkpoint" in config["trainer"]
        ), "must resume from checkpoint to dump JIT script"
        script = smodel.get_jit_model()
        torch.jit.save(script, config["dumpjit"])
        print(f"# saved model to {config['dumpjit']}")
        sys.exit(0)

    print("mcheckpoint.dirpath", mcheckpoint.dirpath)
    trainer.fit(lmodel, data)


if __name__ == "__main__":
    train(sys.argv[1:])
