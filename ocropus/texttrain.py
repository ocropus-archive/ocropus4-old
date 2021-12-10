"""Text recognition."""

import io, json, sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
from torch.autograd.grad_mode import F
import torch.jit
import yaml
from matplotlib import gridspec
from numpy import amax, arange, newaxis, tile
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy import ndimage as ndi
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import typer
from torchmore import layers

from . import confparse, jittable, textdata, textmodels


app = typer.Typer()


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
        charset: Optional[str] = None,
        display_freq: int = 1000,
        lr: float = 3e-4,
        lr_halflife: int = 1000,
        config: Dict[Any, Any] = {},
    ):
        super().__init__()
        self.display_freq = display_freq
        self.lr = lr
        self.lr_halflife = lr_halflife
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.total = 0
        self.hparams.config = json.dumps(config)
        self.save_hyperparameters(ignore="display_freq".split())
        self.model = textmodels.TextModel(mname, charset=charset)
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

@app.command()
def train(
    mname: str = "ocropus.textmodels.ctext_model_211124",
    charset: str = "ocropus.textmodels.charset_ascii",
    lr: float = 0.03,
    lr_halflife: int = 10,
    gpus: int = 1,
    default_root_dir: str = "./_logs",
    checkpoint: int = 1,
    display_freq: int = 1000,
    train_bucket: Optional[str] = None,
    train_shards: Optional[str] = None,
    val_shards: Optional[str] = None,
    max_epochs: int = 10000,
    train_bs: int = 16,
    val_bs: int = 16,
    augment: str = "distort",
    wandb: str = "",
    dumpjit: str = "",
    resume: Optional[str] = "",
):
    config = dict(locals())

    data = textdata.TextDataLoader(
        train_bucket=train_bucket,
        train_shards=train_shards,
        val_shards=val_shards,
        train_bs=train_bs,
        val_bs=val_bs,
        augment=augment,
    )
    print(
        "# checking training batch size", next(iter(data.train_dataloader()))[0].size()
    )

    lmodel = TextLightning(mname=mname, charset=charset)
    for k, v in config.items():
        setattr(lmodel.hparams, k, v)

    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval="step"),
    )

    mcheckpoint = ModelCheckpoint(
        every_n_epochs=checkpoint,
    )
    callbacks.append(mcheckpoint)

    if wandb != "":
        from pytorch_lightning.loggers import WandbLogger
        wconfig = eval(f"{wandb}")
        tconfig["logger"] = WandbLogger(**wconfig)
        print(f"# using wandb logger with config {wconfig}")
    else:
        print(f"# logging locally")

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=gpus,
        resume_from_checkpoint=resume,
        default_root_dir=default_root_dir,
    )

    if dumpjit != "":
        assert resume_from_checkpoint != "", "must specify checkpoint to dump"
        script = smodel.get_jit_model()
        torch.jit.save(script, config["dumpjit"])
        print(f"# saved model to {config['dumpjit']}")
        sys.exit(0)

    print("mcheckpoint.dirpath", mcheckpoint.dirpath)
    trainer.fit(lmodel, data)


if __name__ == "__main__":
    app()
