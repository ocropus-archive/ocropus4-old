"""Text recognition."""

import io, json, sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import re
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
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import typer
from torchmore import layers
from pytorch_lightning.utilities import rank_zero_only
import wandb
from pytorch_lightning.loggers import WandbLogger


from . import confparse, jittable, textdata, textmodels


app = typer.Typer()

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
        mopts: Dict[str, Any] = None,
        charset: Optional[str] = None,
        display_freq: int = 1000,
        lr: float = 3e-4,
        lr_scale: float = 1e-3,
        lr_steps: int = 1000,
        lr_halflife: int = 10,
    ):
        super().__init__()
        self.display_freq = display_freq
        self.lr = lr  # FIXME
        self.lr_halflife = lr_halflife  # FIXME
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.total = 0
        for k, v in mopts.items():
            setattr(self.hparams, k, v)
        self.save_hyperparameters()
        self.model = textmodels.TextModel(mname, charset=charset, config=mopts)
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
        # import os; print("DEBUG "*10); os.system("sleep 3600")
        self.log("train_loss", loss)
        err = self.compute_error(outputs, targets)
        self.log("train_err", err, prog_bar=True)
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, logger=False)
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

    def compute_loss(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(outputs, targets)
        else:
            return self.ctc_loss(outputs, targets[0], targets[1])

    def compute_error(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "compute_error"):
            return self.model.compute_error(outputs, targets)
        else:
            return self.ctc_error(outputs, targets)

    def ctc_loss(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        assert len(targets) == len(outputs)
        targets, tlens = textmodels.pack_for_ctc(targets)
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

    def ctc_error(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> float:
        probs = outputs.detach().cpu().softmax(1)
        targets = [[int(x) for x in t] for t in targets]
        total = sum(len(t) for t in targets)
        predicted = [textmodels.ctc_decode(p) for p in probs]
        errs = [editdistance.distance(p, t) for p, t in zip(predicted, targets)]
        return sum(errs) / float(total)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        scheduler = LambdaLR(optimizer, self.schedule)
        # scale, steps = self.hparams.lr_scale, self.hparams.lr_steps
        # scheduler2 = ExponentialLR(optimizer, gamma=scale ** (1.0 / steps), last_epoch=steps)
        return [optimizer], [scheduler]

    def schedule(self, epoch: int):
        gamma = self.hparams.lr_scale ** (1.0 / self.hparams.lr_steps)
        return gamma ** min(epoch, self.hparams.lr_steps)

    def log_results(
        self,
        index: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        self.show_ocr_results(index, inputs, targets, outputs)
        decoded = textmodels.ctc_decode(outputs.detach().cpu().softmax(1).numpy()[0])
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
        decoded = textmodels.ctc_decode(outputs)
        decode_str = self.model.decode_str
        decode_str(targets[0].cpu().numpy())
        s = decode_str(decoded)
        # log the OCR result for the first image in the batch
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(inputs[0], cmap=plt.cm.gray, interpolation="nearest")
        s_safe = re.sub(r"[}{$\\]", "?", s)
        ax.set_title(f"'{s_safe}' @{self.total}", size=24)
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
            print("*** tensorflow image log")
            exp.add_image(key, image, index)
        else:
            print("*** WANDB IMAGE LOG")
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
        result = [textmodels.ctc_decode(p, **kw) for p in probs]
        return result


###
# Top-Level Commands
###


@app.command()
def train(
    augment: str = "distort",
    charset: str = "ocropus.textmodels.charset_ascii",
    checkpoint: int = 1,
    default_root_dir: str = "./_logs",
    display_freq: int = 20,
    dumpjit: str = "",
    gpus: str = "1",
    lr: float = 1e-3,
    max_epochs: int = 10000,
    mname: str = "ocropus.textmodels.text_model_220204",
    mopts: str = "",
    datamode: str = "default",
    train_bs: int = 12,
    train_bsm: int = 6,
    nepoch: int = 200000,
    resume: Optional[str] = None,
    restart: Optional[str] = None,
    train_shards: Optional[str] = None,
    val_bs: int = 32,
    val_shards: Optional[str] = None,
    wandb: str = "",
    lr_scale: float = 1e-2,
    lr_steps: int = 300,
    lr_halflife: int = 10,  # FIXME
    num_workers: int = 8,
):
    if wandb != "":
        import wandb as wandb_lib
        wandb_lib.require("service")
        wandb_lib.setup()
    config = dict(locals())

    if dumpjit != "":
        assert resume is not None, "dumpjit requires a checkpoint"
        print(f"# loading {resume}")
        ckpt = torch.load(open(resume, "rb"), map_location="cpu")
        print(ckpt["hparams_name"])
        print(ckpt["hyper_parameters"])
        lmodel = TextLightning(**ckpt["hyper_parameters"])
        print("# setting state dict")
        lmodel.cpu()
        lmodel.load_state_dict(ckpt["state_dict"])
        print(lmodel)
        print("# compiling jit model")
        script = lmodel.get_jit_model()
        print(f"# saving {dumpjit}")
        torch.jit.save(script, dumpjit)
        print(f"# saved model to {dumpjit}")
        sys.exit(0)

    ngpus = len(gpus.split(","))
    assert ngpus >= 1
    bs = train_bs + (ngpus - 1) * train_bsm
    print(f"ngpus {ngpus} batch_size/multiplier {train_bs}/{train_bsm} actual {bs}")

    data = textdata.make_dataloader(
        augment=augment,
        nepoch=nepoch,
        train_bs=bs,
        train_shards=train_shards,
        val_bs=val_bs,
        val_shards=val_shards,
        num_workers=num_workers,
        datamode=datamode,
    )

    if restart is not None:
        ckpt = torch.load(open(restart, "rb"), map_location="cpu")
        mname = ckpt["hyper_parameters"]["mname"]
        mopts = ckpt["hyper_parameters"]["mopts"]

    mopts = eval(f"dict({mopts})")
    lmodel = TextLightning(
        mname=mname,
        mopts=mopts,
        charset=charset,
        lr=lr,
        lr_steps=lr_steps,
        lr_scale=lr_scale,
        lr_halflife=lr_halflife,
    )

    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval="step"),
    )

    mcheckpoint = ModelCheckpoint(
        every_n_epochs=checkpoint,
    )
    callbacks.append(mcheckpoint)

    kw = {}
    rank = rank_zero_only.rank
    if wandb != "" and rank == 0:
        wconfig = dict(project=wandb, log_model="all")
        kw["logger"] = WandbLogger(**wconfig)
        print(f"# using wandb logger with config {wconfig} at {rank}")
    else:
        print("# logging locally")

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=gpus,
        resume_from_checkpoint=resume,
        default_root_dir=default_root_dir,
        **kw,
    )

    if restart is not None:
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        lmodel.load_state_dict(state)

    print("mcheckpoint.dirpath", mcheckpoint.dirpath)
    trainer.fit(lmodel, data)


if __name__ == "__main__":
    app()
