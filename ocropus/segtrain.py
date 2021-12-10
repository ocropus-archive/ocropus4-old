import io, json, sys
from io import StringIO
from typing import Any, Dict, List, Optional
from click.core import Option

import numpy as np
import PIL
import PIL.Image
import pytorch_lightning as pl
import torch
from torch.nn.modules.module import register_module_backward_hook
import yaml
import typer
from matplotlib import gridspec
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy import ndimage as ndi
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from . import confparse, segmodels, segdata

app = typer.Typer()

default_config = """
data:
    train_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-wseg-sub/uw3-wseg-{000000..000116}.tar"
    train_bs: 2
    val_shards: "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-wseg-sub/uw3-wseg-{000117..000117}.tar"
    val_bs: 2
    augmentation: default
    num_workers: 8
    nepoch: 200000
checkpoint:
    every_n_epochs: 1
lightning:
    mname: ocropus.segmodels.segmentation_model_210910
    lr: 0.01
    lr_halflife: 5
    display_freq: 100
trainer:
    max_epochs: 10000
    gpus: 1
    default_root_dir: ./_logs
logging: {}
"""

default_config = yaml.safe_load(StringIO(default_config))


###
# Loading and Preprocessing
###


class SegLightning(pl.LightningModule):
    def __init__(
        self,
        *,
        mname="seg",
        margin=16,
        weightmask=0,
        bordermask=16,
        lr=0.01,
        lr_halflife=10,
        display_freq=100,
        segmodel: Dict[Any, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = segmodels.SegModel(mname, **segmodel)
        self.get_jit_model()
        print("model created and is JIT-able")

    def get_jit_model(self):
        script = torch.jit.script(self.model)
        return script

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        scheduler = LambdaLR(optimizer, self.schedule)
        return [optimizer], [scheduler]

    def schedule(self, epoch: int):
        return 0.5 ** (epoch // self.hparams.lr_halflife)

    def make_weight_mask(self, targets, w, d):
        mask = targets.detach().cpu().numpy()
        assert mask.ndim == 3
        mask = (mask >= 0.5).astype(float)
        if w > 0:
            mask = ndi.maximum_filter(mask, (0, w, w), mode="constant")
        if d > 0:
            mask[:, :d, :] = 0
            mask[:, -d:, :] = 0
            mask[:, :, :d] = 0
            mask[:, :, -d:] = 0
        mask = torch.tensor(mask, device=targets.device)
        return mask

    def compute_loss(self, outputs, targets, mask=None):
        """Compute loss taking a margin into account."""
        b, d, h, w = outputs.shape
        b1, h1, w1 = targets.shape
        assert h <= h1 and w <= w1 and h1 - h < 5 and w1 - w < 5, (
            outputs.shape,
            targets.shape,
        )
        targets = targets[:, :h, :w]
        # lsm = outputs.log_softmax(1)
        if self.hparams.margin > 0:
            m = self.hparams.margin
            outputs = outputs[:, :, m:-m, m:-m]
            targets = targets[:, m:-m, m:-m]
            if mask is not None:
                mask = mask[:, m:-m, m:-m]
        if mask is None:
            loss = nn.CrossEntropyLoss()(outputs, targets.to(outputs.device))
        else:
            loss = nn.CrossEntropyLoss(reduction="none")(
                outputs, targets.to(outputs.device)
            )
            loss = torch.sum(loss * mask.to(loss.device)) / (0.1 + mask.sum())
        return loss

    def training_step(self, batch, index, mode="train"):
        inputs, targets = batch
        assert inputs.ndim == 4, (inputs.shape, outputs.shape, targets.shape)
        assert inputs.shape[1] == 3, inputs.shape
        outputs = self.model.forward(inputs)
        assert outputs.ndim == 4, (inputs.shape, outputs.shape, targets.shape)
        assert targets.ndim == 3, (inputs.shape, outputs.shape, targets.shape)
        assert outputs.shape[0] < 100 and outputs.shape[1] < 10, outputs.shape
        if outputs.shape != inputs.shape:
            assert outputs.shape[0] == inputs.shape[0]
            assert outputs.ndim == 4
            bs, h, w = targets.shape
            outputs = outputs[:, :, :h, :w]
        if self.hparams.weightmask >= 0 or self.hparams.bordermask >= 0:
            mask = self.make_weight_mask(
                targets, self.hparams.weightmask, self.hparams.bordermask
            )
            self.last_mask = mask
        else:
            mask = None
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets, mask=mask)
        self.log(f"{mode}_loss", loss)
        with torch.no_grad():
            pred = outputs.softmax(1).argmax(1)
            errors = (pred != targets).sum()
            err = float(errors) / float(targets.nelement())
            self.log(f"{mode}_err", err, prog_bar=True)
        if mode == "train" and index % self.hparams.display_freq == 0:
            self.display_result(index, inputs, targets, outputs, mask)
        return loss

    def validation_step(self, batch, index):
        return self.training_step(batch, index, mode="val")

    def display_result(self, index, inputs, targets, outputs, mask):
        import matplotlib.pyplot as plt

        cmap = plt.cm.nipy_spectral

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2)
        fig_img, fig_out, fig_slice, fig_gt = [
            fig.add_subplot(gs[k // 2, k % 2]) for k in range(4)
        ]
        fig_img.set_title(f"{index}")
        doc = inputs[0, 0].detach().cpu().numpy()
        mask = getattr(self, "last_mask")
        if mask is not None:
            mask = mask[0].cpu().detach().numpy()
            combined = np.array([doc, doc, mask]).transpose(1, 2, 0)
            fig_img.imshow(combined)
        else:
            fig_img.imshow(doc, cmap="gray")
        p = outputs.detach().cpu().softmax(1)
        assert not torch.isnan(inputs).any()
        assert not torch.isnan(outputs).any()
        b, d, h, w = outputs.size()
        result = p.numpy()[0].transpose(1, 2, 0)
        if result.shape[2] > 3:
            result = result[..., 1:4]
        else:
            result = result[..., :3]
        fig_out.imshow(result, vmin=0, vmax=1)
        # m = result.shape[1] // 2
        m = min(max(10, result[:, :, 2:].sum(2).sum(0).argmax()), result.shape[1] - 10)
        fig_out.set_title(f"x={m}")
        fig_out.plot([m, m], [0, h], color="white", alpha=0.5)
        colors = [cmap(x) for x in np.linspace(0, 1, p.shape[1])]
        for i in range(0, d):
            fig_slice.plot(p[0, i, :, m], color=colors[i % len(colors)])
        if p.shape[1] <= 4:
            t = targets[0].detach().cpu().numpy()
            t = np.array([t == 1, t == 2, t == 3]).astype(float).transpose(1, 2, 0)
            fig_gt.imshow(t)
        else:
            fig_gt.imshow(p.argmax(1)[0], vmin=0, vmax=p.shape[1], cmap=cmap)
        self.log_matplotlib_figure(fig, self.global_step, size=(1000, 1000))
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

@app.command()
def train(
    train_bucket: Optional[str] = None,
    train_shards: Optional[str] = None,
    train_bs: int = 2,
    val_shards: Optional[str] = None,
    val_bs: int = 2,
    augmentation: str = "default",
    num_workers: int = 8,
    nepoch: int = 200000,
    checkpoint: int = 200000,
    mname: str = "ocropus.segmodels.segmentation_model_210910",
    lr: float = 0.01,
    lr_halflife: int = 500000,
    display_freq: int = 100,
    max_epochs: int = 10000,
    gpus: str = "0,",
    default_root_dir: str = "./_logs",
    resume: Optional[str] = None,
    dumpjit: Optional[str] = None,
    wandb: str = "",
) -> None:
    data = segdata.SegDataLoader(
        train_bucket = train_bucket,
        train_shards = train_shards,
        val_shards = val_shards,
        train_bs = train_bs,
        val_bs = val_bs,
        augmentation = augmentation,
        num_workers = num_workers,
        nepoch = nepoch,
    )
    batch = next(iter(data.train_dataloader()))
    print(
        f"# checking training batch size {batch[0].size()} {batch[1].size()}",
    )

    smodel = SegLightning(
        mname=mname,
        lr=lr,
        lr_halflife=lr_halflife,
        display_freq=display_freq,
    )
    smodel.train_bs = train_bs
    smodel.augmentation = augmentation

    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval="step"),
    )
    mcheckpoint = ModelCheckpoint(
        every_n_epochs=checkpoint,
    )
    callbacks.append(mcheckpoint)

    if wandb != "":
        wconfig = eval("{"+wandb+"}")
        print(f"# logging to {wconfig}")
        from pytorch_lightning.loggers import WandbLogger

        tconfig["logger"] = WandbLogger(**wconfig)
    else:
        print(f"# logging locally")

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=gpus,
        default_root_dir=default_root_dir,
        resume_from_checkpoint=resume,
    )

    if dumpjit is not None:
        assert resume is not None, "dumpjit requires a checkpoint"
        script = smodel.get_jit_model()
        torch.jit.save(script, dumpjit)
        print(f"# saved model to {dumpjit}")
        sys.exit(0)

    trainer.fit(smodel, data)


if __name__ == "__main__":
    app()
