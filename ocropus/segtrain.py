import io, json, sys
from io import StringIO
from typing import Any, Dict, List

import numpy as np
import PIL
import PIL.Image
import pytorch_lightning as pl
import torch
import yaml
from matplotlib import gridspec
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy import ndimage as ndi
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from . import confparse, segmodels, segdata

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
        self.params = confparse.Params(locals())
        self.save_hyperparameters()
        self.hparams.config = json.dumps(self.params.__dict__)
        self.model = segmodels.SegModel(mname, **segmodel)
        self.get_jit_model()
        print("model created and is JIT-able")

    def get_jit_model(self):
        script = torch.jit.script(self.model)
        return script

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr)
        scheduler = LambdaLR(optimizer, self.schedule)
        return [optimizer], [scheduler]

    def schedule(self, epoch: int):
        return 0.5 ** (epoch // self.params.lr_halflife)

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
        if self.params.margin > 0:
            m = self.params.margin
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
        if self.params.weightmask >= 0 or self.params.bordermask >= 0:
            mask = self.make_weight_mask(
                targets, self.params.weightmask, self.params.bordermask
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
        if mode == "train" and index % self.params.display_freq == 0:
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


def train(argv: List[str]) -> None:
    argv = argv or []
    config = confparse.parse_args(argv, default_config)
    yaml.dump(config, sys.stdout)

    dataconfig = config["data"]
    data = segdata.SegDataLoader(**dataconfig)
    batch = next(iter(data.train_dataloader()))
    print(
        f"# checking training batch size {batch[0].size()} {batch[1].size()}",
    )

    smodel = SegLightning(**config["lightning"])
    smodel.hparams.train_bs = dataconfig["train_bs"]
    smodel.hparams.augmentation = dataconfig["augmentation"]

    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval="step"),
    )
    cpconfig = config["checkpoint"]
    mcheckpoint = ModelCheckpoint(**cpconfig)
    callbacks.append(mcheckpoint)

    tconfig = config["trainer"]

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

    if config.get("dumpjit"):
        assert (
            "resume_from_checkpoint" in config["trainer"]
        ), "must resume from checkpoint to dump JIT script"
        script = smodel.get_jit_model()
        torch.jit.save(script, config["dumpjit"])
        print(f"# saved model to {config['dumpjit']}")
        sys.exit(0)

    trainer.fit(smodel, data)


if __name__ == "__main__":
    train(sys.argv[1:])
