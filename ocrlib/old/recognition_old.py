import sys
import numpy as np
import scipy.ndimage as ndi
import torch
import time
import typer
import editdistance
import webdataset as wds
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmore import layers
import matplotlib.pyplot as plt

from . import ocrmodels as models
from . import ocrhelpers as helpers
from . import utils
from . import slog

app = typer.Typer()

default_charset = [chr(i) for i in range(32, 126)]

display = True


def dshow(image, where=111):
    plt.subplot(where)
    plt.imshow(image)


class Codec(object):
    # FIXME: generalize this
    def __init__(self, charset=default_charset):
        self.charset = charset

    def __len__(self):
        return len(self.charset) + 1

    def encode_chr(self, c):
        try:
            return self.charset.index(c) + 1
        except ValueError:
            # last character is always used as default/missing
            return len(self.charset)

    def encode(self, s):
        return [self.encode_chr(c) for c in s]

    def decode(self, l):
        return "".join([self.charset[k - 1] for k in l])


the_codec = Codec()


def seq_is_normalized(probs):
    return (abs(probs.sum(1) - 1) < 1e-4).all()


def ctc_decode(probs, sigma=1.0, threshold=0.7, kind=None, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    # plt.ginput(1, 100); plt.clf()
    probs = helpers.asnp(probs.T)
    assert seq_is_normalized(probs), "input not normalized; did you apply .softmax()?"
    # dshow(probs, 221)
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, np.newaxis]
    # dshow(probs, 222)
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = np.tile(labels[:, np.newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    # dshow(probs, 223)
    maxima = ndi.maximum_position(probs, mask, np.arange(1, np.amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]
    # plt.ginput(1, 100)


def batch_images(*args):
    images = [a.cpu().float() for a in args]
    dims = np.array([tuple(a.shape) for a in images])
    maxdims = [x for x in np.max(dims, 0)]
    result = torch.zeros([len(images)] + maxdims)
    for i, a in enumerate(images):
        d, h, w = a.shape
        if d == 1:
            result[i, :, :h, :w] = a
        else:
            result[i, :d, :h, :w] = a
    return result


def collate4ocr(samples):
    """Collate image+sequence samples into batches.

    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :w, :h, :d] = im
    return (result, seqs)


def model_device(model):
    """Find the device of a model."""
    return next(model.parameters()).device


def pack_for_ctc(seqs):
    """Pack a list of sequences for nn.CTCLoss."""
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


ctc_loss = nn.CTCLoss(zero_infinity=True)


def ctc_loss_bdl(log_probs, targets, device="cpu"):
    """Compute CTC Loss on BDL-order tensors.

    This is a wrapper around nn.CTCLoss that does a few things:
    - it accepts the output as a plain tensor (without lengths)
    - it accepts output tensors in BDL order (regular CTC: LBD)
    """
    assert isinstance(targets, tuple) and len(targets) == 2
    assert float(log_probs.min()) < 0
    assert not seq_is_normalized(log_probs)  # REMOVE THIS LATER
    assert log_probs.shape[1] == len(the_codec)  # REMOVE THIS LATER
    layers.check_order(log_probs, "BDL")
    bs, d, seqlen = log_probs.size()
    olens = torch.full((bs,), seqlen, dtype=torch.long)
    log_probs = layers.reorder(log_probs, "BDL", "LBD")
    targets, tlens = targets
    assert tlens.size(0) == bs
    assert tlens.sum() == targets.size(0)
    loss = ctc_loss(log_probs.to(device=device), targets.to(device=device), olens.to(device=device), tlens.to(device=device))
    return loss


class LineTrainer(object):
    def __init__(
        self,
        model,
        *,
        lossfn=None,
        lr=1e-4,
        every=3.0,
        savefreq=600,
        count=0,
        device=None,
        maxgrad=10.0,
        after_batch=None,
        base=None,
        **kw,
    ):
        super().__init__()
        self.base = base
        self.model = model
        self.device = None
        self.every = every
        self.savefreq = savefreq
        self.last_save = 0
        self.losses = []
        self.last_lr = 1e33
        self.last_report = 0
        self.set_lr(lr)
        self.clip_gradient = maxgrad
        self.after_batch = after_batch
        self.count = count
        self.writer = None
        self.model_written = False
        self.testbatch = None

    def set_lr(self, lr, momentum=0.9, delta=0.1):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if abs(lr - self.last_lr) / min(lr, self.last_lr) < delta:
            return False
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.last_lr = lr
        return True

    def train_batch(self, inputs, seqs):
        """All the steps necessary for training a batch.

        Stores the last batch in self.last_batch.
        Adds the loss to self.losses.
        Clips the gradient if self.clip_gradient is not None.
        """
        self.model.train()
        self.optimizer.zero_grad()
        targets = pack_for_ctc(seqs)
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = self.model.forward(inputs)
        assert inputs.size(0) == outputs.size(0)
        loss = ctc_loss_bdl(outputs.log_softmax(1), targets, device=outputs.device)
        assert not np.isinf(float(loss))
        assert not np.isnan(float(loss))
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (inputs.detach().cpu(), seqs, outputs.detach().cpu())
        self.count += len(inputs)
        if callable(self.after_batch):
            self.after_batch(self)
        return loss.detach().item()

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(1)

    def train_epoch(self, loader, ntrain=1e12):
        """Train over a dataloader for the given number of epochs."""
        count = 0
        for sample in loader:
            if count >= ntrain:
                break
            images, targets = sample
            loss = self.train_batch(images, targets)
            if np.isinf(float(loss)) or np.isnan(float(loss)):
                print(f"bad loss, ignoring batch", file=sys.stderr)
                continue
            self.losses.append(float(loss))
            self.count += len(images)
            now = time.time()
            if len(self.losses) > 20 and now - self.last_save > self.savefreq:
                if self.base is not None:
                    fname = utils.model_name(
                        self.base, self.count, np.mean(self.losses[-20:])
                    )
                    print(f"# saving {fname}", file=sys.stderr)
                    torch.save(self.model.state_dict(), fname)
                else:
                    print("# not saving because self.base is None", file=sys.stderr)
                self.last_save = now
            if now - self.last_report > self.every:
                inputs, targets, outputs = self.last_batch
                probs = outputs.detach().cpu().softmax(1)
                correct = the_codec.decode(targets[0])
                result = the_codec.decode(ctc_decode(probs[0]))
                print(
                    f"# {self.count:10d} {loss:8.2g} '{correct}' '{result}'",
                    file=sys.stderr,
                )
                self.write_report(loss, inputs, outputs, targets)
                self.last_report = now
                if display:
                    import matplotlib.pyplot as plt

                    plt.ion()
                    plt.clf()
                    plt.subplot(2, 2, 1)
                    plt.imshow(inputs[0][0].numpy())
                    plt.subplot(2, 2, 2)
                    plt.imshow(probs[0], vmin=0, vmax=1.0)
                    plt.subplot(2, 2, 4)
                    plt.imshow(probs[0, :, :100])
                    plt.subplot(2, 2, 3)
                    for i in range(len(the_codec)):
                        plt.plot(probs[0, i])
                    plt.ginput(1, 0.01)

    def write_report(self, loss, inputs, outputs, targets, prefix=""):
        if self.writer is None:
            return
        self.writer.add_scalar("loss", loss, self.count)
        self.writer.add_image(prefix + "input", inputs[0], self.count)
        self.writer.add_image(prefix + "output", outputs[0].softmax(0), self.count)
        self.writer.flush()

    def predict_batch(self, inputs, **kw):
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result

    def old_errors(self, loader):
        """Compute OCR errors using edit distance."""
        total = 0
        errors = 0
        for inputs, targets in loader:
            targets, tlens = targets
            predictions = self.predict_batch(inputs)
            start = 0
            for p, l in zip(predictions, tlens):
                t = targets[start : start + l].tolist()
                errors += editdistance.distance(p, t)
                total += len(t)
                start += l
                if total > self.maxcount:
                    break
            if total > self.maxcount:
                break
        return errors, total


@app.command()
def recognize():
    raise NotImplementedError


def identity(x):
    return x


def line2tensor(x):
    return (torch.FloatTensor(x) / 255.0).unsqueeze(0)
    return (torch.tensor(x).float() / 255.0).unsqueeze(0)


def text2tensor(s):
    return torch.tensor(the_codec.encode(s)).long()


def goodsample(pair):
    image, txt = pair
    if len(txt) < 2:
        return False
    if image.shape[0] < 5 or image.shape[1] < 5:
        return False
    return True


@app.command()
def train(
    dataset: str,
    extensions: str = "jpg;jpeg;ppm;png txt",
    learning_rate=1e-4,
    summary: str = "text:",
    ntrain: int = 10000000,
    batchsize: int = 20,
    shuffle: int = 20000,
    nepochs: int = 300,
    nworkers: int = 1,
    model: str = "lstm_ctc",
):
    if isinstance(extensions, str):
        extensions = extensions.split()
    assert len(extensions) == 2
    training = (
        wds.Dataset(dataset, handler=wds.warn_and_stop)
        .shuffle(shuffle)
        .decode("l8", handler=wds.warn_and_continue)
        .to_tuple(*extensions, handler=wds.warn_and_continue)
        .select(goodsample)
        .map_tuple(line2tensor, text2tensor)
    )
    training_dl = DataLoader(
        training, batch_size=batchsize, collate_fn=collate4ocr, num_workers=nworkers
    )
    model = models.make(model, noutput=len(the_codec))
    print(model)
    trainer = LineTrainer(model)
    trainer.set_lr(learning_rate)
    trainer.writer = slog.Logger()
    for epoch in range(nepochs):
        trainer.train_epoch(training_dl, ntrain)


if __name__ == "__main__":
    app()
