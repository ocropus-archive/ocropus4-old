import random as pyrand
import sys
import simplejson

import numpy as np
import webdataset as wds
import matplotlib.pylab as plt
from torch import optim, nn

import torch
import ocrlib.ocrmodels as models
from torch.utils.data import DataLoader, IterableDataset
from . import psegutils
from . import utils
from . import slices
from . import summaries

import typer

from itertools import islice
import scipy.ndimage as ndi

app = typer.Typer()


def model_name(base, ntrain, loss, nscale=1e-3, lscale=1e6):
    ierr = int(lscale * loss)
    itrain = int(nscale * ntrain)
    return f"{base}-{itrain:08d}-{ierr:010d}.pth"


class SegTrainer(object):
    def __init__(
        self,
        model,
        *,
        lossfn=None,
        probfn=None,
        lr=1e-4,
        loginterval=60.0,
        saveinterval=600.0,
        savebase="segmodel",
        device=None,
        start_count=0,
        maxgrad=10.0,
        after_batch=None,
        base=None,
        writer=None,
        **kw,
    ):
        super().__init__()
        self.model = model
        self.lossfn = lossfn or nn.CrossEntropyLoss()
        self.probfn = probfn or (lambda x: x.softmax(1))
        self.last_lr = 1e33
        self.set_lr(lr)
        self.maybe_save = utils.Every(saveinterval)
        self.device = device
        self.base = base

        self.writer = writer
        self.losses = []
        self.clip_gradient = maxgrad
        self.after_batch = after_batch
        self.count = start_count
        self.margin = 32
        self.testbatch = None

    def set_lr(self, lr, momentum=0.9, delta=0.1):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if abs(lr - self.last_lr) / min(lr, self.last_lr) < delta:
            return False
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.last_lr = lr
        return True

    def compute_loss(self, outputs, targets):
        """Compute loss taking a margin into account."""
        b, d, h, w = outputs.shape
        b1, h1, w1 = targets.shape
        assert h <= h1 and w <= w1 and h1 - h < 5 and w1 - w < 5, (
            outputs.shape,
            targets.shape,
        )
        targets = targets[:, :h, :w]
        # lsm = outputs.log_softmax(1)
        if self.margin > 0:
            m = self.margin
            outputs = outputs[:, :, m:-m, m:-m]
            targets = targets[:, m:-m, m:-m]
        loss = self.lossfn(outputs, targets.to(outputs.device))
        return loss

    def train_batch(self, inputs, targets):
        """All the steps necessary for training a batch.

        Stores the last batch in self.last_batch.
        Adds the loss to self.losses.
        Clips the gradient if self.clip_gradient is not None.
        """
        self.model.train()
        self.optimizer.zero_grad()
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = self.model.forward(inputs)
        assert inputs.size(0) == outputs.size(0)
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (
            inputs.detach().cpu(),
            targets.detach().cpu(),
            outputs.detach().cpu(),
        )
        self.count += len(inputs)
        if callable(self.after_batch):
            self.after_batch(self)
        return loss.detach().item()

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch. Uses `probfn`."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return self.probfn(outputs.detach().cpu())

    def train(self, loader, ntrain):
        """Train over a dataloader for the given number of samples."""
        count = 0
        while count < ntrain:
            for sample in loader:
                images, targets = sample
                loss = self.train_batch(images, targets)
                self.losses.append(float(loss))
                count += len(images)
                if (
                    self.base is not None
                    and len(self.losses) > 20
                    and self.maybe_save()
                ):
                    fname = model_name(
                        self.base, self.count, np.mean(self.losses[-20:])
                    )
                    print(f"# saving {fname}", file=sys.stderr)
                    torch.save(self.model.state_dict(), fname)
                if self.maybe_log():
                    print(
                        f"{count:9d} {self.count:9d} {np.mean(self.losses[-20:]):.4g}",
                        file=sys.stderr,
                    )
                    self.writer.add_scalar("loss", loss, self.count)
                    inputs, targets, outputs = self.last_batch
                    self.writer.add_images(
                        "last",
                        utils.batch_images(inputs[0], outputs[0].softmax(0)),
                        self.count,
                    )
                    if self.testbatch is not None:
                        inputs, targets = self.testbatch
                        probs = self.probs_batch(inputs)
                        self.writer.add_images(
                            "test", utils.batch_images(inputs[0], probs[0]), self.count
                        )
                    self.writer.flush()
        self.writer.close()


def convert_img(
    image,
    degrade_probability=0.9,
    sigma=(0.5, 1.5),
    noise=(0.005, 0.05),
    nsigma=(1.0, 3.0),
):
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    image = image.astype(float) / 255.0
    if np.random.uniform() < degrade_probability:
        sigma = np.random.uniform(*sigma)
        if sigma > 1e-3:
            image = ndi.gaussian_filter(image, sigma)
        if noise[1] > noise[0]:
            noise_image = np.random.uniform(-1.0, 1.0, size=image.shape)
            noise_image = ndi.gaussian_filter(noise_image, np.random.uniform(*nsigma))
            noise_image -= np.amin(noise_image)
            noise_image /= np.amax(noise_image)
            noise_image *= np.random.uniform(noise[0], noise[1])
            image += noise_image
        image -= np.amin(image)
        image /= np.amax(image)
    assert image.ndim == 2
    return torch.tensor(image).unsqueeze(0)


def convert_seg(cs):
    return torch.tensor(cs).long()


class RandomRotationCounterexamples(IterableDataset):
    def __init__(self, source, probability, angles=[90.0, 270.0]):
        self.source = source
        self.p = probability
        self.angles = angles

    def __iter__(self):
        for sample in self.source:
            yield sample
            if np.random.uniform() >= self.p:
                continue
            # with the given probability, generate rotated versions of the input patch
            # and train them as non-text
            image, target = sample
            old_image, old_target = image.size(), target.size()
            assert image.ndim == 3 and image.shape[0] == 1
            image, target = image.numpy()[0], target.numpy()
            assert target.shape == image.shape and image.ndim == 2
            angle = pyrand.choice(self.angles)
            image, target = [
                ndi.rotate(a, angle, order=0, reshape=False) for a in [image, target]
            ]
            target[:, :] = 0
            image, target = torch.tensor(image).unsqueeze(0), torch.tensor(target)
            assert image.size() == old_image and target.size() == old_target, (
                image.size(),
                old_image,
                target.size(),
                old_target,
            )
            yield image, target


def make_seg_loader(
    dataset,
    extensions="png seg.png",
    batchsize=16,
    shuffle=0,
    rotation_probability=0.0,
    workers=0,
):
    transforms = [convert_img, convert_seg]
    if isinstance(extensions, str):
        extensions = extensions.split()
    assert len(extensions) == 2
    training = (
        wds.Dataset(dataset, handler=wds.warn_and_stop)
        .shuffle(shuffle)
        .decode("l8", handler=wds.warn_and_continue)
        .to_tuple("__key__", *extensions, handler=wds.warn_and_continue)
        .transform(transforms)
    )
    if rotation_probability > 0:
        training = RandomRotationCounterexamples(training, rotation_probability)
    training_dl = DataLoader(training, batch_size=batchsize, num_workers=workers)
    return training_dl


@app.command()
def train(
    dataset,
    extensions: str = "png seg.png",
    workers: int = 0,
    batchsize: int = 5,
    mname: str = "seg_lstm",
    ntrain: int = 1000000000,
    learningrates: str = "1e-4",
    rotation_probability: float = 0.0,
    shuffle: int = 10000,
    load: str = "",
    start_count: int = 0,
    model_base: str = "_segmodel",
    summary: str = "",
    testbatch: str = "",
    loginterval: float = 60.0,
    saveinterval: float = 600.0,
):
    """Train a segmentation model.

        :param dataset: dataset in POSIX tar format (containing image patches and segmentation patches)
        :param extensions: extensions to be used for training
        :param workers: number of workers used for loading (use 0 on Docker/K8s)
        :param batchsize: batchsize used for training
        :param mname: model name to be instantiated
        :param learningrates: floating point or function-of-#samples giving learning rate
        :param every: how often to log results
        :param rotation_probability: augmentation by 90/270 degree rotations
        :param shuffle: size of input shuffle
        :param load: model to preload
        :param start_count: starting count for model
        :param logdir: directory for logging Tensorboard
        :param model_base: basename for saving models
        :param testbatch: used to load a test batch for display (1234@tarfile)
    """

    training_dl = make_seg_loader(
        dataset,
        extensions=extensions,
        batchsize=batchsize,
        shuffle=shuffle,
        rotation_probability=rotation_probability,
        workers=workers,
    )
    model = models.make(mname)
    if load not in ["", "none"]:
        model = utils.load_model(model, load)

    learningrates = eval(learningrates)
    trainer = SegTrainer(
        model, model_base=model_base, loginterval=loginterval, saveinterval=saveinterval
    )
    trainer.writer = summaries.make_tensorboard_style_summary(summary)

    if testbatch != "":
        n, tname = testbatch.split("@", 1)
        tloader = make_seg_loader(tname, extensions=extensions)
        trainer.testbatch = next(islice(tloader, int(n)))

    trainer.count = start_count
    trainer.train(training_dl, ntrain)


def label_marked_components(
    probs, threshold=0.5, sep_threshold=0.3, extra={}, max_dist_from_marker=30
):
    """Given an RGB image with G=sep and B=marker, computes a segmentation."""
    word_markers = probs[:, :, 2] > 0.5
    extra["markers"] = word_markers = ndi.minimum_filter(
        ndi.maximum_filter(word_markers, (1, 3)), (1, 3)
    )
    word_labels, n = ndi.label(word_markers)
    distances, sources = ndi.distance_transform_edt(
        1 - word_markers, return_indices=True
    )
    extra["sources"] = word_sources = word_labels[sources[0], sources[1]] * (
        distances < max_dist_from_marker
    )
    word_boundaries = np.maximum(
        (np.roll(word_sources, 1, 0) != word_sources),
        np.roll(word_sources, 1, 1) != word_sources,
    )
    extra["boundaries"] = word_boundaries = ndi.minimum_filter(
        ndi.maximum_filter(word_boundaries, 4), 2
    )
    extra["separators"] = separators = np.maximum(
        probs[:, :, 1] > sep_threshold, word_boundaries
    )
    extra["all_components"] = all_components, n = ndi.label(1 - separators)
    word_markers = (probs[:, :, 2] > 0.5) * (1 - separators)
    extra["markers"] = word_markers = ndi.minimum_filter(
        ndi.maximum_filter(word_markers, (1, 3)), (1, 3)
    )
    extra["labels"] = word_labels, n = ndi.label(word_markers)
    correspondence = 1000000 * word_labels + all_components
    ncomponents = np.amax(all_components) + 1
    wordmap = np.zeros(ncomponents, dtype=int)
    for word, comp in [
        (k // 1000000, k % 1000000) for k in np.unique(correspondence.ravel())
    ]:
        if comp == 0:
            continue
        if word == 0:
            continue
        if wordmap[comp] > 0:
            print("wordmap computation warning:", word, comp)
        wordmap[comp] = word
    extra["result"] = result = wordmap[all_components]
    return result


class Segmenter(object):
    def __init__(self, device="cpu", marker_threshold=0.9, separator_threshold=0.9):
        self.device = device
        self.threshold = marker_threshold
        self.sep_threshold = separator_threshold

    def load(self, fname):
        model = models.make("seg_lstm", device="cpu")
        model = utils.load_model(model, fname)
        model.eval()
        print(f"# loaded {fname}", file=sys.stderr)
        model.to(self.device)
        self.model = model

    def predict_probs(self, batch):
        assert float(batch.max()) <= 1.0
        assert batch.ndim == 4
        assert batch.size(1) == 1
        with torch.no_grad():
            result = self.model(batch.to(self.device)).detach().cpu().softmax(1)
        return result

    def predict_np(self, image):
        # print("#<", np.amin(image), np.mean(image), np.amax(image))
        assert image.ndim in [2, 3]
        if image.ndim == 3:
            image = np.mean(image, 2)
        input = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        probs = self.predict_probs(input)[0].cpu().numpy().transpose(1, 2, 0)
        # print("#>", np.amin(probs), np.mean(probs), np.amax(probs))
        # print(array_infos(input=input, probs=probs))
        return probs

    def predict_tiled(self, image, r=1024, b=32, debug_tiles=False):
        import scipy.ndimage as ndi

        if image.ndim == 3:
            image = np.mean(image, 2)
        h, w = image.shape
        result = None
        for y in range(0, h, r - 1):
            for x in range(0, w, r - 1):
                ph, pw = min(r, h - y) + 2 * b + 2, min(r, w - x) + 2 * b + 2
                patch = ndi.affine_transform(
                    image,
                    np.eye(2),
                    offset=(y - b, x - b),
                    output_shape=(ph, pw),
                    order=1,
                    mode="nearest",
                )
                output = self.predict_np(patch)
                # pred_images.save(patch); pred_probs.save(output)
                output = output[b:-b, b:-b]
                # print("#", patch.shape, output.shape)
                if result is None:
                    result = np.zeros((h, w, output.shape[2]))
                oh, ow, od = output.shape
                result[y : y + oh, x : x + ow, :] = output
        return result


extract_segment_params = dict(
    outer_maxheight=3.0,  # maxheight of outer limit box, multiple of marker height
    outer_hpad=5,  # horizontal padding of outer limit box
    vclose=0,  # vertical morphological closing of separators
    hclose=10,  # horizontal morphological closing of separators
    hpad=8,  # horizontal padding prior to extraction
    vpad=10,  # vertical padding prior to extraction
    marker_threshold=[0.3, 0.7],  # hysteresis threshold for markers
    separator_threshold=[0.3, 0.7],  # hysteresis threshold for separator
)


def extract_segments(probs, params=extract_segment_params, show=0.0):
    markers = psegutils.hysteresis_threshold(probs[:, :, 2], *params.marker_threshold)
    separators = psegutils.hysteresis_threshold(
        probs[:, :, 1], *params.separator_threshold
    )
    if params.hclose > 0:
        separators = ndi.minimum_filter(
            ndi.maximum_filter(separators, (1, params.hclose)), (1, params.hclose)
        )
    if params.vclose > 0:
        separators = ndi.minimum_filter(
            ndi.maximum_filter(separators, (params.vclose, 1)), (params.vclose, 1)
        )
    word_labels, n = ndi.label(markers)
    segment_labels, n = ndi.label(1 - np.clip(separators, 0, 1))
    seg2word = {
        s: w for s, w in psegutils.correspondences(segment_labels, word_labels).T
    }
    if show > 0:
        plt.clf()
        plt.subplot(121)
        plt.imshow(separators)
        plt.subplot(122)
        plt.imshow(probs)
    word_boxes = [(slice(0, 0), slice(0, 0))] + ndi.find_objects(word_labels)
    result = []
    for i, seg_box in enumerate(ndi.find_objects(segment_labels)):
        label = i + 1
        wlabel = seg2word.get(label, 0)
        if wlabel == 0:
            continue
        wh, ww = word_box = word_boxes[wlabel]
        outer_box = (
            slices.dilate_slice(wh, params.outer_maxheight),
            slices.pad_slice(ww, params.outer_hpad),
        )
        final = slices.intersect_boxes(outer_box, seg_box)
        final = (
            slices.pad_slice(final[0], params.vpad),
            slices.pad_slice(final[1], params.hpad),
        )
        if slices.box_area(final) < 10:
            continue
        mask = (segment_labels[final] == label).astype("uint8")
        if show > 0:
            slices.plot_box(seg_box, color="orange")
            slices.plot_box(word_box, color="cyan")
            slices.plot_box(outer_box, color="yellow")
            slices.plot_box(final, color="white")
        result.append((final, mask))
    if show > 0:
        plt.ginput(1, show)
    return result


@app.command()
def segment(
    input: str,
    output: str = "",
    extensions: str = "png;jpg;jpeg",
    mode: str = "json",
    model: str = "",
    tilesize: int = 2048,
    device: str = "cuda",
    maxcount: int = 999999999,
    show: float = 0.0,
):
    """Segment page images and write either .json bounding boxes or individual segment images with masks.

        :param input: input file in POSIX tar format
        :param output: input file in POSIX tar format
        :param extensions: extensions used for getting image files
        :param mode: one of "probs", "json", "json+", "extract"
        :param model: PyTorch model to be loaded
        :param tilesize: size of tiles for inference (for dealing with GPU memory limits)
        :param device: device used for inference (e.g., "cpu", "cuda", "cuda:1")
        :param maxcount: stop after processing this many images (for testing)
        :param show: if >0.0, show each result for this many seconds (or until clicked) in GUI
    """
    segmenter = Segmenter(device=device)
    segmenter.load(model)
    figsize = (18, 6)
    assert output != ""
    # assert not os.path.exists(output)
    assert model != ""
    if isinstance(extensions, str):
        extensions = extensions.split()
    testing = wds.Dataset(input).decode().extract("__key__", *extensions)
    if show > 0.0:
        plt.ion()
        plt.figure(figsize=figsize)
    with wds.TarWriter(output) as sink:
        count = 0
        for key, raw_image in testing:
            if count >= maxcount:
                break
            print(key, file=sys.stderr)
            image = convert_img(raw_image, sigma=(0.3, 0.3), noise=(0.0, 0.0))
            image = image.numpy()[0]
            assert np.amin(image) >= 0 and np.amax(image) <= 1
            print(f"{key}: predicting", file=sys.stderr)
            probs = segmenter.predict_tiled(image, r=tilesize)
            if mode == "probs":
                sample = {"__key__": key, "png": raw_image, "probs.png": probs}
                sink.write(sample)
                continue
            print(f"{key}: extracting", file=sys.stderr)
            boxes = extract_segments(probs)
            if show > 0.0:
                plt.clf()
                index = 1
                rows, cols = 10, 20
                for box, mask in boxes[: rows * cols]:
                    wimage = image[box]
                    plt.subplot(rows, cols, index)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(wimage)
                    index += 1
                plt.ginput(1, show)
            if mode in ["json", "json+"]:
                result = [utils.encode_bbox(box) for box, mask in boxes]
                result = simplejson.dumps(result)
                sample = {"__key__": key, "png": raw_image, "seg.json": result}
                if mode == "json+":
                    sample["probs.png"] = probs
                sink.write(sample)
            elif mode == "extract":
                for box, mask in boxes:
                    wimage = raw_image[box]
                    wkey = key + "/" + ",".join(map(str, utils.bbox2list(box)))
                    sink.write(
                        {
                            "__key__": wkey,
                            "bbox.json": simplejson.dumps(utils.encode_bbox(box)),
                            "png": wimage,
                            "mask.png": mask,
                        }
                    )
            else:
                raise ValueError(f"{mode}: unknown mode")
            count += 1


if __name__ == "__main__":
    app()
