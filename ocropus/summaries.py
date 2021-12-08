import os
import sys

import matplotlib.pyplot as plt
import numpy as np


class MockSummaryWriter(object):
    """A summary writer that does nothing."""

    def __init__(self, *args, **kw):
        pass

    def add_scalar(self, *args, **kw):
        pass

    def add_image(self, *args, **kw):
        pass

    def add_images(self, *args, **kw):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class TextSummaryWriter(object):
    """A summary writer providing minimal text output."""

    def __init__(self, path, *args, **kw):
        self.path = path

    def add_scalar(self, name, value, count, **kw):
        print(f"[{self.path}/{name}] {count:9d} {value:8.4g}", file=sys.stderr)

    def add_image(self, name, image, count, **kw):
        print(
            f"[{self.path}/{name}] {count:9d} image:{tuple(image.shape)}",
            file=sys.stderr,
        )

    def add_images(self, name, images, count, **kw):
        print(
            f"[{self.path}/{name}] {count:9d} images:{tuple(images.shape)}",
            file=sys.stderr,
        )

    def flush(self):
        pass

    def close(self):
        pass


class DynamicMatplotlibSummaryWriter(object):
    """A summary writer using matplotlib for displaying progress."""

    def __init__(self, path):
        self.fig = plt.figure()
        self.data = {}
        if "/" in path:
            path, names = path.split("/", 1)
            names = names.split(",")
        else:
            names = []
        self.rows, self.cols = map(int, path.split(","))
        self.names = []
        for i, name in enumerate(names):
            ax = self.axis(name)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.ion()

    def axis(self, name):
        try:
            index = self.names.index(name)
            return self.fig.axes[index]
        except ValueError:
            pass
        if len(self.names) == self.rows * self.cols:
            return None
        self.names.append(name)
        ax = self.fig.add_subplot(self.rows, self.cols, len(self.names))
        return ax

    def add_scalar(self, name, value, count, *args, **kw):
        self.data.setdefault(name, []).append((count, value))
        ax = self.axis(name)
        if ax is None:
            return
        ax.cla()
        ax.set_title(name)
        ax.plot(*zip(*self.data[name]))

    def add_image(self, name, image, count, *args, **kw):
        self.data[name] = image
        try:
            image = image.numpy()
        except:
            pass
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        ax = self.axis(name)
        if ax is None:
            return
        ax.cla()
        ax.set_title(name)
        ax.imshow(image)

    def add_images(self, basename, images, count, **kw):
        for i in range(len(images)):
            name = f"{basename}[{i}]"
            self.add_image(name, images[i], count)

    def flush(self):
        self.fig.ginput(1, 0.001)

    def close(self):
        self.fig.close()
        del self.fig


class RedisSummaryWriter(object):
    """A summary writer writing to REDIS."""

    def __init__(self, *args, **kw):
        raise Excpetion("unimplemented")

    def add_scalar(self, *args, **kw):
        pass

    def add_image(self, *args, **kw):
        pass

    def add_images(self, *args, **kw):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def make_tensorboard_style_summary(dest):
    """Creates a Tensorboard-style logger (not necessarily Tensorboard).

    Destination:
    - "" or None: return None
    - plain directory or "tensorboard:..." = torch.utils.tensorboard
    - "matplotlib:" = directly log to matplotlib window
    - "redis:" = logging to Redis (unimplemented)
    - "sqlite3:" = logging to Sqlite3 using Python pickles (unimplemented)
    """

    if dest is None or dest == "":
        return TextSummaryWriter("summary")

    if ":" not in dest:
        dest = "tensorboard:" + dest

    schema, path = dest.split(":", 1)

    if schema == "text":
        return TextSummaryWriter(path)
    elif schema == "redis":
        return RedisSummaryWriter(path)
    elif schema in ["mpl", "matplotlib"]:
        return DynamicMatplotlibSummaryWriter(path)
    elif schema == ["tb", "tensorboard"]:
        from torch.utils import tensorboard

        if os.path.exists(path):
            new_directory = utils.make_unique(path)
            print(f"# {path} already exists, using {new_directory} instead")
            path = new_directory
        os.makedirs(path, exist_ok=True)
        return tensorboard.SummaryWriter(path)
