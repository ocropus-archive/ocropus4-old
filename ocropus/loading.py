import requests
import hashlib
import os.path
import tempfile
import shutil
import time
import os
import sys
import re
from urllib.parse import urlparse
import importlib
import torch
import io
import tempfile
import warnings
import os.path

#
# Modules
#

default_path = "ocropus.models:ocropus.experimental_models:ocropus.old_models"
module_path = os.environ.get("MODEL_MODULES", default_path).split(":")

num_modules = 0


def load_module(file_name, module_name=None):
    """Load a module from a file."""
    global num_modules
    if module_name is None:
        module_name = "__mod{num_modules}"
        num_modules += 1
    loader = importlib.machinery.SourceFileLoader(module_name, file_name)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def make_module(text, module_name=None):
    """Create a module from a string containing source code."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as stream:
        stream.write(text)
        stream.flush()
        return load_module(stream.name, module_name=module_name)


#
# File I/O
#


def read_file(fname):
    """Read a file into memory."""
    with open(fname) as stream:
        return stream.read()


#
# Loading/Saving Structures
#


def torch_dumps(obj):
    """Dump a data structure to a string using torch.save."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer()


def torch_loads(buf, device="cpu"):
    """Load a data structure from a string using torch.load."""
    return torch.load(io.BytesIO(buf), map_location=torch.device(device))


#
# looking up functions in modules and/or source files
#


def load_function(fun_name, src):
    """Instantiate a PyTorch model from Python source code."""
    warnings.warn("direct loading of model source code")
    if src.endswith(".py"):
        src = read_file(src)
    mmod = make_module(src)
    return getattr(mmod, fun_name, None)


def find_function(name, path):
    path = path.split(":") if isinstance(path, str) else path
    for mname in path:
        mmod = importlib.import_module(mname)
        if hasattr(mmod, name):
            return getattr(mmod, name)
    return None


#
# model loading and saving
#


modeldir = os.environ.get("OCROMODELS", os.path.join(os.environ.get("HOME", "/tmp"), ".ocropus4/models"))


def download_cache(url, modeldir=modeldir, timeout=60):
    """Download a file to a cache directory and return the path to the file."""

    if modeldir is None:
        modeldir = os.environ["HOME"] + "/.cache/ocropus"
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    _, fname = os.path.split(urlparse(url).path)
    # cache_path = os.path.join(modeldir, hashlib.sha1(url.encode("utf-8")).hexdigest() + "_" + fname)
    cache_path = os.path.join(modeldir, fname)
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) > time.time() - timeout:
            print(f"# using {cache_path}", file=sys.stderr)
            return cache_path
        else:
            os.remove(cache_path)
    print(f"# downloading {url} to {cache_path}", file=sys.stderr)
    r = requests.get(url, stream=True)
    with open(cache_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return cache_path


def load_jit_model(fname, device="cpu"):
    import torch.jit

    if re.search(r"(?i)^https?:.*", fname):
        cached = download_cache(fname)
        model = torch.jit.load(cached)
    else:
        model = torch.jit.load(fname)
    return model
