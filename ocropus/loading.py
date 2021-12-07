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

from . import slog

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
# model and state bundles
#


def dict_to_model(state, module_path=module_path):
    constructor = find_function(state["mname"], module_path)
    assert constructor is not None, f"can't find {state['mname']} in {module_path}"
    args, kw = state["margs"]
    model = constructor(*args, **kw)
    model.mname_ = state.get("mname")
    model.margs_ = state["margs"]
    model.step_ = state.get("step", 0)
    extra = state.get("extra", {})
    model.extra_ = extra if extra != {} else dict(state, mstate=None)
    model.load_state_dict(state["mstate"])
    return model


def model_to_dict(model):
    return dict(
        mstate=model.state_dict(),
        mname=model.mname_,
        margs=model.margs_,
        extra=getattr(model, "extra_", {}),
        step=getattr(model, "step_", 0),
    )


def construct_model(name, *args, module_path=module_path, **kw):
    if name.endswith(".py") or name.startswith("\n"):
        warnings.warn("source used in construct_model")
        constructor = load_function(name)
    else:
        constructor = find_function(name, module_path)
        assert constructor is not None, f"can't find {name} in {module_path}"
    model = constructor(*args, **kw)
    model.mname_ = name
    model.margs_ = (args, kw)
    model.step_ = 0
    return model


#
# model loading and saving
#


modeldir = os.environ.get("OCROMODELS", os.path.join(os.environ.get("HOME", "/tmp"), ".ocropus4/models"))


def download_cache(url, modeldir=modeldir):
    """Download a file to a cache directory and return the path to the file."""

    if modeldir is None:
        modeldir = os.environ["HOME"] + "/.cache/ocropus"
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    _, fname = os.path.split(urlparse(url).path)
    cache_path = os.path.join(modeldir, hashlib.sha1(url.encode("utf-8")).hexdigest() + "_" + fname)
    if os.path.exists(cache_path):
        return cache_path
    print(f"# downloading {url} to {cache_path}", file=sys.stderr)
    r = requests.get(url, stream=True)
    with open(cache_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return cache_path


def load_only_model(fname, *args, module_path=module_path, device="cpu", **kw):
    if re.search(r"(?i)^https?:.*\.pth$", fname):
        return load_only_model(download_cache(fname), *args, module_path=module_path, device=device, **kw)
    if fname.endswith(".sqlite3"):
        assert os.path.exists(fname)
        logger = slog.Logger(fname)
        state = logger.load_last()
    else:
        state = torch.load(fname, map_location=torch.device(device))
    model = dict_to_model(state, module_path=module_path)
    return model


def load_jit_model(fname, device="cpu"):
    import torch.jit
    if re.search(r"(?i)^https?:.*", fname):
        cached = download_cache(fname)
        print("*** remote", cached)
        model = torch.jit.load(cached)
    else:
        print("*** local", fname)
        model = torch.jit.load(fname)
    return model


def load_or_construct_model(path, *args, module_path=module_path, device="cpu", **kw):
    _, ext = os.path.splitext(path)
    if ext in [".py", ".sqlite3", ".pth", ".pt", ".jit", ".onnx"]:
        return load_only_model(path, *args, module_path=module_path, device=device, **kw)
    else:
        return construct_model(path, *args, module_path=module_path, **kw)


##
## Model Saving
##
#
#
#def save_model_as_dict(model, fname, step=None):
#    """Save a PyTorch model (parameters and source code) to a file."""
#    state = model_to_dict(model)
#    torch.save(state, fname)
#
#
#def dump_model_as_dict(model):
#    """Dump a model to a string using torch.save."""
#    buf = io.BytesIO()
#    save_model_as_dict(model, buf)
#    return buf.getbuffer()
#
#
#def log_model(logger, model, loss=None, step=None, optimizer=None, **kw):
#    assert loss is not None
#    assert step is not None
#    state = model_to_dict(model)
#    state.update(kw)
#    logger.save_ocrmodel(model, scalar=loss, step=step)
#    logger.flush()
#
