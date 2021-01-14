import importlib
import torch
import io
import tempfile
import warnings
import os.path

from . import slog

default_path = "ocrlib.models:ocrlib.experimental_models:old_models"
module_path = os.environ.get("MODEL_MODULES", default_path).split(":")

#
# Modules
#

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


def torch_loads(buf):
    """Load a data structure from a string using torch.load."""
    return torch.load(io.BytesIO(buf))


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
    args, kw = state["margs"]
    model = constructor(*args, **kw)
    model.mname_ = state.get("mname")
    model.margs_ = state["margs"]
    model.step_ = state.get("step", 0)
    model.extra_ = state.get("extra", {})
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
    model = constructor(*args, **kw)
    model.mname_ = name
    model.margs_ = (args, kw)
    model.step_ = 0
    return model


#
# model loading and saving
#


def load_only_model(fname, *args, module_path=module_path, **kw):
    if fname.endswith(".sqlite3"):
        assert os.path.exists(fname)
        logger = slog.Logger(fname)
        state = logger.load_last()
    else:
        state = torch.load(fname)
    model = dict_to_model(state, module_path=module_path)
    return model


def load_or_construct_model(path, *args, module_path=module_path, **kw):
    if os.path.splitext(path)[1] in ["py", "sqlite3"]:
        return load_only_model(path, *args, module_path=module_path, **kw)
    else:
        return construct_model(path, *args, module_path=module_path, **kw)


#
# Model Saving
#


def save_model_as_dict(model, fname, step=None):
    """Save a PyTorch model (parameters and source code) to a file."""
    state = model_to_dict(model)
    torch.save(state, fname)


def dump_model_as_dict(model):
    """Dump a model to a string using torch.save."""
    buf = io.BytesIO()
    save_model_as_dict(model, buf)
    return buf.getbuffer()


def log_model(logger, model, loss=None, step=None, optimizer=None):
    assert loss is not None
    assert step is not None
    state = model_to_dict(model)
    logger.save("model", state, scalar=loss, step=step)
    logger.flush()
