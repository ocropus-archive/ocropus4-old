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
# Model Creation/Loading
#


def get_function(path):
    mname, fun_name = path.rsplit(".", 1)
    try:
        mmod = importlib.import_module(mname)
    except ModuleNotFoundError:
        return None
    if not hasattr(mmod, fun_name):
        return None
    return getattr(mmod, fun_name)


def make_model(src, *args, fun_name="make_model", **kw):
    """Instantiate a PyTorch model from Python source code."""
    if src.endswith(".py"):
        src = read_file(src)
    mmod = make_module(src)
    model = getattr(mmod, fun_name)(*args, **kw)
    model.msrc_ = src
    model.margs_ = (args, kw)
    return model


def construct_model(path, *args, module_path=[], **kw):
    if path.endswith(".py"):
        warnings.warn(".py file used in construct_model")
        src = read_file(path)
        mmod = make_module(src)
        model = getattr(mmod, "make_model")(*args, **kw)
        model.msrc_ = src
        model.margs_ = (args, kw)
        model.step_ = 0
        return model
    elif path.startswith("\n"):
        warnings.warn("source code used in construct_model")
        mmod = make_module(src)
        model = getattr(mmod, "make_model")(*args, **kw)
        model.msrc_ = src
        model.margs_ = (args, kw)
        model.step_ = 0
        return model
    else:
        model = get_function(path)(*args, **kw)
        model.mpath_ = path
        model.margs_ = (args, kw)
        model.step_ = 0
        return model


def load_model(path, *args, module_path=[], **kw):
    if path.endswith(".sqlite3"):
        logger = slog.Logger(path)
        state = logger.load_last()
    else:
        state = torch.load(path)
    args, kw = state.get("margs", ([], {}))
    if "msrc" in state:
        warnings.warn("msrc used in load_model")
        src = state["msrc"]
        mmod = make_module(src)
        model = getattr(mmod, "make_model")(*args, **kw)
        model.msrc_ = src
    else:
        model = get_function(state["make_model"], *args, **kw)
        model.mpath_ = state["make_model"]
    model.margs_ = (args, kw)
    model.load_state_dict(state["mstate"])
    model.step_ = state.get("step", 0)
    return model


def load_or_construct_model(path, *args, module_path=["models", "experimental_models"], **kw):
    if os.path.splitext(path)[1] in ["py", "sqlite3"]:
        return load_model(path, *args, module_path=module_path, **kw)
    else:
        return construct_model(path, *args, module_path=module_path, **kw)


#
# Model Saving
#


def save_model_as_dict(model, fname, step=None):
    """Save a PyTorch model (parameters and source code) to a file."""
    if step is None:
        step = getattr(model, "step_", 0)
    state = dict(
        msrc=model.msrc_, margs=model.margs_, mstate=model.state_dict(), step=step
    )
    torch.save(state, fname)


def dump_model_as_dict(model):
    """Dump a model to a string using torch.save."""
    buf = io.BytesIO()
    save_model_as_dict(model, buf)
    return buf.getbuffer()


def log_model(logger, model, loss=None, step=None, optimizer=None):
    assert loss is not None
    assert step is not None
    state = dict(
        mdef="",
        msrc="",
        mstate=model.state_dict(),
        ostate=optimizer.state_dict() if optimizer is not None else None,
    )
    logger.save("model", state, scalar=loss, step=step)
    logger.flush()
