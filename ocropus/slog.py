import os
import typer
import sys
import io
import pickle
import sqlite3
import time
import json as jsonlib
import importlib.machinery
import importlib.util
import tempfile
from typing import List
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import torch

from . import loading

app = typer.Typer()

log_schema = """
create table if not exists log (
    step real,
    logtime real,
    key text,
    msg text,
    scalar real,
    json text,
    obj blob
)
"""

log_verbose = int(os.environ.get("LOG_VERBOSE", 0))


def load_module(mname, text):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as stream:
        stream.write(text)
        stream.flush()
        loader = importlib.machinery.SourceFileLoader(mname, stream.name)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod


def torch_dumps(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer()


def torch_loads(buf):
    return torch.load(io.BytesIO(buf))


class NoLogger:
    def __getattr__(self, name):
        def noop(*args, **kw):
            pass

        return noop


class Logger:
    def __init__(self, fname=None, prefix="", sysinfo=True):
        if fname is None or fname == "":
            import datetime

            fname = prefix + "-"
            fname += datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            fname += ".sqlite3"
            print(f"log is {fname}", file=sys.stderr)
        self.wandb = None
        self.fname = fname
        self.con = sqlite3.connect(fname)
        self.con.execute(log_schema)
        self.last = 0
        self.interval = 10
        if "WANDB" in os.environ:
            import wandb
            wandb.init(project=os.environ["WANDB"])
            self.wandb = wandb
        if sysinfo:
            self.sysinfo()

    def maybe_commit(self):
        if time.time() - self.last < self.interval:
            return
        for i in range(10):
            try:
                self.commit()
                break
            except sqlite3.OperationalError as exn:
                print("ERROR:", exn, file=sys.stderr)
                time.sleep(1.0)
        self.commit()
        self.last = time.time()

    def commit(self):
        self.con.commit()

    def flush(self):
        self.commit()

    def close(self):
        self.con.commit()
        self.con.close()
        if self.wandb is not None:
            self.wandb.finish()
            self.wandb = None

    def raw(
        self, key, step=None, msg=None, scalar=None, json=None, obj=None, walltime=None
    ):
        if log_verbose:
            print("#LOG#", key, step, msg, scalar, json, type(obj), file=sys.stderr)
        if msg is not None:
            assert isinstance(msg, (str, bytes)), msg
        if step is not None:
            step = float(step)
        if scalar is not None:
            scalar = float(scalar)
        if json is not None:
            assert isinstance(json, (str, bytes)), json
        # if obj is not None:
        #     assert isinstance(obj, bytes), obj
        if walltime is None:
            walltime = time.time()
        self.con.execute(
            "insert into log (logtime, step, key, msg, scalar, json, obj) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (walltime, step, key, msg, scalar, json, obj),
        )
        self.maybe_commit()

    def insert(
        self,
        key,
        step=None,
        msg=None,
        scalar=None,
        json=None,
        obj=None,
        dumps=pickle.dumps,
        walltime=None,
    ):
        if json is not None:
            json = jsonlib.dumps(json)
        if obj is not None:
            obj = dumps(obj)
        self.raw(
            key,
            step=step,
            msg=msg,
            scalar=scalar,
            json=json,
            obj=obj,
            walltime=walltime,
        )

    def scalar(self, key, scalar, step=None, **kw):
        self.insert(key, scalar=scalar, step=step, **kw)
        if self.wandb is not None:
            self.wandb.log({key: scalar}, step=step)

    def message(self, key, msg, step=None, **kw):
        self.insert(key, msg=msg, step=step, **kw)

    def json(self, key, json, step=None, **kw):
        self.insert(key, json=json, step=step, **kw)
        if self.wandb is not None and isinstance(json, dict):
            self.wandb.log(json, step=step)

    def save(self, key, obj, step=None, **kw):
        self.insert(key, obj=obj, dumps=torch_dumps, step=step, **kw)

    def save_model(self, obj, key="model", step=None, **kw):
        self.insert(key, obj=obj, dumps=torch_dumps, step=step, **kw)
        self.flush()
        if self.wandb is not None:
            with tempfile.NamedTemporaryFile(suffix=".pth") as stream:
                torch.save(obj, stream)
                stream.flush()
                self.wandb.save(stream.name)

    def save_dict(self, key="model", step=None, **kw):
        self.insert(key, obj=kw, dumps=torch_dumps, step=step, **kw)
        self.flush()

    def save_smodel(self, obj, key="model", step=None, **kw):
        self.insert(key, obj=obj, dumps=loading.dump_model_as_dict, step=step, **kw)
        self.flush()

    def load_last(self, key="model"):
        obj, step = self.con.execute(
            f"select obj, step from log where key='{key}' order by logtime desc limit 1"
        ).fetchone()
        state = torch_loads(obj)
        state["step"] = step
        return state

    def load_best(self, key="model"):
        obj, step = self.con.execute(
            f"select obj, step from log where key='{key}' order by scalar limit 1"
        ).fetchone()
        state = torch_loads(obj)
        state["step"] = step
        return state

    def sysinfo(self):
        cmd = "hostname; uname"
        cmd += "; lsb_release -a"
        cmd += "; cat /proc/meminfo; cat /proc/cpuinfo"
        cmd += "; nvidia-smi -L"
        with os.popen(cmd) as stream:
            info = stream.read()
        self.message("__sysinfo__", info)

    # The following methods are for compatibility with Tensorboard

    def add_hparams(self, hparam_dict=None, metric_dict=None):
        if hparam_dict is not None:
            self.json("__hparams__", hparam_dict)
        if metric_dict is not None:
            self.json("__metrics__", metric_dict)

    def add_image(self, tag, obj, step=-1, walltime=None):
        # FIXME: convert to PNG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_figure(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to PNG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_video(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to MJPEG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_audio(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to FLAC
        self.save(tag, obj, step=step, walltime=walltime)

    def add_text(self, tag, obj, step=-1, bins=None, walltime=None):
        self.message(tag, obj, step=step, walltime=walltime)

    def add_embedding(self, tag, obj, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_graph(self, tag, obj, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_scalar(self, tag, value, step=-1, walltime=None):
        self.scalar(tag, value, step=step, walltime=walltime)

    def add_scalars(self, tag, value, step=-1, bins=None, walltime=None):
        for k, v in value.items():
            self.add_scalar(f"{tag}/{k}", v, step=step, walltime=walltime)

    def add_histogram(self, tag, values, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        raise Exception("unimplemented")

    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ):
        raise Exception("unimplemented")


@app.command()
def getargs(fname):
    """Get the arguments used for training from a logfile."""
    con = sqlite3.connect(fname)
    query = "select json from log where key='args'"
    result = list(con.execute(query))
    assert len(result) > 0
    args = jsonlib.loads(result[0][0])
    for k, v in args.items():
        print(k, repr(v)[:60])


@app.command()
def getmodel(fname):
    """Get the model definition used for training from a logfile."""
    con = sqlite3.connect(fname)
    query = "select obj from log where key='model'"
    result = list(con.execute(query))
    assert len(result) > 0, "found no model"
    mbuf = result[0][0]
    obj = torch_loads(mbuf)
    print(obj.get("msrc"))


@app.command()
def getstep(fname, step: int, output=None):
    """Get the model at the given step."""
    assert output is not None
    assert not os.path.exists(output)
    con = sqlite3.connect(fname)
    query = f"select obj, step, scalar from log where key='model' order by abs({step}-step) limit 1"
    result = list(con.execute(query))
    assert len(result) > 0, "found no model"
    mbuf = result[0][0]
    print(f"saving step {result[0][1]} scalar {result[0][2]}")
    with open(output, "wb") as stream:
        stream.write(mbuf)


def gettrained(fname, query, output=None):
    con = sqlite3.connect(fname)
    result = list(con.execute(query))
    assert len(result) > 0, "found no model"
    mbuf = result[0][0]
    print(f"saving step {result[0][1]} scalar {result[0][2]}")
    if output is None:
        model = loading.dict_to_model(loading.torch_loads(mbuf))
        print(model)
    else:
        assert not os.path.exists(output)
        with open(output, "wb") as stream:
            stream.write(mbuf)


@app.command()
def getbest(fname, output=None):
    """Get the model associated with the smallest scalar value (usu. test error)."""
    query = (
        "select obj, step, scalar from log where key='model' order by scalar limit 1"
    )
    return gettrained(fname, query, output=output)


@app.command()
def getlast(fname, output=None):
    """Get the last model from a log."""
    query = "select obj, step, scalar from log where key='model' order by logtime desc limit 1"
    return gettrained(fname, query, output=output)


@app.command()
def val2model(fname):
    """Move the validation values into the scalar field for the corresponding model."""
    con = sqlite3.connect(fname)
    print(list(con.execute("select step, scalar from log where key='model'")))
    query = "select step, scalar from log where key='val/err'"
    errs = [tuple(row) for row in con.execute(query)]
    print(errs)
    for step, scalar in errs:
        query = f"update log set scalar = {scalar} where key = 'model' and step = {int(step)}"
        print(query)
        con.execute(query)
        con.commit()
    con.close()


@app.command()
def schema(fname):
    con = sqlite3.connect(fname)
    print(list(con.execute("select sql from sqlite_master where name = 'log'"))[0][0])
    print(list(con.execute("select distinct(key) from log")))


@app.command()
def show(fname, keys="model", xscale="linear", yscale="linear"):
    plt.title(fname)
    con = sqlite3.connect(fname)
    plt.xscale(xscale)
    plt.yscale(yscale)
    for k in keys.split(","):
        values = list(con.execute(f"select step, scalar from log where key = '{k}'"))
        xs, ys = zip(*values)
        plt.ylim(np.amin(ys), min(np.amax(ys), 6*np.median(ys)))
        plt.plot(xs, ys, label=k)
    plt.legend()
    plt.show()


@app.command()
def showdump(fname):
    dump = torch.load(fname)
    if not isinstance(dump, dict):
        print(repr(dump)[:1000])
        return
    for k, v in dump.items():
        print(f"{k:20s} {repr(v)[:60]}")


@app.command()
def info(fnames: List[str], verbose: bool = False):
    """Print information on each of the log files."""
    fnames = sorted(fnames)
    result = []
    for fname in fnames:
        try:
            con = sqlite3.connect(fname)
            args = list(con.execute("select json from log where key='args'"))
            if len(args) == 0:
                result.append((fname, None, None, None, None, None))
                continue
            json = jsonlib.loads(args[0][0])
            models = list(
                con.execute(
                    "select step, scalar from log where key='model' order by scalar"
                )
            )
            maxsteps = list(
                con.execute("select step from log order by step desc limit 1")
            )
            if len(maxsteps) == 0:
                continue
            maxstep = maxsteps[0][0]
            if verbose:
                print("nmodels", len(models), "maxstep", maxstep)
            step, scalar = (None, None) if len(models) == 0 else models[0]
            if maxstep is not None:
                maxstep = int(maxstep)
            if step is not None:
                step = int(step)
            if scalar is None:
                step = None
            mdef = json.get("mdef") if isinstance(json, dict) else None
            result.append((fname, mdef, maxstep, len(models), step, scalar))
            con.close()
            del con
        except sqlite3.DatabaseError as exn:
            print(fname, ": database error")
            print(exn)
            sys.exit(1)
        except ValueError:
            result.append((fname, None, None, None, None, None))
    headers = "file model steps #saves best_step best_cost".split()
    print(tabulate(result, headers=headers))


@app.command()
def findempty(fnames: List[str], n=0):
    """Find all files that don't contain a saved model.

    This is used for deleting partial logs.
    """
    fnames = sorted(fnames)
    for fname in fnames:
        try:
            con = sqlite3.connect(fname)
            models = list(
                con.execute(
                    "select step, scalar from log where key='model' order by scalar"
                )
            )
            if len(models) <= 0:
                print(fname)
        except Exception as exn:
            print(f"ERROR: {fname} {repr(exn)[:30]}", file=sys.stderr)


@app.command()
def steps(fnames: List[str], clean:int=-1):
    """Find all files that don't contain a saved model.

    This is used for deleting partial logs.
    """
    fnames = sorted(fnames)
    for fname in fnames:
        try:
            con = sqlite3.connect(fname)
            result = list( con.execute("select max(step) from log"))
            value = result[0][0]
            value = int(value) if isinstance(value, float) else 0
            if value < clean:
                print(f"{value:12d} {fname} REMOVING")
                os.unlink(fname)
            else:
                print(f"{value:12d} {fname}")
        except Exception as exn:
            print(f"ERROR: {fname} {repr(exn)[:30]}", file=sys.stderr)


if __name__ == "__main__":
    app()
