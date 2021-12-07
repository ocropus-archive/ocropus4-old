import os
import os.path
import sys
import webdataset.gopen as gopen_mod
import urllib.parse
import re
import time

cachedir = os.environ.get("OCROCACHE", os.path.join(os.environ.get("HOME", "/tmp"), ".ocropus4/cache"))
modeldir = os.environ.get("OCROMODELS", os.path.join(os.environ.get("HOME", "/tmp"), ".ocropus4/models"))

def cached_gopen(url, mode="rb", cachedir=cachedir, verbose=False, maxage=1e33, **kw):
    key = re.sub("/", "%2F", urllib.parse.quote(url))
    os.makedirs(cachedir, 0o755, exist_ok=True)
    path = os.path.join(cachedir, key)
    temppath = f"{path}.~{os.getpid()}~"
    if not os.path.exists(path) or os.path.getmtime(path) + maxage < time.time():
        try:
            if verbose:
                print(f"{url}: downloading", file=sys.stderr)
            with open(temppath, "wb") as sink:
                with gopen_mod.gopen(url, mode=mode, **kw) as source:
                    while True:
                        data = source.read(1000000)
                        if len(data) == 0:
                            break
                        sink.write(data)
            os.rename(temppath, path)
            if verbose:
                print(f"{url}: done", file=sys.stderr)
        finally:
            try:
                os.unlink(temppath)
            except:
                pass
    if verbose:
        print(f"{path}: opening", file=sys.stderr)
    return open(path, "rb")


def model_gopen(url, mode="rb", modeldir=modeldir, verbose=False, maxage=1e33, **kw):
    return cached_gopen(url, mode=mode, cachedir=modeldir, verbose=verbose, maxage=1e33, **kw)
