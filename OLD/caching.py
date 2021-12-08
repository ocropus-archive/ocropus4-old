import io
import os
import os.path
import re
import sys
import urllib.parse

import webdataset.gopen as gopen

cachedir = os.path.join(os.environ.get("HOME", "/tmp"), ".datacache")


class CachedReader(io.IOBase):
    def __init__(self, stream, cache, on_close=None):
        self.stream = stream
        self.cache = cache
        self.on_close = on_close

    def close(self):
        self.cache.close()
        if self.on_close is not None:
            self.on_close()
        self.stream = None
        self.cache = None
        return None

    def read(self, size=-1):
        data = self.stream.read(size)
        # print("read", None if data is None else len(data))
        if data is None:
            return self.close()
        self.cache.write(data)
        return data

    def readall(self):
        data = self.stream.readall()
        if data is None:
            return self.close()
        self.cache.write(data)
        return data

    def readinto(self, b):
        n = self.stream.readinto(b)
        if n is None:
            return self.close()
        self.cache.write(b[:n])
        return n


def cached_gopen(url, mode="rb", cachedir=cachedir, verbose=False, **kw):
    key = re.sub("/", "%2F", urllib.parse.quote(url))
    os.makedirs(cachedir, 0o755, exist_ok=True)
    path = os.path.join(cachedir, key)
    if os.path.exists(path):
        if verbose:
            print(f"[{url}: using cache]", file=sys.stderr)
        return open(path, "rb")
    else:
        if verbose:
            print(f"[opening {url}]", file=sys.stderr)
    stream = gopen.gopen(url, mode=mode, **kw)
    cache = open(path + ".TEMP", "wb")

    def closer():
        cache.close()
        os.rename(path + ".TEMP", path)

    return CachedReader(stream, cache, on_close=closer)


def cached_open(url, mode="rb", cachedir=cachedir, verbose=False, **kw):
    with cached_gopen(
        url, mode=mode, cachedir=cachedir, verbose=verbose, **kw
    ) as stream:
        while stream.read(1000000) is not None:
            pass
    return cached_gopen(url, mode=mode, cachedir=cachedir, verbose=verbose, **kw)
