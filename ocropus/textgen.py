import os, os.path, random, re, sys
from typing import List, Any
import glob

import numpy as np
import typer
import webdataset as wds
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
import ray

app = typer.Typer()


def add_margin(pil_img: PIL.Image.Image, top: int, right: int, bottom: int, left: int, color: Any):
    """Add a margin to an image.

    Args:
        pil_img (PIL.Image.Image): input image
        top (int): margin
        right (int): margin
        bottom (int): margin
        left (int): margin
        color (Any): PIL color spec

    Returns:
        PIL: image with margin
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def generate_words(words):
    def f():
        return random.choice(words)

    return f


def generate_text(words: List[str]):
    def f():
        nwords = random.randint(1, 5)
        result = []
        specials = [chr(c) for c in range(33, 127)]
        specials = [c for c in specials if not c.isalpha()]
        for i in range(nwords):
            word = random.choice(words)
            result.append(word)
            if random.uniform(0, 1) < 0.2:
                result.append(random.choice(specials))
                if random.uniform(0, 1) < 0.9:
                    result.append(" ")
            else:
                result.append(" ")
        result = "".join(result)
        result = result.strip()
        return result

    return f


def generate_ascii(_):
    def f():
        l = random.randint(3, 10)
        s = "".join([chr(random.randint(33, 126)) for _ in range(l)])
        return s
    return f


def generate_numbers(_):
    def f():
        value = 10 ** random.uniform(0.0, 5.0) * np.sign(random.uniform(-1.0, 1.0))
        v = random.uniform(0, 1)
        if v < 0.3:
            value = int(value)
            total = random.randint(1, 10)
            result = "%*d" % (total, value)
            return result
        elif v < 0.6:
            mant = random.randint(0, 7)
            total = max(mant + 2, random.randint(0, 10))
            result = "%*.*e" % (total, mant, value)
            return result
        else:
            mant = random.randint(0, 7)
            total = max(mant + 2, random.randint(0, 10))
            result = "%*.*f" % (total, mant, value)
            return result
    return f

def read_dict(wordlist: str) -> List[str]:
    return [s.strip() for s in open(wordlist).readlines()]


@app.command()
def clone_google_fonts():
    os.system("git clone https://github.com/google/fonts.git google-fonts")


google_exclude_fonts = """\
creepstercaps fontdinerswanky homemadeapple jsmath justanotherhand
adobeblank bungee ewert fasterone geostar barcode notosans
notoserif portersansblock redacted sixcaps stalemate stalinistone warnes
zillaslab rocksalt ballet flowblock monoton rock3d""".split()


def font_is_excluded(fname):
    for s in google_exclude_fonts:
        if s in fname:
            return True
    return False


def get_google_fonts():
    assert os.path.exists("./google-fonts")
    fonts = [s.strip() for s in os.popen("find ./google-fonts -name '*.ttf'").readlines()]
    assert len(fonts) > 0
    fonts = [f for f in fonts if not font_is_excluded(f)]
    assert len(fonts) > 0
    return fonts


@app.command()
def generate(
    output: str = "generated-%06d.tar",
    generator: str = "text",
    fontlist: str = "core",
    wordlist: str = "/usr/share/dict/words",
    sizes: str = "20, 80",
    shardsize: int = 2000,
    nwords: int = 100000,
):
    words = read_dict(wordlist)
    print(f"got {len(words)} words")

    # factory = globals()["generate_" + generator]
    if generator == "text":
        generator = generate_text(words)
    elif generator == "words":
        generator = generate_words(words)
    elif generator == "ascii":
        generator = generate_ascii(words)
    elif generator == "numbers":
        generator = generate_numbers(words)
    else:
        raise ValueError(f"unknown generator {generator}")

    if fontlist == "google":
        fonts = get_google_fonts()
        print(f"got {len(fonts)} fonts from ./google-fonts")
    elif fontlist == "italics":
        fonts = get_google_fonts()
        fonts = [f for f in fonts if "italic" in f.lower()]
        print(f"got {len(fonts)} italic fonts from ./google-fonts")
    elif fontlist == "core":
        fonts = glob.glob("/usr/share/fonts/truetype/msttcorefonts/[A-Z]*.ttf")
        fonts += glob.glob("./google-fonts/ofl/ebgaramond/*.ttf")
        fonts += glob.glob("./google-fonts/ofl/cormorantgaramond/*.ttf")
        fonts += glob.glob("./google-fonts/ofl/librecaslontext/*.ttf")
        fonts += glob.glob("./google-fonts/ofl/librecaslondisplay/*.ttf")
        fonts = [s for s in fonts if "webdings" not in s.lower()]
        assert len(fonts) > 0
        print(f"got {len(fonts)} core fonts")
    elif fontlist == "ms":
        fonts = glob.glob("/usr/share/fonts/truetype/msttcorefonts/[A-Z]*.ttf")
        fonts = [s for s in fonts if "webdings" not in s.lower()]
        print(f"got {len(fonts)} ms fonts")
        assert len(fonts) > 0
    else:
        fonts = [s.strip() for s in open(fontlist).readlines()]
        fonts = [s for s in fonts if s[0] != "#"]
        for f in fonts:
            assert os.path.exists(f)
        print(f"got {len(fonts)} fonts")
    assert len(fonts) > 0

    sizes = eval(f"({sizes})")
    if "%" in output:
        sink = wds.ShardWriter(output, maxcount=shardsize)
    else:
        sink = wds.TarWriter(output)
        print(f"writing to {output}")
    iw, ih = 1024, 1024
    for i in range(nwords):
        word = generator()
        fontname = random.choice(fonts)
        size = int(np.exp(random.uniform(np.log(sizes[0]), np.log(sizes[1]))))
        try:
            image = Image.new("RGB", (ih, iw), color="black")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fontname, size)
            draw.text((20, ih // 2), word, font=font)
            bbox = image.getbbox()
            image = image.crop(bbox)
        except Exception as exn:
            print("error during image generation:", repr(exn)[:200])
            print("parameters:", (iw, ih), fontname, size)
            continue
        if image.width < 20 or image.height < 20:
            continue
        if image.width > 1000 or image.height > 100:
            continue
        m = [random.randint(3, 30) for _ in range(4)]
        image = add_margin(image, m[0], m[1], m[2], m[3], "black")
        image = np.array(image)
        sample = {
            "__key__": f"{i:08d}",
            "json": dict(size=size, font=fontname, word=word),
            "txt": word,
            "jpg": image,
        }
        sink.write(sample)
        if i % 100 == 0:
            print(i, end=" ", flush=True, file=sys.stderr)
    sink.close()
    return 0


@ray.remote
def generate_(*args, **kw):
    return generate(*args, **kw)

@app.command()
def all():
    nw = 100
    nshards = 1
    ray.init()
    ray.get([generate_.remote(output=f"_core-words-{i:06d}.tar", nwords=nw, fontlist="core", generator="words") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_core-text-{i:06d}.tar", nwords=nw, fontlist="core") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_google-text-{i:06d}.tar", nwords=nw, fontlist="google") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_italics-text-{i:06d}.tar", nwords=nw, fontlist="italics") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_core-numbers-{i:06d}.tar", nwords=nw, fontlist="core", generator="numbers") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_google-numbers-{i:06d}.tar", nwords=nw, fontlist="google", generator="numbers") for i in range(nshards)])
    ray.get([generate_.remote(output=f"_core-ascii-{i:06d}.tar", nwords=nw, fontlist="core", generator="ascii") for i in range(nshards)])

if __name__ == "__main__":
    app()
