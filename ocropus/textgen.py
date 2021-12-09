import os, os.path, random, re, sys
from typing import List

import numpy as np
import typer
import webdataset as wds
from PIL import Image, ImageDraw, ImageFont

app = typer.Typer()


def add_margin(
    pil_img: PIL.Image.Image, top: int, right: int, bottom: int, left: int, color: Any
):
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


def generate_text(words: List[str]):
    """Generate text from a list of words."""
    case = random.randint(0, 3)
    specials = "-:/,.$%@&*^`~!?()[]{}_"
    specials = [c for c in specials]
    if case <= 0:
        prefix = random.choice([""] * 10 + specials)
        w = random.choice(words)
        sp = random.choice([""] * 3 + [" "])
        suffix = random.choice([""] * 5 + specials)
        return prefix + w + sp + suffix
    elif case <= 1:
        w1 = random.choice(words)
        sp1 = random.choice([""] * 3 + [" "])
        sep = random.choice([" "] * 5 + specials)
        sp2 = random.choice([""] * 3 + [" "])
        w2 = random.choice(words)
        result = w1 + sp1 + sep + sp2 + w2
        result = re.sub(" +", " ", result)
        return result
    elif case <= 2:
        value = 10 ** random.uniform(0.0, 5.0) * np.sign(random.uniform(-1.0, 1.0))
        case = random.randint(0, 6)
        if case == 0:
            return str(value)
        if case == 1:
            return str(int(value))
        elif case == 2:
            return "%.1f" % value
        elif case == 3:
            return "%.1f" % value
        elif case == 4:
            return "%.2e" % value
        elif case == 5:
            return "%.3e" % value
        elif case == 6:
            return "%08d" % int(value)
    elif case <= 3:
        l = random.randint(3, 10)
        s = "".join([chr(random.randint(33, 126)) for _ in range(l)])
        return s


@app.command()
def generate(
    output: str = "generated-%06d.tar",
    fontlist: str = "",
    wordlist: str = "/usr/share/dict/words",
    sizes: str = "10, 80",
    shardsize: int = 10000,
    nwords: int = 1000000,
):
    """Generate a dataset of printed text."""
    words = [s.strip() for s in open(wordlist).readlines()]
    print(f"got {len(words)} words")
    if fontlist != "":
        fonts = [s.strip() for s in open(fontlist).readlines()]
        fonts = [s for s in fonts if s[0] != "#"]
        for f in fonts:
            assert os.path.exists(f)
        print(f"got {len(fonts)} fonts")
    else:
        fonts = ["/usr/share/fonts/truetype/msttcorefonts/arial.ttf"]
    print(f"got {len(fonts)} fonts")
    sizes = eval(f"({sizes})")
    sink = wds.ShardWriter(output, maxcount=shardsize)
    iw, ih = 1024, 1024
    for i in range(nwords):
        word = generate_text(words)
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
        if image.width < 40 or image.height < 10:
            continue
        if image.width > 400 or image.height > 100:
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


if __name__ == "__main__":
    app()
