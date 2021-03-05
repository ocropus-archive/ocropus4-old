import numpy as np
from scipy import ndimage as ndi

import torch
import torch.functional as F
from .utils import safe_randint


def mrot(a):
    """Make a rotation matrix."""
    from math import sin, cos

    return np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])


def eye23(m):
    """2D to 3D matrix"""
    result = np.eye(3)
    result[:2, :2] = m
    return result


def off23(d):
    """2D to 3D offset"""
    result = np.zeros(3)
    result[:2] = d
    return result


def make_affine(src, dst):
    """Compute affine transformation from src to dst points."""
    assert len(dst) == len(src), (src, dst)
    assert len(dst) >= 4, (src, dst)
    assert len(dst[0]) == 2, (src, dst)
    assert len(dst[0]) == len(src[0]), (src, dst)
    dst0 = dst - np.mean(dst, 0)[None, :]
    src0 = src - np.mean(src, 0)[None, :]
    H = np.dot(dst0.T, src0)
    U, S, V = np.linalg.svd(H)
    m = np.dot(V.T, U)
    d = np.dot(m, np.mean(dst, 0)) - np.mean(src, 0)
    # print(d)
    return m, d


def apply_affine(image, size, md, **kw):
    """Apply an affine transformation to an image.

    This takes care of the ndim==2 and ndim==3 cases."""
    h, w = size
    m, d = md
    if image.ndim == 2:
        return ndi.affine_transform(image, m, offset=-d, output_shape=(h, w), **kw)
    elif image.ndim == 3:
        return ndi.affine_transform(
            image, eye23(m), offset=-off23(d), output_shape=(h, w, 3), **kw
        )


def get_affine_patch(image, size, coords, **kw):
    """Get patch of the given size from the given source coordinates.

    Keyword arguments are passed on to ndi.affine_transform."""
    h, w = size
    y0, y1, x0, x1 = coords
    src = [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
    dst = [(0, 0), (0, w), (h, w), (h, 0)]
    md = make_affine(src, dst)
    return apply_affine(image, size, md, **kw)


def get_affine_patches(dst, src, images, size=None):
    """Extracts patches from `images` under the affine transformation
    estimated by transforming the points in src to the points in dst."""
    if size is None:
        pts = np.array(dst, "i")
        size = np.amax(pts, axis=0)
        h, w = size
    m, d = make_affine(src, dst)
    result = []
    for image in images:
        patch = apply_affine(image, (h, w), (m, d), order=1)
        result.append(patch)
    return result


def get_patch(image, y0, y1, x0, x1, **kw):
    return ndi.affine_transform(image, np.eye(2), offset=(y0, x0), output_shape=(y1 - y0, x1 - x0), **kw)


def interesting_patches(
    indicator_image, threshold, images, r=256, n=50, trials=500, margin=0.1, jitter=5
):
    """
    Find patches that are "interesting" according to the indicator image; i.e., they need
    to include more than `threshold` values when summed over the patch.
        :param indicator_image: indicator image
        :param threshold: threshold for determining whether a patch is interesting
        :param images: list of images
        :param r=256: size of patch
        :param n=50: number of patches
        :param trials=500: number of trials
        :param margin=0.1: margin for the indicator image
        :param jitter=5: small deformation of source rectangle
    """
    from numpy.random import uniform

    h, w = indicator_image.shape[:2]
    count = 0
    for i in range(trials):
        if count >= n:
            break
        y = safe_randint(-r // 2, h - r//2 - 1)
        x = safe_randint(-r // 2, w - r//2 - 1)
        rx, ry = int(uniform(0.8, 1.2) * r), int(uniform(0.8, 1.2) * r)
        if margin < 1.0:
            dx, dy = int(rx * margin), int(ry * margin)
        else:
            dx, dy = int(margin), int(margin)
        patch = get_patch(indicator_image, y + dy , y + ry - dy, x + dx , x + rx - dx, order=0)
        if np.sum(patch) < threshold:
            continue
        rect = [y, x, y + ry, x + rx]
        rect = [c + safe_randint(-jitter, jitter) for c in rect]
        y0, x0, y1, x1 = rect
        src = [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
        dst = [(0, 0), (0, r), (r, r), (r, 0)]
        # print("*", src, dst)
        patches = get_affine_patches(dst, src, images)
        yield i, (x, y), patches
        count += 1


def overlapping_tiles(image, r=(32, 32), s=(512, 512)):
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h, s[0]):
        for x in range(0, w, s[1]):
            patch = ndi.affine_transform(
                image,
                np.eye(3),
                offset=(y - r[0], x - r[1], 0),
                output_shape=(s[0] + 2 * r[0], s[1] + 2 * r[1], image.shape[-1]),
                order=0,
            )
            patches.append(patch)
    return patches


def stuff_back(output, patches, r=(32, 32), s=(512, 512)):
    count = 0
    h, w = output.shape[:2]
    for y in range(0, h, s[0]):
        for x in range(0, w, s[1]):
            p = patches[count]
            count += 1
            ph, pw = min(s[0], h - y), min(s[1], w - x)
            output[y : y + ph, x : x + pw, ...] = p[
                r[0] : r[0] + ph, r[1] : r[1] + pw, ...
            ]
    return output


def full_inference(batch, model, *args, **kw):
    with torch.no_grad():
        result = model(batch).detach().softmax(1).cpu()
    return result


def patchwise_inference(image, model, patchsize=(512, 512), overlap=(64, 64)):
    patches = overlapping_tiles(image, r=overlap, s=patchsize)
    outputs = []
    for npatch in patches:
        patch = torch.tensor(npatch).permute(2, 0, 1).unsqueeze(0)
        probs = full_inference(patch, model)
        if probs.shape[-2:] != patch.shape[-2:]:
            print("# shape mismatch", probs.shape, patch.shape)
            probs = F.interpolate(probs, size=patch.shape[-2:])
        nprobs = probs[0].permute(1, 2, 0).numpy()
        outputs.append(nprobs)
    result = np.zeros(image.shape[:2] + (outputs[0].shape[-1],))
    stuff_back(result, outputs, r=overlap, s=patchsize)
    return result
