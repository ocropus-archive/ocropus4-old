import random
import random as pyr
import warnings

import numpy as np
import pylab
import scipy.ndimage as ndi


def normalize(image, lo=0.005, hi=0.995):
    lo, hi = np.percentile(image.flat, lo * 100), np.percentile(image.flat, hi * 100)
    return np.clip((image - lo) / max(hi - lo, 0.1), 0.0, 1.0)


def autoinvert(image):
    assert np.amin(image) >= 0
    assert np.amax(image) <= 1
    if sum(image > 0.9) > sum(image < 0.1):
        return 1 - image
    else:
        return image


def fgbg(selector, fg, bg, check=False):
    lo, hi = np.amin(selector), np.amax(selector)
    assert selector.shape == fg.shape
    assert selector.shape == bg.shape
    if check:
        assert lo >= 0 and lo <= 0.1
        assert hi >= 0.9 and hi <= 1.0
    return selector * fg + (1.0 - selector) * bg


def random_transform(translation=(-0.01, 0.01), rotation=(-2, 2), scale=(-0.1, 0.0), aniso=(-0.1, 0.1)):
    """Generate a random affine transform. Return a dict."""
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    angle = pyr.uniform(*rotation)
    angle = angle * np.pi / 180.0
    scale = 10 ** pyr.uniform(*scale)
    aniso = 10 ** pyr.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy))


def transform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    """Transform an image with a random set of transformations.

    Output is same size as input."""
    dx, dy = translation
    scale = 1.0 / scale
    c = np.cos(angle)
    s = np.sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], "f")
    m = np.array([[c, -s], [s, c]], "f")
    m = np.dot(sm, m)
    w, h = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])
    return ndi.affine_transform(image, m, offset=d, order=order, mode="constant", output=image.dtype)


def transform_all(*args, order=1, **kw):
    """Perform the same random transformation to all images.

    Output is same size as input."""
    if not isinstance(order, list):
        order = [order] * len(args)
    t = random_transform(**kw)
    return tuple(transform_image(x, order=o, **t) for x, o in zip(args, order))


def xtransform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    """Transform an image with a random set of transformations.

    Uses individual transforms and grows/shrinks image as necessary."""
    if angle != 0.0:
        image = ndi.rotate(image, angle, order=1, mode="constant")
    if scale != 1.0 or aniso != 1.0:
        image = ndi.zoom(image, (scale*aniso, scale/aniso), order=1, mode="constant")
    return image


def xtransform_all(*args, order=1, **kw):
    """Perform the same random transformation to all images.

    Output is same size as input."""
    if not isinstance(order, list):
        order = [order] * len(args)
    t = random_transform(**kw)
    return tuple(xtransform_image(x, order=o, **t) for x, o in zip(args, order))


def bounded_gaussian_noise(shape, sigma, maxdelta):
    """Generate gaussian smoothed gaussian noise vectors with a maximum delta."""
    n, m = shape
    deltas = pylab.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2 * deltas - 1) * maxdelta
    return deltas


def distort_with_noise(image, deltas, order=1):
    """Given a (2, n, m) displacement vector field, distort the image with it."""
    assert deltas.shape[0] == 2
    assert image.shape == deltas.shape[1:], (image.shape, deltas.shape)
    n, m = image.shape
    xy = np.transpose(np.array(np.meshgrid(range(n), range(m))), axes=[0, 2, 1])
    deltas += xy
    return ndi.map_coordinates(image, deltas, order=order, mode="reflect")


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = ndi.gaussian_filter(pylab.randn(w), sigma)
    noise *= magnitude / np.amax(abs(noise))
    dys = np.array([noise] * h)
    deltas = np.array([dys, np.zeros((h, w))])
    return deltas


def distort_all(*args, sigma=(0.5, 3.0), maxdelta=(0.1, 2.0), order=1):
    if isinstance(sigma, tuple):
        sigma = random.uniform(*sigma)
    if isinstance(maxdelta, tuple):
        maxdelta = random.uniform(*maxdelta)
    if not isinstance(order, list):
        order = [order] * len(args)
    noise = bounded_gaussian_noise(args[0].shape, sigma, maxdelta)
    return tuple(distort_with_noise(x, noise, order=o) for x, o in zip(args, order))


def percent_black(image):
    """Return the percentage of black in the image."""
    n = np.prod(image.shape)
    k = np.sum(image < 0.5)
    return k * 100.0 / n


def binary_blur(image, sigma, noise=0.0):
    """Blur a binary image, preserving the number of fg/bg pixels."""
    p = percent_black(image)
    blurred = ndi.gaussian_filter(image, sigma)
    if noise > 0:
        blurred += pylab.randn(*blurred.shape) * noise
    t = np.percentile(blurred, p)
    return np.array(blurred > t, "f")


def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h / scale + 1), int(w / scale + 1)
    data = pylab.rand(h0, w0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ndi.zoom(data, scale, order=2)
    result = result - np.amin(result)
    result /= max(np.amax(result), 0.001)
    return np.clip(result[:h, :w], 0, 1)


def noisify(image, sigma=(0.0, 2.0), amp1=0.05, sigma1=(0.5, 2.0), amp2=0.2, sigma2=(20.0, 40.0)):
    if isinstance(sigma, tuple):
        sigma = random.uniform(*sigma)
    if isinstance(sigma1, tuple):
        sigma1 = random.uniform(*sigma1)
    if isinstance(sigma2, tuple):
        sigma2 = random.uniform(*sigma2)
    if isinstance(amp1, tuple):
        amp1 = random.uniform(*amp1)
    if isinstance(amp2, tuple):
        amp2 = random.uniform(*amp2)
    image = image - np.amin(image)
    image /= max(np.amax(image), 0.001)
    image += amp1 * make_noise_at_scale(image.shape, sigma1)
    image += amp2 * make_noise_at_scale(image.shape, sigma2)
    image = image - np.amin(image)
    image /= max(np.amax(image), 0.001)
    return image

