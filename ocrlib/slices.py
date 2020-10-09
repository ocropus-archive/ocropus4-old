import numpy as np
import scipy.ndimage as ndi


def slpad(sl, r):
    """Pad an array slice."""
    if isinstance(r, int):
        return slice(max(sl.start - r, 0), sl.stop + r)
    elif isinstance(r, float):
        d = int((0.5 + sl.stop - sl.start) * r)
        return slice(max(sl.start - d, 0), sl.stop + d)
    else:
        raise ValueError(f"range {r}")


def slspad(sls, rs):
    """Pad an array slice."""
    if not isinstance(rs, list):
        rs = [rs] * len(sls)
    return tuple(slpad(x, r) for x, r in zip(sls, rs))


def sl2bbox(sl):
    """Convert a 2D array slice to a bounding box (x0, y0, x1, y1)."""
    ys, xs = sl[:2]
    x0, y0, x1, y1 = xs.start, ys.start, xs.stop, ys.stop
    return (x0, y0, x1, y1)


def padded_slices_for_regions(markers, pad=5):
    """Returns an iterator over the bounding boxes of the marked regions, with padding.

    The result is suitable for subscripting:

        for region in extract_regions(...):
            image[region]
            sl2bbox(region)
    """
    for i, sl in enumerate(ndi.find_objects(markers)):
        region = slspad(sl, pad)
        yield region


def pad_slice(sl, r):
    """Pad a slice by #pixels (r>=1) or proportionally (0<r<1)."""
    return slice(max(sl.start - r, 0), sl.stop + r)


def dilate_slice(sl, r):
    """Dilate a slice by `r` around its center."""
    center = (sl.start + sl.stop) / 2.0
    width = sl.stop - sl.start
    return slice(int(max(center - width * r / 2.0, 0)), int(center + width * r / 2.0))


def intersect_slices(s1, s2):
    lo = max(s1.start, s2.start)
    hi = min(s1.stop, s2.stop)
    return slice(lo, max(hi, lo))


def intersect_boxes(b1, b2):
    return [intersect_slices(s1, s2) for s1, s2 in zip(b1, b2)]


def box_area(box):
    return np.prod([s.stop - s.start for s in box])


def plot_box(b, **kw):
    import matplotlib.pyplot as plt
    ys, xs = b[:2]
    y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], **kw)


def plot_boxes(boxes, **kw):
    for b in boxes:
        plot_box(b, **kw)
