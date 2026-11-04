"""
Mean skin colour of a region, mirroring the engine's ``meanSkinRgb()``.

This is a direct port of the skin sampler in
``app/src/main/cpp/rppg_core.cpp``, constant for constant. It has to be: the
detector compares a face region against a neck region and against the numbers
the phone reports for the same person, so if the laptop classified skin pixels
differently from the handset, every comparison downstream would be measuring
the difference between two samplers rather than between two regions of a body.

The phone reads YUV_420_888 planes straight from the camera. OpenCV gives BGR,
so here the frame is converted once to YCrCb per frame and every region is
sampled from that — the same colour space, the same full-range chroma, the
same thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RegionSample", "to_ycrcb", "sample_region"]

# ── Constants, copied from rppg_core.cpp ────────────────────────────────
# Classic YCbCr skin box (full-range chroma) and luma limits.
CR_MIN, CR_MAX = 133, 173
CB_MIN, CB_MAX = 77, 127
LUMA_MIN = 30
LUMA_GLARE = 235
MIN_SKIN_FRACTION = 0.25


@dataclass(frozen=True)
class RegionSample:
    """One region's mean colour for one frame."""

    valid: bool
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    luma: float = 0.0
    skin_fraction: float = 0.0
    pixels: int = 0


def to_ycrcb(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Converts one BGR frame to full-range YCrCb.

    Done once per frame rather than once per region: with six regions that is
    six times less colour conversion, which matters when the loop has to keep
    up with the camera to avoid gaps in the pulse signal.
    """
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)


def sample_region(
    ycrcb: np.ndarray,
    rect: tuple[int, int, int, int],
    step: int = 2,
) -> RegionSample:
    """
    Mean skin RGB inside ``rect``.

    Args:
        ycrcb: the whole frame in YCrCb, from :func:`to_ycrcb`.
        rect: ``(x, y, w, h)`` in image coordinates.
        step: sample every Nth pixel in each direction, as the engine does.
            The default of 2 matches the phone and cuts the work fourfold
            without measurably changing the mean.

    Returns:
        A :class:`RegionSample`. ``valid`` is False when the rectangle falls
        outside the frame or every pixel was rejected as glare.
    """
    x, y, w, h = rect
    height, width = ycrcb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return RegionSample(valid=False)

    step = max(1, int(step))
    patch = ycrcb[y0:y1:step, x0:x1:step]
    if patch.size == 0:
        return RegionSample(valid=False)

    luma = patch[:, :, 0].astype(np.int16)
    cr = patch[:, :, 1].astype(np.int16)
    cb = patch[:, :, 2].astype(np.int16)

    sampled = luma.size
    # Specular glare carries no pulse and saturates the mean: drop it first,
    # exactly as the engine does, before anything else is counted.
    not_glare = luma < LUMA_GLARE
    all_n = int(np.count_nonzero(not_glare))
    if sampled == 0 or all_n == 0:
        return RegionSample(valid=False)

    is_skin = (
        not_glare
        & (luma >= LUMA_MIN)
        & (cr >= CR_MIN)
        & (cr <= CR_MAX)
        & (cb >= CB_MIN)
        & (cb <= CB_MAX)
    )
    skin_n = int(np.count_nonzero(is_skin))

    # The engine divides by every pixel it looked at, glare included, so the
    # fraction reports how much of the region actually looked like skin.
    skin_fraction = skin_n / float(sampled)

    if skin_fraction >= MIN_SKIN_FRACTION:
        mask = is_skin
        used = skin_n
    else:
        # Unusual light: fall back to every non-glare pixel rather than
        # returning nothing, which is what the engine does.
        mask = not_glare
        used = all_n

    mean_y = float(luma[mask].mean())
    mean_cr = float(cr[mask].mean())
    mean_cb = float(cb[mask].mean())

    # BT.601 full-range YCbCr -> RGB. Linear, so converting the mean equals
    # the mean of the converted pixels.
    r = mean_y + 1.402 * (mean_cr - 128.0)
    g = mean_y - 0.344136 * (mean_cb - 128.0) - 0.714136 * (mean_cr - 128.0)
    b = mean_y + 1.772 * (mean_cb - 128.0)

    return RegionSample(
        valid=True,
        r=r,
        g=g,
        b=b,
        luma=mean_y,
        skin_fraction=skin_fraction,
        pixels=used,
    )
