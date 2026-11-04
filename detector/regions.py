"""
Where the detector samples skin, and why.

A face swap (DeepFaceLive, Deep-Live-Cam, roop, any inswapper variant)
replaces pixels inside a mask fitted to the face and composites them over the
original frame. Everything outside that mask is still the real person in front
of the camera. That asymmetry is the whole basis of this detector, so the
region layout is not cosmetic - it decides whether the detector works.

Three roles:

``INNER``
    Skin that is certainly *inside* any swap mask: forehead and both cheeks.
    These are the same three rectangles the Android app already uses
    (``FaceRois.kt``), so a number measured here is directly comparable with
    what the phone reports for the same person.

``OUTER``
    Skin that is certainly *outside* the mask. The neck is the primary one and
    the only one worth relying on: no swap tool extends there, because the mask
    is fitted to facial landmarks. Ear and collar strips are secondary - they
    sit near the mask boundary, so a generous mask can contaminate them.

``CONTROL``
    Off-skin background to either side of the head. There is no blood here, so
    any periodic signal found in a control region came from the light, the
    camera's auto-exposure, or compression. It is the detector's own sanity
    check against blaming a person for the room's lighting.

All rectangles are fractions of the detected face bounding box in upright
image coordinates, which is the same convention ``FaceRois.kt`` uses. Values
below 0 or above 1 are deliberate: they address the neck, collar and
background, which lie outside the face box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["DEFAULT_LAYOUT", "Region", "RegionRole", "resolve_regions"]


class RegionRole(str, Enum):
    """What a region is for. See the module docstring."""

    INNER = "INNER"
    OUTER = "OUTER"
    CONTROL = "CONTROL"


@dataclass(frozen=True)
class Region:
    """A sampling rectangle, as fractions of the face bounding box."""

    name: str
    role: RegionRole
    left: float
    top: float
    right: float
    bottom: float
    # Secondary regions sit close to a swap mask boundary, so they can be
    # contaminated by a generous mask. The detector prefers primary regions and
    # falls back to secondary ones only when it has nothing better.
    primary: bool = True
    note: str = ""


# Ordered: primary regions first within each role.
DEFAULT_LAYOUT: tuple[Region, ...] = (
    # ── Inside the swap mask (identical to FaceRois.kt) ──────────────────
    Region("forehead", RegionRole.INNER, 0.30, 0.08, 0.70, 0.24,
           note="Avoids brows and hairline; the strongest rPPG site on a real face."),
    Region("cheek_left", RegionRole.INNER, 0.16, 0.50, 0.38, 0.72,
           note="Image-left cheek, avoiding the nose and mouth."),
    Region("cheek_right", RegionRole.INNER, 0.62, 0.50, 0.84, 0.72,
           note="Image-right cheek, avoiding the nose and mouth."),

    # ── Outside the swap mask ────────────────────────────────────────────
    Region("neck", RegionRole.OUTER, 0.28, 1.08, 0.72, 1.34,
           note="Primary reference. Below the face box, so no landmark-fitted "
                "mask reaches it. Well perfused and usually unobstructed."),
    Region("collar", RegionRole.OUTER, 0.15, 1.36, 0.85, 1.60, primary=False,
           note="Upper chest. Often out of frame or covered by clothing, so "
                "secondary; useful on a seated desktop framing."),
    Region("ear_left", RegionRole.OUTER, 0.00, 0.44, 0.09, 0.66, primary=False,
           note="Sideburn / ear strip at the face-box edge. Close to the mask "
                "boundary, so treated as secondary."),
    Region("ear_right", RegionRole.OUTER, 0.91, 0.44, 1.00, 0.66, primary=False,
           note="Mirror of ear_left."),

    # ── Off skin ─────────────────────────────────────────────────────────
    Region("background_left", RegionRole.CONTROL, -0.70, 0.05, -0.20, 0.55,
           note="No blood here. A pulse-band signal means the light flickered "
                "or auto-exposure moved."),
    Region("background_right", RegionRole.CONTROL, 1.20, 0.05, 1.70, 0.55,
           note="Mirror of background_left."),
)


def resolve_regions(
    face_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    layout: tuple[Region, ...] = DEFAULT_LAYOUT,
    min_area_px: int = 400,
    min_retained_fraction: float = 0.4,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Turns the fractional layout into pixel rectangles for one frame.

    Args:
        face_box: ``(x, y, w, h)`` of the detected face in upright image
            coordinates.
        image_width, image_height: frame size, used to clip.
        layout: regions to resolve; defaults to :data:`DEFAULT_LAYOUT`.
        min_area_px: drop a region clipped below this area - too few pixels
            makes the colour mean noisy enough to be worthless.
        min_retained_fraction: drop a region if clipping removed more than
            this much of it, so a half-visible neck is not silently compared
            against a full one.

    Returns:
        ``{name: (x, y, w, h)}`` for the regions that survived. Regions that
        fall outside the frame are absent rather than returned empty, so the
        caller can see what was actually measurable: a framing that crops the
        neck should make the detector say "inconclusive", not "synthetic".
    """
    fx, fy, fw, fh = face_box
    if fw <= 0 or fh <= 0:
        return {}

    out: dict[str, tuple[int, int, int, int]] = {}
    for region in layout:
        x0 = fx + region.left * fw
        y0 = fy + region.top * fh
        x1 = fx + region.right * fw
        y1 = fy + region.bottom * fh
        wanted = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if wanted <= 0:
            continue

        cx0 = int(round(max(0.0, min(x0, image_width))))
        cy0 = int(round(max(0.0, min(y0, image_height))))
        cx1 = int(round(max(0.0, min(x1, image_width))))
        cy1 = int(round(max(0.0, min(y1, image_height))))
        w, h = cx1 - cx0, cy1 - cy0
        if w <= 0 or h <= 0:
            continue
        if w * h < min_area_px:
            continue
        if (w * h) / wanted < min_retained_fraction:
            continue
        out[region.name] = (cx0, cy0, w, h)
    return out


def roles_for(layout: tuple[Region, ...] = DEFAULT_LAYOUT) -> dict[str, RegionRole]:
    """``{name: role}`` for a layout, which is what the detector takes."""
    return {region.name: region.role for region in layout}


def primary_names(layout: tuple[Region, ...] = DEFAULT_LAYOUT) -> set[str]:
    """Names of the primary (not boundary-adjacent) regions."""
    return {region.name for region in layout if region.primary}
