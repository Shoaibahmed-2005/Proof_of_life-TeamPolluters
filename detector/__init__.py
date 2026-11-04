"""
Proof of Life — synthetic-media / camera-injection detector.

The detector answers a different question from the liveness engine on the
phone. The phone asks "is there a pulse?". This asks "does the pulse belong
to the face in front of the camera?" — which is what a face swap breaks.

Modules:
    dsp                 the rPPG DSP, reused from the app's own reference port
    regions             where to sample (inside the swap mask, outside it, off-skin)
    skin                mean skin colour per region, mirroring the C++ sampler
    faces               face tracking, smoothed and gap-tolerant
    capture             the camera front end: video in, a verdict out
    pulse_consistency   the multi-region detector and its risk fusion
    synth               synthetic multi-region scenarios for tests and calibration

Quick start:
    python -m detector.capture --list-cameras
    python -m detector.capture --source 0 --seconds 20
"""

from detector.pulse_consistency import (  # noqa: F401
    PulseConsistencyResult, RegionPulse, Thresholds, Verdict, analyze_regions,
)
from detector.regions import RegionRole, resolve_regions, roles_for  # noqa: F401

__all__ = [
    "PulseConsistencyResult", "RegionPulse", "Thresholds", "Verdict",
    "analyze_regions", "RegionRole", "resolve_regions", "roles_for",
]


def __getattr__(name: str):
    """
    Capture helpers are imported lazily.

    They need OpenCV, which the analysis side does not: the detector and its
    tests must stay runnable on a machine with no camera stack installed.
    """
    if name in {"CaptureResult", "capture", "analyze_capture", "list_cameras"}:
        from detector import capture as _capture

        return getattr(_capture, name)
    if name in {"FaceBox", "FaceTracker"}:
        from detector import faces as _faces

        return getattr(_faces, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
