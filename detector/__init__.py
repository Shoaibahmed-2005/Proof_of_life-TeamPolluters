"""
Proof of Life — synthetic-media / camera-injection detector.

The detector answers a different question from the liveness engine on the
phone. The phone asks "is there a pulse?". This asks "does the pulse belong
to the face in front of the camera?" — which is what a face swap breaks.

Modules:
    dsp                 the rPPG DSP, reused from the app's own reference port
    regions             where to sample (inside the swap mask, outside it, off-skin)
    pulse_consistency   the multi-region detector and its risk fusion
    synth               synthetic multi-region scenarios for tests and calibration
"""

from detector.pulse_consistency import (  # noqa: F401
    PulseConsistencyResult, RegionPulse, Thresholds, Verdict, analyze_regions,
)
from detector.regions import RegionRole, resolve_regions  # noqa: F401

__all__ = [
    "PulseConsistencyResult", "RegionPulse", "Thresholds", "Verdict",
    "analyze_regions", "RegionRole", "resolve_regions",
]
