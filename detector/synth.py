"""
Synthetic multi-region capture, for tests and threshold calibration.

This generates the per-region colour traces a camera would produce, with the
pulse placed in regions by role, so the detector can be exercised without a
GPU, a camera, DeepFaceLive or OBS. It extends the single-region generator in
``tools/rppg/synth.py`` with the thing that matters here: the face and the neck
can be given *different* pulses.

What this is and is not: these are signal-level models of what a face swap does
to the rPPG signal, good enough to prove the detector's logic, its reason codes
and its refusal to accuse on bad input. They are **not** a substitute for
calibration on real captured clips - a synthetic "deepfake" is as weak as the
assumptions baked into it. Real DeepFaceLive output is messier: it leaks some
of the driver's pulse through the blend edge, and compression reintroduces
periodic structure. Thresholds must come from real captures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SCENARIOS", "MultiRegionScenario", "generate"]

# Mean skin colour and the relative pulsatile strength per channel (PBV), as in
# tools/rppg/synth.py so the two generators agree.
SKIN_RGB = np.array([182.0, 128.0, 108.0])
PBV = np.array([0.33, 0.77, 0.53])
BACKGROUND_RGB = np.array([118.0, 120.0, 124.0])

# Per-region pulse strength on a real face, relative to the scenario amplitude.
# The forehead is the best rPPG site; the neck is close behind; cheeks are
# weaker. Phase offsets are the few-degree spread a real head actually shows.
REGION_GAIN: dict[str, float] = {
    "forehead": 1.00,
    "cheek_left": 0.82,
    "cheek_right": 0.80,
    "neck": 0.95,
    "collar": 0.55,
    "ear_left": 0.45,
    "ear_right": 0.44,
}
REGION_PHASE_DEG: dict[str, float] = {
    "forehead": 0.0,
    "cheek_left": 4.0,
    "cheek_right": 5.0,
    "neck": 9.0,
    "collar": 13.0,
    "ear_left": 6.0,
    "ear_right": 7.0,
}
INNER_REGIONS = ("forehead", "cheek_left", "cheek_right")
OUTER_REGIONS = ("neck", "collar", "ear_left", "ear_right")
CONTROL_REGIONS = ("background_left", "background_right")


@dataclass
class MultiRegionScenario:
    """One synthetic capture. Amplitudes are relative (0.004 = 0.4%)."""

    name: str
    description: str = ""
    duration_s: float = 20.0
    fps: float = 30.0

    # The body's true heart rate, which drives the OUTER regions.
    bpm: float = 72.0
    outer_pulse_amp: float = 0.0038

    # The face. ``inner_bpm=None`` means it shares the body's rate.
    inner_pulse_amp: float = 0.0040
    inner_bpm: float | None = None
    inner_phase_deg: float = 0.0

    # Sensor and scene noise.
    noise_std: float = 0.15
    motion_std: float = 0.002
    drift_amp: float = 0.02

    # Illumination that moves. ``flicker_hz=None`` puts it at the heart rate,
    # which is the genuinely awkward case for any rPPG detector.
    flicker_hz: float | None = 0.0
    flicker_amp: float = 0.0
    # How chromatic the flicker is. Perfectly achromatic flicker is cancelled
    # by POS, so a confuser has to tint.
    flicker_tint: np.ndarray = field(default_factory=lambda: np.array([0.85, 1.0, 1.15]))

    drop_prob: float = 0.02
    seed: int = 0
    expect: dict = field(default_factory=dict)


def _phase_track(rel: np.ndarray, bpm: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Heart-rate phase with HRV and a breathing component, integrated."""
    fine = np.arange(0, duration_s + 1, 0.01)
    hr = (
        bpm
        + 1.5 * np.sin(2 * np.pi * 0.25 * fine)
        + rng.normal(0, 0.3, fine.size).cumsum() * 0.02
    )
    phase = 2 * np.pi * np.cumsum(hr / 60.0) * 0.01
    return np.interp(rel, fine, phase)


def generate(s: MultiRegionScenario) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns ``(t, {region_name: rgb[n, 3]})``.

    All regions share the timestamp vector, as they would when sampled from the
    same frames.
    """
    rng = np.random.default_rng(s.seed)

    n_nominal = int(s.duration_s * s.fps)
    t = np.arange(n_nominal) / s.fps + rng.normal(0, 0.003, n_nominal)
    t = np.sort(t) + 5.0  # sensor clocks do not start at zero
    t = t[rng.random(n_nominal) >= s.drop_prob]
    rel = t - t[0]

    body_phase = _phase_track(rel, s.bpm, s.duration_s, rng)
    if s.inner_bpm is None:
        inner_phase = body_phase + np.radians(s.inner_phase_deg)
    else:
        # A separate heart: its own rate and its own HRV, unrelated to the body.
        inner_phase = _phase_track(rel, s.inner_bpm, s.duration_s, rng) + np.radians(
            s.inner_phase_deg
        )

    def waveform(phase: np.ndarray) -> np.ndarray:
        # Fundamental plus the dicrotic second harmonic a real pulse carries.
        return np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)

    flicker_hz = s.bpm / 60.0 if s.flicker_hz is None else s.flicker_hz
    drift = s.drift_amp * np.sin(2 * np.pi * 0.03 * rel + rng.uniform(0, 6))
    flicker = s.flicker_amp * np.sin(2 * np.pi * flicker_hz * rel) if s.flicker_amp else 0.0

    def illumination(tint: bool) -> np.ndarray:
        """(n, 3) multiplicative illumination, shared by every region."""
        base = 1.0 + drift
        out = np.repeat(base[:, None], 3, axis=1)
        if s.flicker_amp:
            out = out + np.outer(flicker, s.flicker_tint if tint else np.ones(3))
        return out

    illum = illumination(tint=True)

    def skin_region(name: str, phase: np.ndarray, amp: float) -> np.ndarray:
        # The region's own phase offset is applied to the phase track itself,
        # so the shift is exact rather than approximated on the waveform.
        gain = REGION_GAIN.get(name, 0.6)
        wave = waveform(phase + np.radians(REGION_PHASE_DEG.get(name, 0.0)))
        pulsatile = 1.0 + amp * gain * wave[:, None] * PBV[None, :]
        motion = np.convolve(
            rng.normal(0, s.motion_std, rel.size), np.ones(15) / np.sqrt(15), mode="same"
        )
        return (
            illum * SKIN_RGB[None, :] * pulsatile
            + motion[:, None] * SKIN_RGB[None, :]
            + rng.normal(0, s.noise_std, (rel.size, 3))
        )

    regions: dict[str, np.ndarray] = {}
    for name in INNER_REGIONS:
        regions[name] = skin_region(name, inner_phase, s.inner_pulse_amp)
    for name in OUTER_REGIONS:
        regions[name] = skin_region(name, body_phase, s.outer_pulse_amp)
    for name in CONTROL_REGIONS:
        # No blood. Only the light and the sensor.
        regions[name] = illum * BACKGROUND_RGB[None, :] + rng.normal(
            0, s.noise_std, (rel.size, 3)
        )

    return t, regions


SCENARIOS: tuple[MultiRegionScenario, ...] = (
    MultiRegionScenario(
        name="real_human",
        description="A living person on a real camera. One heart, one phase.",
        inner_pulse_amp=0.0040, outer_pulse_amp=0.0038, inner_phase_deg=6.0, seed=1,
        expect={"verdict": "LIVE"},
    ),
    MultiRegionScenario(
        name="deepfake_pulse_destroyed",
        description="The common case. The swap renders a face with no "
                    "blood-volume signal; the attacker's own neck still pulses.",
        inner_pulse_amp=0.00012, outer_pulse_amp=0.0038, seed=2,
        expect={"verdict": "SYNTHETIC", "code": "FACE_PULSE_ABSENT"},
    ),
    MultiRegionScenario(
        name="deepfake_wrong_rate",
        description="A sophisticated swap that leaks the source identity's "
                    "pulse: two hearts in one frame.",
        inner_pulse_amp=0.0042, inner_bpm=98.0, outer_pulse_amp=0.0038, seed=3,
        expect={"verdict": "SYNTHETIC", "code": "PULSE_BPM_DISAGREEMENT"},
    ),
    MultiRegionScenario(
        name="deepfake_phase_scrambled",
        description="Right rate, right strength, wrong phase - a swap that "
                    "reproduced the rhythm but not the blood flow.",
        inner_pulse_amp=0.0040, outer_pulse_amp=0.0038, inner_phase_deg=135.0, seed=4,
        expect={"verdict_in": ("SUSPECT", "SYNTHETIC"), "code": "PULSE_PHASE_INCOHERENT"},
    ),
    MultiRegionScenario(
        name="still_photo",
        description="A printed photo or a frozen frame: nothing pulses "
                    "anywhere, so there is nothing to compare and no accusation.",
        inner_pulse_amp=0.0, outer_pulse_amp=0.0, seed=5,
        expect={"verdict": "INCONCLUSIVE", "code": "PULSE_REFERENCE_WEAK"},
    ),
    MultiRegionScenario(
        name="real_human_poor_light",
        description="A real person in bad light. The detector must refuse to "
                    "decide rather than call a customer a fraud.",
        inner_pulse_amp=0.0009, outer_pulse_amp=0.0008, noise_std=0.9, seed=6,
        expect={"verdict": "INCONCLUSIVE"},
    ),
    MultiRegionScenario(
        name="light_flicker_at_heart_rate",
        description="Real person, but the room light pulses at their heart "
                    "rate. The control regions catch it.",
        inner_pulse_amp=0.0040, outer_pulse_amp=0.0038,
        flicker_hz=None, flicker_amp=0.010, seed=7,
        expect={"verdict_not": "SYNTHETIC", "code": "ILLUMINATION_ARTIFACT"},
    ),
)
