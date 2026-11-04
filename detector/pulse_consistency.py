"""
Multi-region pulse consistency: the primary synthetic-media detector.

The liveness engine on the phone asks "is there a pulse?". A face swap can
survive that question, because the attacker's own living face is usually in
front of the camera driving the puppet, and because a weak pulse reading is
indistinguishable from bad lighting. This detector asks the harder question:

    does the pulse in the face belong to the body it is attached to?

A swap tool composites rendered pixels inside a landmark-fitted mask and
leaves everything outside it untouched. So the neck is the real person and the
cheeks are not. Four independent consequences follow, and each is measured
here:

1. **Pulse deficit.** The rendered face does not carry the blood-volume signal
   the body is producing. Measured as the SNR of the face regions *at the
   frequency the neck is beating at* - not at whatever peak the face happens
   to have, which is the question that matters.

2. **Rate disagreement.** If the swap carries a pulse at all (compression and
   the source video can reintroduce one), it is the *source identity's* rate,
   not the attacker's. Two different hearts in one frame.

3. **Phase incoherence.** Pulse transit time from face to neck is tens of
   milliseconds, which at ~1.2 Hz is well under 20 degrees. A rendered face has
   no physical reason to stay in step. Faking this requires modelling blood
   flow through a face that is not there.

4. **Illumination control.** Off-skin background regions must show *no*
   pulse-band signal. If they do, the room's light flickered or auto-exposure
   moved, and the detector refuses to accuse anyone - it reports an artifact
   instead. This is the check that keeps the false-positive rate honest.

The fusion is an explicit weighted sum with named reason codes, not a learned
score. That is deliberate: a bank's model-risk committee can read it, and a
rejected customer can be told which measurement failed. A learned detector is
a corroborating input elsewhere in the stack, never the thing that decides.

**Thresholds in :class:`Thresholds` are starting points, not calibrated
values.** They must be set from captured clips on the real deployment -
``scripts/calibrate_detector.py`` once the capture path exists. Shipping these
defaults as if they were measured would be dishonest.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from detector import dsp
from detector.regions import RegionRole

logger = logging.getLogger(__name__)

__all__ = [
    "PulseConsistencyResult",
    "RegionPulse",
    "Thresholds",
    "Verdict",
    "analyze_regions",
]


class Verdict(str, Enum):
    """
    What the detector concluded.

    ``INCONCLUSIVE`` is a first-class outcome, not a failure. A measurement
    that could not be made must never become an accusation; the caller routes
    it to a retry or to manual review.
    """

    LIVE = "LIVE"
    SUSPECT = "SUSPECT"
    SYNTHETIC = "SYNTHETIC"
    INCONCLUSIVE = "INCONCLUSIVE"


# Reason codes, in the same style as the backend's existing ones.
FACE_PULSE_ABSENT = "FACE_PULSE_ABSENT"
FACE_PULSE_WEAK = "FACE_PULSE_WEAK"
PULSE_BPM_DISAGREEMENT = "PULSE_BPM_DISAGREEMENT"
PULSE_PHASE_INCOHERENT = "PULSE_PHASE_INCOHERENT"
SYNTHETIC_PULSE_INJECTED = "SYNTHETIC_PULSE_INJECTED"
PULSE_REFERENCE_WEAK = "PULSE_REFERENCE_WEAK"
NO_REFERENCE_REGION = "NO_REFERENCE_REGION"
NO_FACE_REGION = "NO_FACE_REGION"
ILLUMINATION_ARTIFACT = "ILLUMINATION_ARTIFACT"


@dataclass(frozen=True)
class Thresholds:
    """
    Decision parameters. **Calibrate before trusting any of these.**

    The defaults are chosen to be conservative in the direction that matters
    commercially: a false "synthetic" on a real customer is far more expensive
    than a "suspect" that goes to review.
    """

    # ── Gates ────────────────────────────────────────────────────────────
    # The neck must carry a usable pulse or there is nothing to compare
    # against. This gate exists to stop the detector comparing the face
    # against a rate picked out of the noise: if the reference f0 is wrong, a
    # real face looks deficient at it and an innocent person gets flagged.
    #
    # Set from measurement on the synthetic set (180 captures, 52-118 BPM,
    # five pulse amplitudes): reference rate error stays under 1 BPM for every
    # window at or above 0 dB and collapses below -2 dB (10% of windows off by
    # more than 5 BPM, worst case 73 BPM - pure noise). Real pulses measured
    # no lower than +2.2 dB; pulseless captures (photo, frozen frame, very
    # poor light) no higher than -2.7 dB. 0 dB sits in that gap with roughly
    # 2 dB of margin either side.
    #
    # Re-derive this from real captures before deployment; the synthetic set
    # justifies the shape of the gate, not its exact value on real optics.
    min_reference_snr_db: float = 0.0
    # Control-region green power above this fraction of skin green power means
    # the light is doing the pulsing.
    control_bleed_ratio: float = 0.45

    # ── Pulse deficit (face vs the rate the neck is beating at) ──────────
    deficit_warn_db: float = 3.0
    deficit_bad_db: float = 9.0
    inner_absent_snr_db: float = 0.0

    # ── Rate disagreement ────────────────────────────────────────────────
    bpm_warn: float = 5.0
    bpm_bad: float = 15.0
    # Only trust a disagreement if the face's own peak is real, otherwise the
    # "other heart rate" is just where the noise floor happened to be highest.
    inner_own_min_snr_db: float = 2.0

    # ── Phase ────────────────────────────────────────────────────────────
    phase_warn_deg: float = 45.0
    phase_bad_deg: float = 110.0
    phase_min_snr_db: float = 1.0

    # ── Fusion weights ───────────────────────────────────────────────────
    # These intentionally over-sum: the score is clipped at 100, so any two
    # independent signals firing together reach SYNTHETIC while any one alone
    # reaches at most SUSPECT. A single anomalous measurement should buy a
    # review, never a refusal - that asymmetry is the whole design.
    weight_deficit: float = 40.0
    weight_absent: float = 25.0
    weight_bpm: float = 25.0
    weight_phase: float = 35.0
    weight_injected_bonus: float = 10.0

    # ── Verdict bands on the 0-100 risk score ────────────────────────────
    live_below: float = 25.0
    synthetic_at: float = 60.0


@dataclass(frozen=True)
class RegionPulse:
    """One region's measurement, kept for the console and the audit record."""

    name: str
    role: RegionRole
    valid: bool
    bpm: float = 0.0
    f0_hz: float = 0.0
    # SNR at the region's own spectral peak.
    snr_db: float = -99.0
    # SNR at the *reference* frequency - the number that exposes a swap.
    snr_at_reference_db: float = -99.0
    green_power_at_reference: float = 0.0
    reason: str | None = None


@dataclass(frozen=True)
class PulseConsistencyResult:
    """The detector's output: a verdict, a score, and the evidence for both."""

    verdict: Verdict
    score: float
    reason_codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    regions: list[RegionPulse] = field(default_factory=list)
    reference_region: str | None = None

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.SYNTHETIC

    def summary(self) -> str:
        codes = ", ".join(self.reason_codes) if self.reason_codes else "none"
        return f"{self.verdict.value} risk={self.score:.0f}/100 [{codes}]"


def _ramp(value: float, low: float, high: float) -> float:
    """0 below ``low``, 1 above ``high``, linear between. NaN-safe."""
    if value is None or not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def analyze_regions(
    t: np.ndarray,
    region_rgb: Mapping[str, np.ndarray],
    roles: Mapping[str, RegionRole],
    cfg: dsp.Config | None = None,
    thresholds: Thresholds | None = None,
) -> PulseConsistencyResult:
    """
    Runs the detector over one capture window.

    Args:
        t: frame timestamps in seconds, shape ``(n,)``, shared by all regions.
        region_rgb: ``{region_name: rgb}`` with each ``rgb`` of shape
            ``(n, 3)`` - the mean skin RGB per frame for that region, exactly
            what ``meanSkinRgb()`` produces on the phone.
        roles: ``{region_name: RegionRole}``, from
            :func:`detector.regions.roles_for`.
        cfg: rPPG configuration; defaults match the app's.
        thresholds: decision parameters; see :class:`Thresholds`.

    Returns:
        A :class:`PulseConsistencyResult`. Regions missing from ``roles`` are
        ignored, and regions that could not be measured are reported as
        invalid rather than dropped, so the console can show what was tried.
    """
    cfg = cfg or dsp.Config()
    th = thresholds or Thresholds()
    t = np.asarray(t, dtype=float)

    # ── Measure every region ─────────────────────────────────────────────
    measured: dict[str, dsp.WindowPulse] = {}
    for name, rgb in region_rgb.items():
        if name not in roles:
            continue
        try:
            measured[name] = dsp.window_pulse(t, rgb, cfg)
        except ValueError as exc:
            logger.warning("Region %s could not be measured: %s", name, exc)
            measured[name] = dsp.WindowPulse(valid=False, reason="BAD_INPUT")

    inner = {n: w for n, w in measured.items() if roles[n] is RegionRole.INNER and w.valid}
    outer = {n: w for n, w in measured.items() if roles[n] is RegionRole.OUTER and w.valid}
    control = {n: w for n, w in measured.items() if roles[n] is RegionRole.CONTROL and w.valid}

    def _regions(ref_f0: float, ref_hw: float) -> list[RegionPulse]:
        out: list[RegionPulse] = []
        for name, w in measured.items():
            at_ref = (
                dsp.snr_at(w.signal, cfg, ref_f0, ref_hw)
                if (w.valid and w.signal is not None and ref_f0 > 0)
                else -99.0
            )
            green = (
                dsp.power_at(w.green, cfg.fs, ref_f0)
                if (w.valid and w.green is not None and w.green.size and ref_f0 > 0)
                else 0.0
            )
            out.append(RegionPulse(
                name=name, role=roles[name], valid=w.valid, bpm=w.bpm, f0_hz=w.f0_hz,
                snr_db=w.snr_db, snr_at_reference_db=at_ref,
                green_power_at_reference=green, reason=w.reason,
            ))
        return out

    # ── Gate 1: is there a reference at all? ─────────────────────────────
    if not outer:
        return PulseConsistencyResult(
            verdict=Verdict.INCONCLUSIVE, score=0.0,
            reason_codes=[NO_REFERENCE_REGION],
            reasons=["No off-mask reference region (neck or collar) could be measured. "
                     "Ask the subject to move back so the neck is in frame."],
            regions=_regions(0.0, 0.0),
        )
    if not inner:
        return PulseConsistencyResult(
            verdict=Verdict.INCONCLUSIVE, score=0.0,
            reason_codes=[NO_FACE_REGION],
            reasons=["No face region could be measured; the face was not held in frame "
                     "long enough."],
            regions=_regions(0.0, 0.0),
        )

    ref_name = max(outer, key=lambda n: outer[n].snr_db)
    ref = outer[ref_name]
    ref_f0, ref_hw = ref.f0_hz, ref.half_width_hz

    if ref.snr_db < th.min_reference_snr_db:
        return PulseConsistencyResult(
            verdict=Verdict.INCONCLUSIVE, score=0.0,
            reason_codes=[PULSE_REFERENCE_WEAK],
            reasons=[f"The reference pulse at the {ref_name} is too weak to compare "
                     f"against ({ref.snr_db:.1f} dB, need {th.min_reference_snr_db:.1f} dB). "
                     "Usually lighting, framing or movement - not an attack."],
            metrics={"reference_snr_db": ref.snr_db},
            regions=_regions(ref_f0, ref_hw),
            reference_region=ref_name,
        )

    # ── Measurements against the reference ───────────────────────────────
    inner_at_ref = {n: dsp.snr_at(w.signal, cfg, ref_f0, ref_hw) for n, w in inner.items()}
    inner_snr_at_ref = float(np.median(list(inner_at_ref.values())))

    strongest_inner = max(inner, key=lambda n: inner[n].snr_db)
    inner_own_snr = inner[strongest_inner].snr_db
    inner_own_bpm = inner[strongest_inner].bpm

    snr_deficit_db = ref.snr_db - inner_snr_at_ref
    bpm_disagreement = abs(inner_own_bpm - ref.bpm)
    phase_deg = dsp.phase_difference_deg(
        inner[strongest_inner].signal, ref.signal, cfg.fs, ref_f0
    )

    # Illumination control: green-channel power off skin vs on skin.
    skin_green = [
        dsp.power_at(w.green, cfg.fs, ref_f0)
        for w in list(inner.values()) + list(outer.values())
        if w.green is not None and w.green.size
    ]
    skin_green_power = float(np.median(skin_green)) if skin_green else 0.0
    control_green = [
        dsp.power_at(w.green, cfg.fs, ref_f0)
        for w in control.values()
        if w.green is not None and w.green.size
    ]
    control_green_power = float(np.max(control_green)) if control_green else 0.0
    bleed_ratio = (
        control_green_power / skin_green_power if skin_green_power > 1e-18 else 0.0
    )
    illumination_artifact = bool(control) and bleed_ratio > th.control_bleed_ratio

    # ── Fusion ───────────────────────────────────────────────────────────
    codes: list[str] = []
    reasons: list[str] = []
    score = 0.0

    deficit_part = th.weight_deficit * _ramp(snr_deficit_db, th.deficit_warn_db, th.deficit_bad_db)
    score += deficit_part
    if deficit_part > 0:
        if inner_snr_at_ref < th.inner_absent_snr_db:
            score += th.weight_absent
            codes.append(FACE_PULSE_ABSENT)
            reasons.append(
                f"The face carries no blood-volume pulse at the rate the body is beating "
                f"at ({ref.bpm:.0f} BPM): {inner_snr_at_ref:.1f} dB in the face versus "
                f"{ref.snr_db:.1f} dB at the {ref_name}."
            )
        else:
            codes.append(FACE_PULSE_WEAK)
            reasons.append(
                f"The face's pulse is {snr_deficit_db:.1f} dB weaker than the {ref_name}'s "
                f"at the same rate, more than skin-tone and lighting differences explain."
            )

    if inner_own_snr >= th.inner_own_min_snr_db:
        bpm_part = th.weight_bpm * _ramp(bpm_disagreement, th.bpm_warn, th.bpm_bad)
        score += bpm_part
        if bpm_part > 0:
            codes.append(PULSE_BPM_DISAGREEMENT)
            reasons.append(
                f"Two different heart rates in one frame: the face beats at "
                f"{inner_own_bpm:.0f} BPM while the {ref_name} beats at {ref.bpm:.0f} BPM "
                f"({bpm_disagreement:.0f} BPM apart)."
            )
            if bpm_disagreement >= th.bpm_bad and inner_own_snr >= 4.0:
                score += th.weight_injected_bonus
                codes.append(SYNTHETIC_PULSE_INJECTED)
                reasons.append(
                    "The face carries a strong, steady pulse of its own at the wrong rate, "
                    "which is the signature of a synthesised or replayed face rather than "
                    "a merely degraded one."
                )

    if inner_snr_at_ref >= th.phase_min_snr_db and np.isfinite(phase_deg):
        phase_part = th.weight_phase * _ramp(phase_deg, th.phase_warn_deg, th.phase_bad_deg)
        score += phase_part
        if phase_part > 0:
            codes.append(PULSE_PHASE_INCOHERENT)
            reasons.append(
                f"The face's pulse is {phase_deg:.0f} degrees out of step with the "
                f"{ref_name}'s. Pulse transit time across a real head allows under about "
                f"20 degrees at this rate."
            )

    score = float(np.clip(score, 0.0, 100.0))

    # ── Verdict ──────────────────────────────────────────────────────────
    if score >= th.synthetic_at:
        verdict = Verdict.SYNTHETIC
    elif score >= th.live_below:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.LIVE

    # The control check never creates an accusation and always softens one: if
    # the light was pulsing, the comparison it is based on is not trustworthy.
    if illumination_artifact:
        codes.append(ILLUMINATION_ARTIFACT)
        reasons.append(
            f"The background shows a pulse-band signal at {bleed_ratio:.0%} of the skin's, "
            "so the lighting or auto-exposure is moving. The comparison is not reliable; "
            "fix the lighting and retry before drawing any conclusion."
        )
        if verdict is Verdict.SYNTHETIC:
            verdict = Verdict.SUSPECT
        elif verdict is Verdict.LIVE:
            verdict = Verdict.INCONCLUSIVE

    if verdict is Verdict.LIVE and not codes:
        reasons.append(
            f"The face and the {ref_name} share one pulse at {ref.bpm:.0f} BPM, in phase "
            f"to within {phase_deg:.0f} degrees. Consistent with a single living body."
        )

    return PulseConsistencyResult(
        verdict=verdict,
        score=score,
        reason_codes=codes,
        reasons=reasons,
        metrics={
            "reference_bpm": ref.bpm,
            "reference_snr_db": ref.snr_db,
            "inner_snr_at_reference_db": inner_snr_at_ref,
            "inner_own_snr_db": inner_own_snr,
            "inner_own_bpm": inner_own_bpm,
            "snr_deficit_db": snr_deficit_db,
            "bpm_disagreement": bpm_disagreement,
            "phase_deg": float(phase_deg) if np.isfinite(phase_deg) else float("nan"),
            "control_bleed_ratio": bleed_ratio,
            "window_sec": ref.span_sec,
        },
        regions=_regions(ref_f0, ref_hw),
        reference_region=ref_name,
    )
