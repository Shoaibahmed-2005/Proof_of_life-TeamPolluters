"""
Tests for the multi-region pulse-consistency detector.

Two things are being checked, and the second matters more than the first:

1. The detector flags the three ways a face swap breaks pulse consistency
   (absent pulse, wrong rate, wrong phase).
2. The detector *refuses to accuse* when the measurement was not good enough -
   a still photo, poor light, a cropped neck, flickering light. A false
   "synthetic" on a real customer is the expensive failure, so these cases get
   more tests than the attacks do.
"""

from __future__ import annotations

import numpy as np
import pytest

from detector import dsp, synth
from detector.pulse_consistency import (
    FACE_PULSE_ABSENT,
    ILLUMINATION_ARTIFACT,
    NO_FACE_REGION,
    NO_REFERENCE_REGION,
    PULSE_BPM_DISAGREEMENT,
    PULSE_PHASE_INCOHERENT,
    PULSE_REFERENCE_WEAK,
    Thresholds,
    Verdict,
    analyze_regions,
)
from detector.regions import DEFAULT_LAYOUT, RegionRole, resolve_regions, roles_for

ROLES = roles_for()


def run(scenario: synth.MultiRegionScenario):
    t, regions = synth.generate(scenario)
    return analyze_regions(t, regions, ROLES)


# ── The DSP layer behaves as the engine does ────────────────────────────


def test_dsp_recovers_the_known_rate():
    s = synth.MultiRegionScenario(name="probe", bpm=72.0, seed=11)
    t, regions = synth.generate(s)
    pulse = dsp.window_pulse(t, regions["forehead"])
    assert pulse.valid
    assert abs(pulse.bpm - 72.0) < 4.0
    assert pulse.snr_db > 4.0


def test_phase_difference_is_symmetric_and_bounded():
    cfg = dsp.Config()
    n = 600
    x = np.sin(2 * np.pi * 1.2 * np.arange(n) / cfg.fs)
    y = np.sin(2 * np.pi * 1.2 * np.arange(n) / cfg.fs - np.pi / 2)
    ab = dsp.phase_difference_deg(x, y, cfg.fs, 1.2)
    ba = dsp.phase_difference_deg(y, x, cfg.fs, 1.2)
    assert 0.0 <= ab <= 180.0
    assert abs(ab - ba) < 1e-6
    assert abs(ab - 90.0) < 3.0


def test_snr_at_a_frequency_the_signal_does_not_carry_is_low():
    cfg = dsp.Config()
    s = synth.MultiRegionScenario(name="probe", bpm=72.0, seed=12)
    t, regions = synth.generate(s)
    pulse = dsp.window_pulse(t, regions["forehead"], cfg)
    at_own = dsp.snr_at(pulse.signal, cfg, pulse.f0_hz, pulse.half_width_hz)
    at_other = dsp.snr_at(pulse.signal, cfg, 3.3, pulse.half_width_hz)
    assert at_own > at_other + 5.0


# ── A real person passes ────────────────────────────────────────────────


def test_real_human_is_live():
    result = run(synth.SCENARIOS[0])
    assert result.verdict is Verdict.LIVE, result.summary()
    assert result.score < Thresholds().live_below
    assert result.reason_codes == []
    assert result.reference_region in ("neck", "collar", "ear_left", "ear_right")
    assert abs(result.metrics["bpm_disagreement"]) < 5.0


def test_real_human_is_live_across_seeds_and_rates():
    """The pass must not depend on one lucky random draw."""
    for seed, bpm in enumerate([58.0, 72.0, 88.0, 104.0], start=20):
        s = synth.MultiRegionScenario(
            name=f"real_{bpm:.0f}", bpm=bpm, inner_pulse_amp=0.0040,
            outer_pulse_amp=0.0038, inner_phase_deg=6.0, seed=seed,
        )
        result = run(s)
        assert result.verdict is Verdict.LIVE, f"{bpm} BPM: {result.summary()}"


# ── The three swap signatures are caught ────────────────────────────────


def test_swap_with_no_pulse_is_synthetic():
    result = run(synth.SCENARIOS[1])
    assert result.verdict is Verdict.SYNTHETIC, result.summary()
    assert FACE_PULSE_ABSENT in result.reason_codes
    assert result.metrics["snr_deficit_db"] > Thresholds().deficit_bad_db
    assert result.blocked


def test_swap_leaking_a_second_heart_rate_is_synthetic():
    result = run(synth.SCENARIOS[2])
    assert result.verdict is Verdict.SYNTHETIC, result.summary()
    assert PULSE_BPM_DISAGREEMENT in result.reason_codes
    assert result.metrics["bpm_disagreement"] > 15.0


def test_swap_with_scrambled_phase_is_at_least_suspect():
    """
    Phase alone is one signal, so it earns a review rather than a refusal.
    That is the intended behaviour, not a weakness.
    """
    result = run(synth.SCENARIOS[3])
    assert result.verdict in (Verdict.SUSPECT, Verdict.SYNTHETIC), result.summary()
    assert PULSE_PHASE_INCOHERENT in result.reason_codes
    assert result.metrics["phase_deg"] > Thresholds().phase_warn_deg


def test_a_single_signal_never_reaches_synthetic_on_its_own():
    """
    The weights over-sum and the score clips, so two independent signals are
    needed to block. Verified here on the phase-only case.
    """
    result = run(synth.SCENARIOS[3])
    assert result.reason_codes == [PULSE_PHASE_INCOHERENT], result.reason_codes
    assert result.verdict is Verdict.SUSPECT, result.summary()


# ── The detector refuses to accuse on a bad measurement ─────────────────


def test_still_photo_is_inconclusive_not_synthetic():
    result = run(synth.SCENARIOS[4])
    assert result.verdict is Verdict.INCONCLUSIVE, result.summary()
    assert PULSE_REFERENCE_WEAK in result.reason_codes
    assert not result.blocked


def test_real_person_in_poor_light_is_inconclusive():
    result = run(synth.SCENARIOS[5])
    assert result.verdict is Verdict.INCONCLUSIVE, result.summary()
    assert not result.blocked
    assert FACE_PULSE_ABSENT not in result.reason_codes


def test_light_flickering_at_the_heart_rate_is_not_an_accusation():
    result = run(synth.SCENARIOS[6])
    assert result.verdict is not Verdict.SYNTHETIC, result.summary()
    assert ILLUMINATION_ARTIFACT in result.reason_codes


def test_illumination_artifact_downgrades_a_synthetic_verdict():
    """A swap measured under pulsing light must be reviewed, not auto-blocked."""
    s = synth.MultiRegionScenario(
        name="swap_under_flicker", inner_pulse_amp=0.00012, outer_pulse_amp=0.0038,
        flicker_hz=None, flicker_amp=0.012, seed=31,
    )
    result = run(s)
    if ILLUMINATION_ARTIFACT in result.reason_codes:
        assert result.verdict is not Verdict.SYNTHETIC, result.summary()


def test_missing_neck_is_inconclusive():
    t, regions = synth.generate(synth.SCENARIOS[1])
    face_only = {k: v for k, v in regions.items() if ROLES[k] is not RegionRole.OUTER}
    result = analyze_regions(t, face_only, ROLES)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert NO_REFERENCE_REGION in result.reason_codes


def test_missing_face_is_inconclusive():
    t, regions = synth.generate(synth.SCENARIOS[0])
    neck_only = {k: v for k, v in regions.items() if ROLES[k] is not RegionRole.INNER}
    result = analyze_regions(t, neck_only, ROLES)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert NO_FACE_REGION in result.reason_codes


def test_window_too_short_is_inconclusive():
    t, regions = synth.generate(synth.SCENARIOS[0])
    keep = t < t[0] + 1.0
    short = {k: v[keep] for k, v in regions.items()}
    result = analyze_regions(t[keep], short, ROLES)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert not result.blocked


# ── Every declared scenario expectation holds ───────────────────────────


@pytest.mark.parametrize("scenario", synth.SCENARIOS, ids=lambda s: s.name)
def test_declared_expectations(scenario):
    result = run(scenario)
    expect = scenario.expect
    if "verdict" in expect:
        assert result.verdict.value == expect["verdict"], result.summary()
    if "verdict_in" in expect:
        assert result.verdict.value in expect["verdict_in"], result.summary()
    if "verdict_not" in expect:
        assert result.verdict.value != expect["verdict_not"], result.summary()
    if "code" in expect:
        assert expect["code"] in result.reason_codes, result.reason_codes


# ── Result shape, for the console and the audit record ──────────────────


def test_result_carries_evidence_for_every_region():
    result = run(synth.SCENARIOS[1])
    names = {r.name for r in result.regions}
    assert names == set(ROLES)
    assert all(r.reason_codes is not None for r in [result])
    assert result.reasons, "a verdict must always come with a human-readable reason"
    for key in ("reference_bpm", "reference_snr_db", "snr_deficit_db", "phase_deg"):
        assert key in result.metrics


def test_reasons_never_leak_region_internals_to_the_customer():
    """Reason text is shown to officers; it must read as plain language."""
    result = run(synth.SCENARIOS[1])
    for reason in result.reasons:
        assert "ndarray" not in reason
        assert reason[0].isupper()
        assert reason.rstrip().endswith((".", "!"))


# ── Region geometry ─────────────────────────────────────────────────────


def test_resolve_regions_clips_and_drops_offscreen_regions():
    resolved = resolve_regions((200, 120, 240, 300), 1280, 720)
    assert "forehead" in resolved and "neck" in resolved
    for x, y, w, h in resolved.values():
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= 1280 and y + h <= 720


def test_resolve_regions_drops_the_neck_when_the_frame_cuts_it_off():
    # Face box at the very bottom of the frame: the neck has nowhere to go.
    resolved = resolve_regions((200, 600, 240, 300), 1280, 720)
    assert "neck" not in resolved
    assert "forehead" in resolved


def test_background_control_regions_resolve_on_a_wide_frame():
    resolved = resolve_regions((500, 150, 240, 300), 1280, 720)
    assert "background_left" in resolved
    assert "background_right" in resolved


def test_layout_has_all_three_roles():
    roles = {r.role for r in DEFAULT_LAYOUT}
    assert roles == {RegionRole.INNER, RegionRole.OUTER, RegionRole.CONTROL}
