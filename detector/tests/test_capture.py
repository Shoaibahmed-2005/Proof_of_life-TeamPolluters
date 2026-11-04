"""
Tests for the camera front end.

These drive the real pixel path — synthetic BGR frames in, a verdict out —
rather than feeding the detector clean waveforms. That matters, because
everything between a camera and the detector is where a working algorithm
quietly stops working: colour conversion, the skin test, region placement,
8-bit quantisation, and frames that have to share one timestamp vector.

No camera is needed. Frames are rendered with skin-coloured region patches
whose brightness is modulated by a pulse, plus sensor-like noise. The noise is
not decoration: without it every pixel in a flat patch would quantise to the
same 8-bit value and a sub-count pulse would vanish entirely. Real sensors
dither; these frames do too.
"""

from __future__ import annotations

import numpy as np
import pytest

from detector import skin
from detector.capture import CaptureResult, analyze_capture
from detector.pulse_consistency import FACE_PULSE_ABSENT, Verdict
from detector.regions import RegionRole, resolve_regions, roles_for

cv2 = pytest.importorskip("cv2", reason="OpenCV is needed for the camera front end")

FRAME_W, FRAME_H = 640, 360
FACE_BOX = (250, 75, 120, 150)
SKIN_RGB = np.array([182.0, 128.0, 108.0])
BACKGROUND_RGB = np.array([118.0, 120.0, 124.0])
PBV = np.array([0.33, 0.77, 0.53])
FPS = 30.0


def _render_frame(rects, roles, region_rgb, rng, noise_std=2.0):
    """One BGR frame: flat background, skin patches at the region rectangles."""
    frame = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = BACKGROUND_RGB[::-1].astype(np.uint8)  # BGR

    for name, (x, y, w, h) in rects.items():
        rgb = region_rgb[name]
        patch = np.repeat(np.repeat(rgb[None, None, ::-1], h, axis=0), w, axis=1)
        patch = patch + rng.normal(0, noise_std, (h, w, 3))
        frame[y:y + h, x:x + w] = np.clip(patch, 0, 255).astype(np.uint8)
    return frame


def _run(seconds, inner_amp, outer_amp, bpm=72.0, inner_phase_deg=6.0, seed=0):
    """Renders a clip and samples it exactly as capture() does."""
    rng = np.random.default_rng(seed)
    roles = roles_for()
    rects = resolve_regions(FACE_BOX, FRAME_W, FRAME_H)
    assert rects, "region layout produced nothing for the test frame"

    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    phase = 2 * np.pi * bpm / 60.0 * t

    rows_t, rows = [], {name: [] for name in rects}
    for i in range(n):
        values = {}
        for name in rects:
            role = roles[name]
            if role is RegionRole.CONTROL:
                values[name] = BACKGROUND_RGB
                continue
            amp = inner_amp if role is RegionRole.INNER else outer_amp
            ph = phase[i] + (np.radians(inner_phase_deg) if role is RegionRole.INNER else 0.0)
            values[name] = SKIN_RGB * (1 + amp * np.sin(ph) * PBV)

        frame = _render_frame(rects, roles, values, rng)
        ycrcb = skin.to_ycrcb(frame)
        samples = {nm: skin.sample_region(ycrcb, r, step=2) for nm, r in rects.items()}
        if not all(s.valid for s in samples.values()):
            continue
        rows_t.append(t[i])
        for nm, s in samples.items():
            rows[nm].append((s.r, s.g, s.b))

    return CaptureResult(
        t=np.asarray(rows_t),
        region_rgb={nm: np.asarray(v) for nm, v in rows.items()},
        roles={nm: roles[nm] for nm in rects},
        frames_seen=n,
        frames_kept=len(rows_t),
        duration_sec=float(rows_t[-1] - rows_t[0]) if len(rows_t) > 1 else 0.0,
    )


# ── The skin sampler matches the engine ─────────────────────────────────


def test_sampler_recovers_a_known_skin_colour():
    rng = np.random.default_rng(1)
    frame = np.empty((80, 80, 3), dtype=np.uint8)
    patch = np.repeat(np.repeat(SKIN_RGB[None, None, ::-1], 80, axis=0), 80, axis=1)
    frame[:, :] = np.clip(patch + rng.normal(0, 2, (80, 80, 3)), 0, 255).astype(np.uint8)

    s = skin.sample_region(skin.to_ycrcb(frame), (0, 0, 80, 80), step=1)
    assert s.valid
    # 8-bit chroma subsampling in the conversion costs a little accuracy.
    assert abs(s.r - SKIN_RGB[0]) < 3.0
    assert abs(s.g - SKIN_RGB[1]) < 3.0
    assert abs(s.b - SKIN_RGB[2]) < 3.0
    assert s.skin_fraction > 0.9


def test_glare_is_excluded():
    frame = np.full((40, 40, 3), 250, dtype=np.uint8)  # blown-out highlight
    s = skin.sample_region(skin.to_ycrcb(frame), (0, 0, 40, 40), step=1)
    assert not s.valid, "fully blown-out region must not produce a colour"


def test_non_skin_region_falls_back_to_all_pixels():
    frame = np.empty((40, 40, 3), dtype=np.uint8)
    frame[:, :] = BACKGROUND_RGB[::-1].astype(np.uint8)
    s = skin.sample_region(skin.to_ycrcb(frame), (0, 0, 40, 40), step=1)
    assert s.valid, "a control region must still report a colour"
    assert s.skin_fraction < skin.MIN_SKIN_FRACTION
    assert s.pixels > 0


def test_region_outside_the_frame_is_invalid():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert not skin.sample_region(skin.to_ycrcb(frame), (100, 100, 20, 20)).valid


# ── Frames to verdict ───────────────────────────────────────────────────


def test_frames_from_a_living_person_read_as_live():
    result = _run(seconds=22, inner_amp=0.012, outer_amp=0.011, seed=2)
    assert result.frames_kept > 500, result.summary()
    verdict = analyze_capture(result)
    assert verdict.verdict is Verdict.LIVE, verdict.summary()
    assert abs(verdict.metrics["reference_bpm"] - 72.0) < 5.0


def test_frames_with_a_pulseless_face_read_as_synthetic():
    """The neck keeps beating, the face does not — a face swap, in pixels."""
    result = _run(seconds=22, inner_amp=0.0, outer_amp=0.011, seed=3)
    verdict = analyze_capture(result)
    assert verdict.verdict is Verdict.SYNTHETIC, verdict.summary()
    assert FACE_PULSE_ABSENT in verdict.reason_codes
    assert verdict.blocked


def test_a_dead_clip_is_inconclusive_not_synthetic():
    result = _run(seconds=22, inner_amp=0.0, outer_amp=0.0, seed=4)
    verdict = analyze_capture(result)
    assert verdict.verdict is Verdict.INCONCLUSIVE, verdict.summary()
    assert not verdict.blocked


# ── Capture plumbing ────────────────────────────────────────────────────


def test_regions_share_one_timestamp_vector():
    result = _run(seconds=6, inner_amp=0.01, outer_amp=0.01, seed=5)
    lengths = {name: len(v) for name, v in result.region_rgb.items()}
    assert len(set(lengths.values())) == 1, lengths
    assert result.t.size == next(iter(lengths.values()))


def test_capture_survives_a_save_and_load_round_trip(tmp_path):
    result = _run(seconds=6, inner_amp=0.01, outer_amp=0.01, seed=6)
    path = result.save(tmp_path / "clip.npz")
    assert path.exists()

    loaded = CaptureResult.load(path)
    assert np.allclose(loaded.t, result.t)
    assert set(loaded.region_rgb) == set(result.region_rgb)
    for name in result.region_rgb:
        assert np.allclose(loaded.region_rgb[name], result.region_rgb[name])
    assert loaded.roles == result.roles


def test_saved_clip_contains_no_images(tmp_path):
    """Calibration clips must be shareable without sharing anyone's face."""
    result = _run(seconds=6, inner_amp=0.01, outer_amp=0.01, seed=7)
    data = np.load(result.save(tmp_path / "clip.npz"))
    for key in data.files:
        assert data[key].ndim <= 2, f"{key} looks like image data"


def test_both_inner_and_outer_regions_were_placed():
    rects = resolve_regions(FACE_BOX, FRAME_W, FRAME_H)
    roles = roles_for()
    placed = {roles[name] for name in rects}
    assert RegionRole.INNER in placed
    assert RegionRole.OUTER in placed, "no neck region: the detector has no reference"


# ── Face tracker ────────────────────────────────────────────────────────


def test_tracker_smooths_jitter_and_coasts_through_a_miss(monkeypatch):
    from detector.faces import FaceTracker

    tracker = FaceTracker(smoothing=0.5, dead_zone_px=0.0, max_coast_frames=3)
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    detections = [np.array([[100.0, 100.0, 200.0, 200.0] + [0.0] * 11])]
    monkeypatch.setattr(tracker, "_detect", lambda _frame: detections[0])
    first = tracker.update(frame)
    assert first is not None and first.as_tuple() == (100, 100, 200, 200)

    # A 20px jump should move the box by half that, not all of it.
    detections[0] = np.array([[120.0, 100.0, 200.0, 200.0] + [0.0] * 11])
    second = tracker.update(frame)
    assert second.x == 110, second.as_tuple()

    # Detection fails: coast on the last box, then give up.
    detections[0] = np.empty((0, 15), dtype=float)
    for expected_miss in (1, 2, 3):
        box = tracker.update(frame)
        assert box is not None and not box.detected
        assert box.frames_since_seen == expected_miss
        assert box.x == 110
    assert tracker.update(frame) is None
