# detector — synthetic media and camera-injection detection

The liveness engine on the phone asks *"is there a pulse?"*. A face swap can
survive that question: the attacker's own living face is usually driving the
puppet, and a weak pulse reading is indistinguishable from bad lighting.

This package asks the harder question:

> **Does the pulse in the face belong to the body it is attached to?**

## Why this works

Every live face-swap tool (DeepFaceLive, Deep-Live-Cam, roop, any inswapper
variant) composites rendered pixels inside a mask fitted to facial landmarks
and leaves everything outside that mask untouched. The cheeks are synthetic.
**The neck is the real person.** That asymmetry is measurable, and faking it
means modelling blood flow through a face that does not exist.

Four independent consequences, each measured and separately reported:

| Signal | What it catches | Metric |
|---|---|---|
| **Pulse deficit** | A rendered face carrying no blood-volume signal | SNR of the face regions *at the rate the neck is beating at* |
| **Rate disagreement** | A swap leaking the source identity's pulse — two hearts in one frame | `\|bpm_face − bpm_neck\|` |
| **Phase incoherence** | A swap that reproduced the rhythm but not the blood flow | Phase difference at the reference frequency |
| **Illumination control** | The detector's own false-positive guard | Green-channel power off-skin vs on-skin |

The key measurement is the first one, and the subtlety is *where* it is
measured. Asking "what is the strongest rhythm in the face?" is the wrong
question — a swap has plenty of periodic structure from compression and from
the driver's motion. Asking "how strongly does the face carry the rate the body
is actually beating at?" is the question a swap cannot pass.

## Design decisions worth knowing

**One rPPG implementation, not two.** `dsp.py` loads
`tools/rppg/reference.py` — the numpy mirror of `app/src/main/cpp/rppg_core.cpp`
that the C++ unit tests already cross-check against. The detector therefore
measures with exactly the same POS → detrend → zero-phase band-pass → Hann
spectrum pipeline as the phone, so a number here is comparable with a number
the phone reported for the same person. When the pybind11 binding of
`rppg_core.cpp` lands, only `_load_reference()` changes.

**POS cannot detect an illumination artifact, so the green channel does.**
POS projects the colour trace onto the plane orthogonal to the skin-tone
vector, which cancels achromatic illumination change *by construction*. That is
precisely what you want for measuring a pulse and precisely what you cannot use
to decide whether a periodic signal came from the light rather than from blood.
`green_signal()` keeps that sensitivity for the control comparison.

**`INCONCLUSIVE` is a first-class verdict.** A measurement that could not be
made must never become an accusation. A cropped neck, a still photo, poor
light, or light flickering at the subject's heart rate all produce
`INCONCLUSIVE` or `SUSPECT`, never `SYNTHETIC`. Commercially this is the whole
point: a false "synthetic" on a real customer costs far more than a review.

**One signal alone never blocks.** The fusion weights deliberately over-sum
and the score clips at 100, so any single anomalous measurement reaches at most
`SUSPECT` while two independent ones reach `SYNTHETIC`. This is asserted by
`test_a_single_signal_never_reaches_synthetic_on_its_own`.

**Explicit weights, not a learned score.** A model-risk committee can read
this, and a rejected customer can be told which measurement failed. A learned
detector belongs elsewhere in the stack as a corroborating input, never as the
thing that decides.

## Layout

| File | Contents |
|---|---|
| `dsp.py` | DSP layer over the app's own rPPG reference; whole-window analysis, green channel, complex coefficients for phase |
| `regions.py` | Where to sample: `INNER` (inside the swap mask), `OUTER` (neck/collar — the real person), `CONTROL` (off-skin) |
| `skin.py` | Mean skin colour per region, a constant-for-constant port of the engine's `meanSkinRgb()` |
| `faces.py` | Face tracking: YuNet detection, EMA smoothing, and coasting through missed frames |
| `capture.py` | The camera front end — video in, a verdict out, plus the CLI |
| `pulse_consistency.py` | The detector and its risk fusion |
| `synth.py` | Synthetic multi-region scenarios — no camera, GPU or OBS needed |
| `tests/` | 41 tests; more cover refusing-to-accuse than cover the attacks |

## Running it on a camera

```bash
python -m detector.capture --list-cameras
```

```bash
python -m detector.capture --source 0 --seconds 20
```

```bash
python -m detector.capture --source 1 --seconds 20 --save clips/swap_01.npz
```

A virtual camera appears in `--list-cameras` exactly like a physical webcam —
which is itself the point: the application layer cannot tell them apart by
reading frames, so the defence has to be physiological.

`--save` writes the sampled time series, **not** the video. Those `.npz` files
are what threshold calibration runs on, and they contain no images, so a
capture can be shared for analysis without sharing anyone's face. Re-analyse
one with `--load`.

### Two notes on the capture loop

**Box stability matters more than detector accuracy.** A face box that jitters
drags the sampling rectangles across different skin and injects motion noise
straight into the pulse band — the detector would then measure the tracker
rather than the person. Hence the EMA smoothing and the dead zone in
`faces.py`.

**Frames are dropped, never interpolated.** All regions must share one
timestamp vector, so only frames where every surviving region was valid are
kept. Carrying a stale sample forward would invent signal that was never
measured. A region measurable in fewer than 90% of frames is dropped entirely
and reported, rather than silently compared against a complete one.

OpenCV 5 removed `CascadeClassifier`, so the Haar cascades vendored in
`sdk/etc/haarcascades/` for the Android build are not usable from Python. The
front end uses YuNet instead, which gives markedly steadier boxes. The ~230 KB
model downloads once into `detector/models/` and is not committed.

## Usage

```python
from detector import analyze_regions
from detector.regions import resolve_regions, roles_for

rects = resolve_regions(face_box=(x, y, w, h), image_width=1280, image_height=720)
# sample mean skin RGB per rect per frame -> {name: rgb[n, 3]}, shared timestamps t
result = analyze_regions(t, region_rgb, roles_for())

print(result.summary())      # "SYNTHETIC risk=85/100 [FACE_PULSE_ABSENT, ...]"
result.verdict               # LIVE | SUSPECT | SYNTHETIC | INCONCLUSIVE
result.reason_codes          # for the ledger and the API
result.reasons               # plain language, for the operations console
result.metrics               # every number behind the verdict
result.regions               # per-region evidence
```

## Running the tests

```bash
PYTHONPATH=. python -m pytest detector/tests -q
```

## Thresholds: what is measured and what is still a guess

`min_reference_snr_db = 0.0` **is** measured. Across 180 synthetic captures
(52–118 BPM, five pulse amplitudes) the reference rate error stays under 1 BPM
at every SNR at or above 0 dB and collapses below −2 dB (10% of windows off by
more than 5 BPM, worst case 73 BPM — a pure noise peak). Real pulses measured
no lower than +2.2 dB; pulseless captures no higher than −2.7 dB. 0 dB sits in
that gap with about 2 dB of margin either side.

**Every other threshold in `Thresholds` is a starting point, not a measured
value.** The deficit, rate, phase and bleed thresholds need to be set from real
captured clips — roughly 30 genuine and 30 swapped, through the actual capture
path, under the actual demo lighting. Synthetic scenarios prove the detector's
*logic*, its reason codes and its refusal to accuse on bad input; they cannot
calibrate it, because a synthetic "deepfake" is only as good as the assumptions
baked into it. Real DeepFaceLive output is messier: it leaks some of the
driver's pulse through the blend edge, and compression reintroduces periodic
structure.

## Not yet built

This package covers the physiological pillar and its camera front end. Still
to come: the active screen-flash probe, blend-boundary and spectral forensics,
camera provenance / virtual-device fingerprinting, and the cross-detector risk
fusion.

**It has not yet been calibrated on real captures.** The pixel-level tests
prove the chain works end to end on rendered frames; they do not tell you how
it behaves on a real face in real light against a real face-swap tool. That
needs roughly 30 genuine and 30 swapped clips through the actual capture path,
collected with `--save` and then used to set the deficit, rate, phase and bleed
thresholds.
