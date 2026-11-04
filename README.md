<div align="center">

# Proof of Life

**A real-time payment fraud engine that proves a living, enrolled human authorised the transaction — not merely that the transaction looked normal.**

Team Polluters · Banking, Financial Services & FinTech · *Detect Payment Fraud in Real Time*

</div>

---

## The problem

Digital payment fraud reported in India reached **₹22,931 crore in 2025**, up **42×** from ₹551 crore in 2021 ([RBI][rbi]). Meanwhile **injection attacks rose 40% year on year**, and roughly **one in five biometric fraud attempts now involves a deepfake** ([Entrust 2026][entrust]).

The challenge asks for two things that normally trade against each other: *identify suspicious transactions* **and** *minimise false alerts for genuine customers*. With a single risk score you can have one or the other — tightening the threshold catches more fraud and declines more innocent people.

Worse, the fastest-growing fraud is invisible to a transaction model entirely:

> In authorised push payment fraud the transaction carries the right device, the right credentials and the right behaviour — because a real customer is being **impersonated, not simulated**. There is no feature in a payment record that encodes whether a human being is alive.

And the usual defence does not help. Liveness vendors detect **presentation** attacks: a photo, a mask, a screen held up to a lens. A deepfake injected through a virtual camera presents nothing to any lens at all.

## The idea

Two layers, and the relationship between them is the product:

**Layer 1 decides _how much proof_ a payment must carry.** Transaction risk from amount against the customer's own history, beneficiary age, velocity, hour, device change, geo-velocity.

**Layer 2 supplies proof that cannot be forged.** Pulse liveness, 1:1 face match, a server-chosen random challenge, synthetic-media detection, and a signature generated inside the customer's phone secure element.

```
                      ┌──────────────┐
  payment ──────────► │   LAYER 1    │ ── low risk ──────────────► settles silently
                      │  risk score  │                             (customer never contacted)
                      └──────┬───────┘
                             │ elevated risk
                             ▼
                      ┌──────────────┐     pass ────► payment RELEASED  (8 seconds, not a decline)
                      │   LAYER 2    │ ── suspect ──► analyst REVIEW
                      │ proof of life│     fail ────► payment BLOCKED
                      └──────┬───────┘
                             ▼
                   signed in the secure chip,
                   written to a hash-chained ledger
```

The commercial consequence: **an alert becomes an eight-second proof rather than a declined payment.** The cost of a false positive collapses, so the fraud model can run *more* aggressively and trouble *fewer* customers at the same time.

## The core insight

Every live face-swap tool composites rendered pixels inside a mask fitted to facial landmarks, and leaves everything outside it untouched. The cheeks are synthetic. **The neck is the real person.**

So we do not ask "is there a pulse?" — a swap can survive that, because a living attacker is usually driving the puppet. We ask:

> **Does the pulse in the face belong to the body it is attached to?**

```
  GENUINE CUSTOMER                      LIVE FACE SWAP

  neck  ∿∿∿∿∿∿∿∿∿∿∿∿  72 BPM           neck  ∿∿∿∿∿∿∿∿∿∿∿∿  72 BPM   ← attacker is alive
  face  ∿∿∿∿∿∿∿∿∿∿∿∿  72 BPM           face  ──────────────  no pulse ← rendered
        in phase, one heart                   deficit + phase incoherent
```

Four independent measurements, each separately reported:

| Signal | What it catches |
|---|---|
| **Pulse deficit** | A rendered face carrying no blood-volume signal, measured *at the rate the neck is beating at* |
| **Rate disagreement** | A swap leaking the source identity's pulse — two hearts in one frame |
| **Phase incoherence** | A swap that reproduced the rhythm but not the blood flow |
| **Illumination control** | Our own false-positive guard: off-skin regions must show *no* pulse |

This is why **image quality is irrelevant to us.** No face-swap tool models blood flow, at any resolution.

## Status

Honest accounting. This repository is mid-pivot: the verification engine is built and tested; the payment-fraud domain layer is being written.

| Component | State |
|---|---|
| rPPG pulse engine (C++, on-device) | **Built** — cross-compiled for `arm64-v8a` and `armeabi-v7a` |
| Face matching, 1:1 cosine with three-band decision | **Built** |
| Random challenge (blink / head turn) | **Built** |
| Secure-element key generation, attestation, ECDSA signing | **Built** |
| Backend verification pipeline + hash-chained ledger | **Built** — 75 tests passing |
| Multi-region pulse-consistency detector | **Built** — 29 tests passing, synthetic scenarios |
| Transaction-risk layer (Layer 1) | **In progress** |
| Fraud operations console | **In progress** |
| Domain model: payments, beneficiaries, step-up | **In progress** — currently carries the previous domain |
| Active illumination probe | Planned |
| Camera-provenance checks | Planned |

**What we do not claim.** There is no accuracy figure here, because we have not measured one on real captures yet. Detector thresholds are starting points marked as such in code — except one, `min_reference_snr_db`, which was set from measurement (see below). We would rather ship a calibrated gate and say so than quote a number we cannot defend.

## Repository layout

```
app/                     Android client
  src/main/cpp/          rPPG engine + face core (C++17, no OpenCV dependency in the core)
  src/main/java/...      CameraX capture, ML Kit tracking, challenge, Keystore signing
backend/                 FastAPI verification service
  app/services/          crypto · liveness · face_match · ledger · verification pipeline
  tests/                 75 tests
detector/                Synthetic-media detection  ← see detector/README.md
  dsp.py                 DSP layer over the app's own rPPG reference
  regions.py             Where to sample: inner (masked) · outer (neck) · control (off-skin)
  pulse_consistency.py   The detector and its risk fusion
  synth.py               Synthetic multi-region scenarios — no camera or GPU needed
src/                     React + Vite portal
tools/rppg/              Laptop test harness and the numpy reference for the C++ engine
SETUP.md                 Machine setup, toolchain versions, and the traps we hit
```

## Quick start

**Backend** — SQLite by default, no database setup needed.

```bash
cd backend && python -m pip install -r requirements.txt && python -m pytest -q
```

```bash
cd backend && python run.py
```

**Detector** — runs with no camera, GPU or network.

```bash
PYTHONPATH=. python -m pytest detector/tests -q
```

**Portal**

```bash
npm install && npm run dev
```

**Android** — needs the SDK, NDK 28.2.13676358 and JDK 21; see [SETUP.md](SETUP.md).

```bash
./gradlew :app:assembleDebug
```

## One threshold that is actually measured

The detector refuses to compare a face against a reference pulse it cannot trust. That gate exists because if the reference frequency is wrong, a real face looks deficient at it and an innocent customer is flagged.

Across 180 synthetic captures (52–118 BPM, five pulse amplitudes):

| Reference SNR | Rate error | Verdict |
|---|---|---|
| ≥ 0 dB | under 1 BPM, worst case 0.95 | reliable |
| −2 to 0 dB | under 1 BPM | marginal |
| below −2 dB | 10% of windows off by >5 BPM, worst case 73 BPM | noise |

Real pulses measured no lower than **+2.2 dB**; pulseless captures (photo, frozen frame, very poor light) no higher than **−2.7 dB**. `min_reference_snr_db` is set to **0.0 dB**, sitting in that gap with roughly 2 dB of margin either side.

Every other threshold needs calibration on real captured clips before deployment.

## Design decisions worth knowing

**One rPPG implementation, not two.** `detector/dsp.py` loads `tools/rppg/reference.py` — the numpy mirror of `app/src/main/cpp/rppg_core.cpp` that the C++ unit tests already cross-check against. A number measured by the detector is directly comparable with one the phone reported for the same person.

**POS cannot detect an illumination artifact, so the green channel does.** POS projects the colour trace onto the plane orthogonal to the skin-tone vector, cancelling achromatic illumination change *by construction*. That is exactly what you want for measuring a pulse and exactly what you cannot use to decide whether a periodic signal came from the light rather than from blood.

**`INCONCLUSIVE` is a first-class verdict.** A measurement that could not be made must never become an accusation. A cropped neck, poor light, or a lamp flickering at the subject's heart rate all produce `INCONCLUSIVE` or `SUSPECT`, never `SYNTHETIC`.

**One signal alone never blocks.** The fusion weights deliberately over-sum and the score clips at 100, so any single anomalous measurement reaches at most `SUSPECT`, while two independent ones reach `SYNTHETIC`. This is asserted by a test.

**Explicit weights, not a learned score.** A model-risk committee can read this, and a rejected customer can be told which measurement failed.

**What a hardware signature does and does not prove.** It proves *which device* signed, not that the video was real — a deepfake fed through a genuine signed app still produces a valid signature. Its real jobs are binding the session to one enrolled secure element, so the attacker must own the victim's phone, and making the verdict non-repudiable in the ledger.

## Security and privacy

- Face templates are encrypted at rest. **No images are stored or transmitted** — only an embedding.
- The ledger holds hashes, never personal data.
- The private key is generated inside the secure element and never leaves it. The key security level (`STRONGBOX` / `TEE` / `SOFTWARE`) is recorded in every signed proof, so weaker assurance is visible rather than hidden.
- Everything runs on the deploying institution's own infrastructure.
- Aligned with the Digital Personal Data Protection Act, 2023.

Never commit `backend/.env` or `local.properties`; both are git-ignored.

## Responsible use

This repository contains **defensive** detection only. The red-team setup — a face-swap tool driving a virtual camera — is documented in `SETUP.md` for testing our own system, lives outside this repository, and is never committed. Use your own face or a consenting participant's.

## References

- Verkruysse, Svaasand, Nelson — *Remote plethysmographic imaging using ambient light*, Optics Express, 2008
- de Haan, Jeanne — *Robust pulse rate from chrominance-based rPPG*, IEEE TBME, 2013
- Wang, den Brinker, Stuijk, de Haan — *Algorithmic principles of remote PPG*, IEEE TBME, 2017
- Ciftci, Demir, Yin — *FakeCatcher: detection of synthetic portrait videos using biological signals*, IEEE TPAMI, 2020
- Chen, Liu, Gao, Han — *MobileFaceNets*, 2018
- ISO/IEC 30107-3:2023 — biometric presentation attack detection: testing and reporting
- [Android Keystore hardware key attestation][attest]

[rbi]: https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=23810
[entrust]: https://www.entrust.com/resources/reports/identity-fraud-report
[attest]: https://developer.android.com/privacy-and-security/security-key-attestation

---

<div align="center">

**Team Polluters**

</div>
