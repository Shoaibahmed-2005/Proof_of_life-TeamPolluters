"""
DSP layer for the detector.

There is exactly one rPPG implementation in this repository, and this module
does not add a second one. The pipeline (POS -> detrend -> zero-phase
Butterworth band-pass -> Hann-windowed spectrum -> peak + SNR) lives in
``app/src/main/cpp/rppg_core.cpp`` and is mirrored sample-for-sample by
``tools/rppg/reference.py``, which the C++ unit tests cross-check against.
This module loads that reference module by path and builds on it, so a change
to the engine cannot silently leave the detector behind.

(When the pybind11 binding of rppg_core.cpp lands, only ``_load_reference()``
changes; everything below keeps working.)

What is added here:

* ``window_pulse()``  - whole-window analysis (the engine streams; the detector
  looks at one captured window at a time).
* ``green_signal()``  - the illumination-sensitive comparison channel. POS is
  deliberately invariant to achromatic illumination change, so it cannot be
  used to detect an illumination artifact; the green channel can.
* ``coefficient_at()`` - the complex Fourier coefficient at one frequency,
  which is what makes a *phase* comparison between two regions possible.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PY = _REPO_ROOT / "tools" / "rppg" / "reference.py"


def _load_reference() -> ModuleType:
    """Imports tools/rppg/reference.py without requiring it to be a package."""
    if "rppg_reference" in sys.modules:
        return sys.modules["rppg_reference"]
    if not _REFERENCE_PY.is_file():
        raise ImportError(
            f"The rPPG reference implementation is missing at {_REFERENCE_PY}. "
            "The detector deliberately has no copy of its own."
        )
    spec = importlib.util.spec_from_file_location("rppg_reference", _REFERENCE_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_REFERENCE_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rppg_reference"] = module
    spec.loader.exec_module(module)
    return module


_ref = _load_reference()

# Re-exported so callers never reach around this module into the reference.
Config = _ref.Config
butter_highpass = _ref.butter_highpass
butter_lowpass = _ref.butter_lowpass
filtfilt = _ref.filtfilt
pos = _ref.pos
detrend_linear = _ref.detrend_linear
band_spectrum = _ref.band_spectrum
peak_frequency = _ref.peak_frequency
snr_db = _ref.snr_db

__all__ = [
    "Config",
    "WindowPulse",
    "band_spectrum",
    "butter_highpass",
    "butter_lowpass",
    "coefficient_at",
    "detrend_linear",
    "filtfilt",
    "green_signal",
    "peak_frequency",
    "phase_difference_deg",
    "pos",
    "power_at",
    "snr_at",
    "snr_db",
    "window_pulse",
]


def _band_sections(cfg: Config) -> list:
    return [
        butter_highpass(cfg.band_low_hz, cfg.fs),
        butter_lowpass(cfg.band_high_hz, cfg.fs),
    ]


def _uniform_rgb(t: np.ndarray, rgb: np.ndarray, cfg: Config):
    """Resamples the irregular camera samples onto cfg.fs, as the engine does."""
    t = np.asarray(t, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    if t.ndim != 1 or rgb.ndim != 2 or rgb.shape[0] != t.size or rgb.shape[1] != 3:
        raise ValueError(f"Expected t[n] and rgb[n, 3]; got {t.shape} and {rgb.shape}")
    if t.size < 2:
        return None
    span = float(t[-1] - t[0])
    n = int(np.floor(span * cfg.fs)) + 1
    if n < max(int(round(cfg.pos_window_sec * cfg.fs)), 2 * int(cfg.fs)):
        return None
    grid = t[0] + np.arange(n) / cfg.fs
    channels = tuple(np.interp(grid, t, rgb[:, k]) for k in range(3))
    return channels, n, span


@dataclass(frozen=True)
class WindowPulse:
    """One region's pulse over one capture window."""

    valid: bool
    bpm: float = 0.0
    f0_hz: float = 0.0
    snr_db: float = -99.0
    span_sec: float = 0.0
    samples: int = 0
    half_width_hz: float = 0.0
    # Band-passed POS signal, and the illumination-sensitive green signal.
    signal: np.ndarray | None = None
    green: np.ndarray | None = None
    reason: str | None = None


def window_pulse(t: np.ndarray, rgb: np.ndarray, cfg: Config | None = None) -> WindowPulse:
    """
    Analyses one region's colour trace over the whole window.

    Identical maths to ``rppg.Engine.update()``, minus the streaming
    bookkeeping: resample -> POS -> linear detrend -> zero-phase 0.7-4 Hz
    band-pass -> Hann spectrum -> parabolic peak -> SNR over f0 and 2*f0.
    """
    cfg = cfg or Config()
    prepared = _uniform_rgb(t, rgb, cfg)
    if prepared is None:
        return WindowPulse(valid=False, reason="WINDOW_TOO_SHORT")
    (r, g, b), n, span = prepared

    pos_len = int(round(cfg.pos_window_sec * cfg.fs))
    sections = _band_sections(cfg)
    pad = min(n - 1, int(cfg.fs))

    signal = filtfilt(sections, detrend_linear(pos(r, g, b, pos_len)), pad)
    green = green_signal(g, cfg, n)

    freqs, power = band_spectrum(
        signal, cfg.fs, cfg.band_low_hz, cfg.band_high_hz, cfg.spectrum_step_bpm / 60.0
    )
    f0 = peak_frequency(freqs, power)
    half_width = 2.0 / (n / cfg.fs)
    return WindowPulse(
        valid=True,
        bpm=f0 * 60.0,
        f0_hz=f0,
        snr_db=snr_db(freqs, power, f0, half_width),
        span_sec=span,
        samples=int(n),
        half_width_hz=half_width,
        signal=signal,
        green=green,
    )


def green_signal(g: np.ndarray, cfg: Config | None = None, n: int | None = None) -> np.ndarray:
    """
    Band-passed, mean-normalised green channel.

    POS projects the colour trace onto the plane orthogonal to the skin-tone
    vector, which cancels achromatic illumination change by construction. That
    is exactly what you want for measuring a pulse, and exactly what you cannot
    use to decide whether a periodic signal came from the light rather than from
    blood. The green channel keeps that sensitivity, so comparing green power
    between skin and an off-skin control region separates the two.
    """
    cfg = cfg or Config()
    g = np.asarray(g, dtype=float)
    if g.size < 4:
        return np.zeros(0)
    mean = float(g.mean())
    normalised = g / mean if abs(mean) > 1e-9 else g
    pad = min(g.size - 1, int(cfg.fs)) if n is None else min(n - 1, int(cfg.fs))
    return filtfilt(_band_sections(cfg), detrend_linear(normalised), pad)


def coefficient_at(x: np.ndarray, fs: float, f_hz: float) -> complex:
    """
    Complex Hann-windowed Fourier coefficient of ``x`` at ``f_hz``.

    Uses the same Hann window as ``band_spectrum``, so magnitudes are directly
    comparable with the power values the engine reports. The band-pass is
    zero-phase (forward + backward), so the phase here is the signal's own
    phase and can be compared across regions.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4 or not np.isfinite(f_hz) or f_hz <= 0:
        return 0j
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    phasors = np.exp(-2j * np.pi * f_hz * np.arange(n) / fs)
    return complex(phasors @ (x * window))


def power_at(x: np.ndarray, fs: float, f_hz: float) -> float:
    """Hann-windowed power of ``x`` at ``f_hz``."""
    return float(abs(coefficient_at(x, fs, f_hz)) ** 2)


def snr_at(x: np.ndarray, cfg: Config, f0_hz: float, half_width_hz: float) -> float:
    """
    SNR in dB of ``x`` measured *at a given* f0 rather than at its own peak.

    This is the right question for a region under suspicion: not "what is the
    strongest rhythm here?" but "how strongly does this region carry the rate
    the body is actually beating at?".
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4 or not np.isfinite(f0_hz) or f0_hz <= 0:
        return -99.0
    freqs, power = band_spectrum(
        x, cfg.fs, cfg.band_low_hz, cfg.band_high_hz, cfg.spectrum_step_bpm / 60.0
    )
    return snr_db(freqs, power, f0_hz, half_width_hz)


def phase_difference_deg(a: np.ndarray, b: np.ndarray, fs: float, f_hz: float) -> float:
    """
    Absolute phase difference between two signals at ``f_hz``, in degrees (0-180).

    Physiology bounds this tightly. Pulse transit time between the forehead and
    the neck is on the order of tens of milliseconds, which at ~1.2 Hz is well
    under 20 degrees. A rendered face has no physical reason to agree, so a
    large difference is hard for a swap to fake without modelling blood flow.
    """
    ca, cb = coefficient_at(a, fs, f_hz), coefficient_at(b, fs, f_hz)
    if abs(ca) < 1e-18 or abs(cb) < 1e-18:
        return float("nan")
    delta = np.degrees(np.angle(ca) - np.angle(cb))
    return float(abs((delta + 180.0) % 360.0 - 180.0))
