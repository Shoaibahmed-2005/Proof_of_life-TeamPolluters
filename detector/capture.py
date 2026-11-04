"""
The camera front end: video in, a verdict out.

This is what turns the detector from something that passes synthetic tests
into something that can watch a real camera. The chain is:

    frame -> face box (smoothed) -> region rectangles -> mean skin RGB
          -> per-region time series -> analyze_regions() -> verdict

Run it against a webcam, a virtual camera, or a recorded file:

    python -m detector.capture --list-cameras
    python -m detector.capture --source 0 --seconds 20
    python -m detector.capture --source 1 --seconds 20 --save clips/swap_01.npz
    python -m detector.capture --video clips/attack.mp4

``--save`` writes the sampled time series, not the video. Those files are what
threshold calibration runs on, and they carry no images — a capture can be
shared for analysis without sharing anyone's face.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detector import dsp, skin
from detector.faces import FaceTracker
from detector.pulse_consistency import PulseConsistencyResult, Thresholds, analyze_regions
from detector.regions import DEFAULT_LAYOUT, RegionRole, resolve_regions, roles_for

__all__ = ["CaptureResult", "capture", "analyze_capture", "list_cameras"]


@dataclass
class CaptureResult:
    """Everything one capture produced."""

    t: np.ndarray
    region_rgb: dict[str, np.ndarray]
    roles: dict[str, RegionRole]
    frames_seen: int = 0
    frames_kept: int = 0
    face_lost_frames: int = 0
    duration_sec: float = 0.0
    dropped_regions: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames_kept / self.duration_sec if self.duration_sec > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.frames_kept}/{self.frames_seen} frames kept over "
            f"{self.duration_sec:.1f}s ({self.fps:.1f} fps), "
            f"face lost on {self.face_lost_frames}, "
            f"regions: {', '.join(sorted(self.region_rgb))}"
        )

    def save(self, path: str | Path) -> Path:
        """Writes the time series to an .npz. No images are stored."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"t": self.t}
        for name, rgb in self.region_rgb.items():
            payload[f"rgb__{name}"] = rgb
        np.savez_compressed(
            path,
            frames_seen=self.frames_seen,
            frames_kept=self.frames_kept,
            face_lost_frames=self.face_lost_frames,
            duration_sec=self.duration_sec,
            **payload,
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "CaptureResult":
        data = np.load(Path(path))
        region_rgb = {
            key[len("rgb__"):]: data[key] for key in data.files if key.startswith("rgb__")
        }
        all_roles = roles_for()
        return CaptureResult(
            t=data["t"],
            region_rgb=region_rgb,
            roles={n: all_roles[n] for n in region_rgb if n in all_roles},
            frames_seen=int(data["frames_seen"]),
            frames_kept=int(data["frames_kept"]),
            face_lost_frames=int(data["face_lost_frames"]),
            duration_sec=float(data["duration_sec"]),
        )


def list_cameras(max_index: int = 8) -> list[tuple[int, str]]:
    """
    Finds openable camera indices, with names where the platform offers them.

    Use this to identify a virtual camera: on Windows, OBS Virtual Camera
    appears in this list exactly like a physical webcam, which is itself the
    point — the application layer cannot tell them apart by reading frames.
    """
    import cv2

    names: dict[int, str] = {}
    try:  # optional; present in some environments
        from pygrabber.dshow_graph import FilterGraph

        for i, name in enumerate(FilterGraph().get_input_devices()):
            names[i] = name
    except Exception:
        pass

    found: list[tuple[int, str]] = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append((index, names.get(index, "camera")))
        finally:
            cap.release()
    return found


def _open(source: int | str, want_fps: float = 30.0, want_width: int = 1280):
    import cv2

    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            # Many USB webcams default to uncompressed YUY2, which the bus can
            # only sustain at 15-16 fps. MJPG usually unlocks the full frame
            # rate. Frame rate matters here: the pulse is sampled per frame, so
            # half the frames is half the evidence in the same window.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(want_width * 9 / 16))
            cap.set(cv2.CAP_PROP_FPS, want_fps)
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    return cap


class BlankCameraError(RuntimeError):
    """The camera streams, but the frames carry no image."""


def _check_frames_have_content(frames: list[np.ndarray], min_std: float = 2.0) -> None:
    """
    Rejects a stream of flat frames before wasting a whole capture on it.

    A covered lens, a camera kill switch or a privacy setting all produce
    frames that arrive on time and contain nothing. Without this check the
    capture runs its full length and then blames the user's face or lighting,
    which sends them looking in exactly the wrong place.
    """
    import cv2

    if not frames:
        return
    spreads = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).std()) for f in frames]
    if max(spreads) < min_std:
        mean_luma = float(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY).mean())
        raise BlankCameraError(
            f"The camera is delivering blank frames (pixel spread {max(spreads):.2f}, "
            f"mean brightness {mean_luma:.0f}). This is not a lighting or framing "
            "problem — there is no image in the stream at all.\n"
            "Check, in order: a physical privacy shutter over the lens; a camera "
            "kill switch (often an Fn key); Windows Settings > Privacy & security > "
            "Camera > 'Let desktop apps access your camera'; and any other app "
            "holding the camera."
        )


def capture(
    source: int | str = 0,
    seconds: float = 20.0,
    step: int = 2,
    min_region_coverage: float = 0.9,
    show: bool = False,
    progress: bool = True,
) -> CaptureResult:
    """
    Samples region colour traces from a camera or a video file.

    Args:
        source: camera index, or a path to a video file.
        seconds: how long to record. The engine needs a 10-second window, so
            anything under about 15 seconds leaves no margin for a lost face.
        step: pixel sampling stride, matching the phone.
        min_region_coverage: a region must be measurable in at least this
            fraction of tracked frames to be used at all. A neck that is only
            half in frame is dropped rather than compared against a full one.
        show: display the frame with region boxes drawn (needs a desktop).
        progress: print a progress line while recording.

    Returns:
        A :class:`CaptureResult` whose regions share one timestamp vector, as
        :func:`analyze_regions` requires.
    """
    import cv2

    cap = _open(source)
    tracker = FaceTracker()
    layout_roles = roles_for()

    samples: list[tuple[float, dict[str, skin.RegionSample]]] = []
    frames_seen = 0
    face_lost = 0
    warmup: list[np.ndarray] = []
    is_file = not isinstance(source, int)

    if not is_file:
        # Webcams need a moment for auto-exposure to settle; sampling before
        # that would put a brightness ramp straight into the pulse band.
        for _ in range(12):
            ok, frame = cap.read()
            if ok:
                warmup.append(frame)
        try:
            _check_frames_have_content(warmup)
        except BlankCameraError:
            cap.release()
            raise

    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file, or camera went away
            frames_seen += 1
            now = time.perf_counter() - started
            if is_file:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    now = pos_ms / 1000.0
            if now > seconds:
                break

            box = tracker.update(frame)
            if box is None:
                face_lost += 1
                if show:
                    cv2.imshow("Proof of Life - capture", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue
            if not box.detected:
                face_lost += 1

            rects = resolve_regions(box.as_tuple(), frame.shape[1], frame.shape[0])
            if not rects:
                continue

            ycrcb = skin.to_ycrcb(frame)
            frame_samples = {
                name: skin.sample_region(ycrcb, rect, step) for name, rect in rects.items()
            }
            samples.append((now, frame_samples))

            if progress and frames_seen % 30 == 0:
                sys.stdout.write(
                    f"\r  {now:5.1f}s / {seconds:.0f}s   frames {frames_seen}   "
                    f"regions {len(rects)}   face lost {face_lost}   "
                )
                sys.stdout.flush()

            if show:
                for name, (x, y, w, h) in rects.items():
                    colour = {
                        RegionRole.INNER: (60, 100, 232),
                        RegionRole.OUTER: (90, 180, 60),
                        RegionRole.CONTROL: (160, 160, 160),
                    }[layout_roles[name]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.imshow("Proof of Life - capture", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if progress:
            sys.stdout.write("\n")

    duration = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
    if not samples:
        return CaptureResult(
            t=np.zeros(0), region_rgb={}, roles={},
            frames_seen=frames_seen, face_lost_frames=face_lost, duration_sec=0.0,
        )

    # Which regions were measurable often enough to be trusted?
    tracked = len(samples)
    counts: dict[str, int] = {}
    for _, per_frame in samples:
        for name, s in per_frame.items():
            if s.valid:
                counts[name] = counts.get(name, 0) + 1
    kept_regions = [n for n, c in counts.items() if c / tracked >= min_region_coverage]
    dropped = sorted(set(counts) - set(kept_regions))

    # All regions must share one timestamp vector, so keep only the frames
    # where every surviving region was valid. Carrying a stale value forward
    # would invent signal that was never measured.
    rows_t: list[float] = []
    rows: dict[str, list[tuple[float, float, float]]] = {n: [] for n in kept_regions}
    for t_i, per_frame in samples:
        if not all(n in per_frame and per_frame[n].valid for n in kept_regions):
            continue
        rows_t.append(t_i)
        for n in kept_regions:
            s = per_frame[n]
            rows[n].append((s.r, s.g, s.b))

    all_roles = roles_for()
    return CaptureResult(
        t=np.asarray(rows_t, dtype=float),
        region_rgb={n: np.asarray(v, dtype=float) for n, v in rows.items() if v},
        roles={n: all_roles[n] for n in kept_regions},
        frames_seen=frames_seen,
        frames_kept=len(rows_t),
        face_lost_frames=face_lost,
        duration_sec=duration,
        dropped_regions=dropped,
    )


def analyze_capture(
    result: CaptureResult,
    cfg: dsp.Config | None = None,
    thresholds: Thresholds | None = None,
) -> PulseConsistencyResult:
    """Runs the detector over a capture."""
    return analyze_regions(result.t, result.region_rgb, result.roles, cfg, thresholds)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m detector.capture",
        description="Capture region pulse traces from a camera and run the detector.",
    )
    parser.add_argument("--source", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="video file instead of a camera")
    parser.add_argument("--seconds", type=float, default=20.0, help="recording length")
    parser.add_argument("--save", type=str, default=None, help="write the time series to an .npz")
    parser.add_argument("--load", type=str, default=None, help="analyse a saved .npz instead")
    parser.add_argument("--show", action="store_true", help="show the frame with region boxes")
    parser.add_argument("--list-cameras", action="store_true", help="list openable cameras and exit")
    args = parser.parse_args(argv)

    if args.list_cameras:
        found = list_cameras()
        if not found:
            print("No cameras could be opened.")
            return 1
        print("Cameras:")
        for index, name in found:
            print(f"  --source {index}    {name}")
        return 0

    if args.load:
        result = CaptureResult.load(args.load)
        print(f"Loaded {args.load}")
    else:
        source: int | str = args.video if args.video else args.source
        print(f"Capturing from {source!r} for {args.seconds:.0f}s ...")
        try:
            result = capture(source=source, seconds=args.seconds, show=args.show)
        except BlankCameraError as exc:
            print(f"\n{exc}")
            return 2

    print(result.summary())
    if result.dropped_regions:
        print(f"  dropped (not measurable often enough): {', '.join(result.dropped_regions)}")
    if args.save and not args.load:
        print(f"  saved to {result.save(args.save)}")

    if result.t.size < 2:
        print("\nNothing measurable was captured. Is a face in frame and lit?")
        return 1

    verdict = analyze_capture(result)
    print("\n" + "=" * 64)
    print(f"  {verdict.summary()}")
    print("=" * 64)
    for reason in verdict.reasons:
        print(f"  - {reason}")
    print("\n  Measurements:")
    for key, value in verdict.metrics.items():
        print(f"    {key:30s} {value:8.2f}")
    print("\n  Per region:")
    for region in sorted(verdict.regions, key=lambda r: (r.role.value, r.name)):
        state = "ok     " if region.valid else "invalid"
        print(
            f"    {region.role.value:8s} {region.name:18s} {state} "
            f"bpm={region.bpm:6.1f}  snr={region.snr_db:6.2f} dB  "
            f"at_ref={region.snr_at_reference_db:6.2f} dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
