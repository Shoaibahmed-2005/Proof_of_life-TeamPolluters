"""
Face tracking for the capture loop.

The detector's regions are all fractions of a face bounding box, so the box is
the thing everything else depends on. Two properties matter more than raw
detection accuracy:

**Stability.** A box that jitters frame to frame drags the sampling rectangles
across different skin, which injects motion noise straight into the pulse band
— the detector would then be measuring the tracker rather than the person. So
the box is smoothed with an exponential moving average, and movements below a
dead zone are ignored entirely.

**Continuity.** A detector that misses one frame in twenty would punch holes in
the time series, and a gap longer than a second makes the engine restart the
scan. So a missed detection reuses the last known box for a short grace period
rather than dropping the frame.

Detection uses **YuNet**, the DNN face detector shipped with OpenCV. Note that
OpenCV 5 removed `CascadeClassifier` and the Haar cascades from the Python
bindings, so the cascades vendored in `sdk/etc/haarcascades/` for the Android
build are not usable here. That turned out to be an improvement: YuNet gives
markedly steadier boxes than Haar, and steadiness is exactly what this loop
needs. The model is ~230 KB and is fetched once into `detector/models/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FaceBox", "FaceTracker", "ensure_model"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _REPO_ROOT / "detector" / "models"
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


@dataclass(frozen=True)
class FaceBox:
    """A face rectangle for one frame."""

    x: int
    y: int
    w: int
    h: int
    detected: bool          # True if found this frame, False if coasting
    frames_since_seen: int = 0
    score: float = 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def ensure_model(path: Path | None = None) -> Path:
    """
    Returns the YuNet model path, downloading it once if absent.

    Kept out of git: it is a third-party binary that is trivially re-fetched.
    """
    target = Path(path) if path else MODEL_DIR / MODEL_NAME
    if target.is_file() and target.stat().st_size > 100_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if len(data) < 100_000:
            raise OSError(f"downloaded file is only {len(data)} bytes")
        target.write_bytes(data)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as guidance
        raise FileNotFoundError(
            f"The face-detection model is missing and could not be downloaded ({exc}).\n"
            f"Fetch it manually to {target}:\n  {MODEL_URL}"
        ) from exc
    return target


class FaceTracker:
    """
    Finds one face per frame and keeps its box steady.

    Args:
        smoothing: EMA weight for a new detection, 0–1. Lower is steadier and
            slower to follow real movement. 0.35 keeps the box calm without
            visibly lagging a person settling into frame.
        dead_zone_px: ignore movements smaller than this, so a box that is
            really still stops shivering by a pixel or two per frame.
        max_coast_frames: how long to reuse the last box when detection fails.
            At 30 fps, 15 frames is half a second — shorter than the engine's
            one-second gap limit, so a brief miss never restarts the scan.
        score_threshold: YuNet confidence floor.
        model_path: override the model location.
    """

    def __init__(
        self,
        # A seated subject barely moves, so the box should barely move. Heavy
        # smoothing lags a person settling into frame by a fraction of a
        # second and is worth it: residual box wander drags the sampling
        # rectangles across different skin, and that drift is far larger than
        # the one-count signal being measured.
        smoothing: float = 0.18,
        dead_zone_px: float = 2.5,
        max_coast_frames: int = 20,
        max_jump_fraction: float = 0.45,
        # 0.7 is a sensible floor when the cost of a wrong face is high. Here
        # the subject is cooperative and framed deliberately, while a rejected
        # frame is a hole in the pulse signal — so the cost is reversed. At
        # 0.7 a real, well-framed face was being dropped on most frames of a
        # 24-second capture; borderline detections are worth keeping, and the
        # EMA smoothing absorbs the occasional poor box.
        score_threshold: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        import cv2

        self._cv2 = cv2
        model = ensure_model(model_path)
        self._detector = cv2.FaceDetectorYN.create(
            str(model), "", (320, 320), score_threshold, 0.3, 5000
        )
        self._input_size: tuple[int, int] | None = None
        self.smoothing = float(smoothing)
        self.dead_zone_px = float(dead_zone_px)
        self.max_coast_frames = int(max_coast_frames)
        self.max_jump_fraction = float(max_jump_fraction)
        self._state: np.ndarray | None = None  # smoothed [x, y, w, h]
        self._misses = 0

    @staticmethod
    def _centre(box: np.ndarray) -> np.ndarray:
        return np.array([box[0] + box[2] / 2.0, box[1] + box[3] / 2.0])

    def reset(self) -> None:
        self._state = None
        self._misses = 0

    def _detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns an (N, 15) array of YuNet detections, possibly empty."""
        height, width = frame_bgr.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def update(self, frame_bgr: np.ndarray) -> FaceBox | None:
        """
        Returns the face box for this frame, or None if there is no usable track.

        A returned box with ``detected=False`` is the last known position being
        reused because detection missed this frame.
        """
        faces = self._detect(frame_bgr)

        if len(faces) == 0:
            self._misses += 1
            if self._state is None or self._misses > self.max_coast_frames:
                return None
            x, y, w, h = (int(round(v)) for v in self._state)
            return FaceBox(x, y, w, h, detected=False, frames_since_seen=self._misses)

        if self._state is None:
            # First acquisition only: the largest face is the subject, being
            # nearest the camera.
            best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        else:
            # Afterwards, follow CONTINUITY rather than size. Picking the
            # largest every frame lets the track flip between the real face
            # and a lower-confidence false positive, and each flip throws the
            # sampling regions onto completely different pixels. That showed
            # up as the box wandering 395px while the subject sat still, with
            # region brightness swinging 150 counts — against a pulse worth
            # about one count.
            previous = self._centre(self._state)
            best = min(
                faces, key=lambda f: float(np.hypot(*(self._centre(f) - previous)))
            )
            jump = float(np.hypot(*(self._centre(best) - previous)))
            if jump > self.max_jump_fraction * float(self._state[2]):
                # Nothing near where the face was. A real head cannot move
                # that far between frames, so treat it as a miss and coast.
                self._misses += 1
                if self._misses > self.max_coast_frames:
                    self._state = None
                    return None
                x, y, w, h = (int(round(v)) for v in self._state)
                return FaceBox(x, y, w, h, detected=False, frames_since_seen=self._misses)

        measured = np.array(best[:4], dtype=float)
        score = float(best[14]) if len(best) > 14 else 0.0
        self._misses = 0

        if self._state is None:
            self._state = measured
        else:
            delta = measured - self._state
            # Dead zone per component, so a still face stops shivering.
            delta[np.abs(delta) < self.dead_zone_px] = 0.0
            self._state = self._state + self.smoothing * delta

        x, y, w, h = (int(round(v)) for v in self._state)
        return FaceBox(x, y, w, h, detected=True, frames_since_seen=0, score=score)
