# SETUP — this machine

Audited state of the development machine, what was installed, and what is still
missing. Written 2026-11-04, build chains verified 2026-11-05.

## Both chains verified

**Red team** — `onnxruntime-gpu` 1.26.0 sees `CUDAExecutionProvider`, so the
face swap runs on the RTX 3050 rather than the CPU. `insightface` 0.7.3
compiled. Both model weights byte-verified.

**Android** — `BUILD SUCCESSFUL`, APK produced, and the native rPPG/face core
cross-compiled for both target ABIs:

```
app\build\outputs\apk\debug\app-debug.apk        191.2 MB
arm64-v8a/libsentinelhard.so                       3901 KB
armeabi-v7a/libsentinelhard.so                     3105 KB
```

`libsentinelhard.so` is `rppg_core.cpp` + `face_core.cpp` through the NDK. The
debug APK is large because it carries OpenCV and both ABIs unminified; a
release build with `isMinifyEnabled` is far smaller. Irrelevant for the demo.

Rebuild and re-verify with:

```bash
./gradlew.bat :app:assembleDebug --no-daemon --console=plain
```

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, 6 GB, driver 582.05 |
| iGPU | AMD Radeon 680M |
| CPU | AMD Ryzen 7 170 |
| RAM | 15.2 GB |
| Disk | C: 204 GB free · D: 154 GB free |

Two consequences that shape the demo:

* **15.2 GB RAM rules out the Android-emulator attack scene.** DeepFaceLive/
  Deep-Live-Cam + OBS + Postgres + the detector + the portal + a browser
  already saturate it. The red team runs on the laptop; the phone runs the real
  app on real hardware.
* **6 GB VRAM is sufficient but not roomy.** The swap tool takes ~2 GB and the
  detector ~1.5 GB. Close OBS previews and spare browser tabs while demoing.

**The GPU needs no further setup.** `nvidia-smi` works and the driver is
current. Do **not** install the CUDA Toolkit — Torch and the swap tools ship
their own CUDA runtimes, and adding the toolkit on top causes version
confusion rather than fixing anything.

## Installed during setup

| Tool | Version | Note |
|---|---|---|
| Node.js LTS | 26.13.0 | required by the Vite portal |
| CMake | 4.4.4 | for the pybind11 binding of `rppg_core.cpp` |
| ffmpeg | — | dataset capture, MJPEG. Aliases `ffmpeg`/`ffplay`/`ffprobe` |
| OBS Studio | — | virtual camera for the red-team scenes |
| Microsoft OpenJDK | 21.0.13 | Gradle daemon toolchain (`toolchainVersion=21`) |

### Two winget gotchas on this machine

**Exit 231, "specify one of them using the --source option".** More than one
winget source is configured, so every install needs `--source winget`.

**Exit 231 with `0x80072ee7`.** That is `ERROR_INTERNET_NAME_NOT_RESOLVED` — a
DNS dropout, not a bad package id. This Wi-Fi drops its name resolution
intermittently, and it killed three separate downloads during setup. Retry
rather than hunting for a different package. Anything long-running should be
wrapped in a retry loop, and anything large should resume (`curl -C -`) rather
than restart.

**Do not rely on the network during the live demo.** The three demo scenes do
not need it — the phone reaches the backend over the LAN, and the detector and
swap tool both run locally on the GPU. Install every dependency beforehand.

## Already present

| Tool | Note |
|---|---|
| PostgreSQL 18 | **installed and running** (service `postgresql-x64-18`). Not on PATH — add `C:\Program Files\PostgreSQL\18\bin` |
| Python 3.12.10 | use this one |
| Python 3.15.0 | the `py` launcher default. **Do not use it** — no wheels for torch, mediapipe or onnxruntime-gpu. Always `py -3.12` |
| Git | `C:\Program Files\Git` |

## Android build chain

Nothing Android-related was present at the start: no `studio64.exe`, no JDK, no
`JAVA_HOME`, no SDK at `%LOCALAPPDATA%\Android\Sdk`, no Gradle cache.

Versions are taken from the project's own build files, not chosen:

| Requirement | Value | Source |
|---|---|---|
| compileSdk / targetSdk | 35 | `app/build.gradle.kts` |
| minSdk | 24 | `app/build.gradle.kts` |
| NDK | 28.2.13676358 (pinned) | `app/build.gradle.kts` |
| SDK CMake | 3.22.1 | `externalNativeBuild` |
| AGP | 9.3.1 | `gradle/libs.versions.toml` |
| Gradle | 9.6.1 | `gradle-wrapper.properties` |
| JDK toolchain | 21 | `gradle-daemon-jvm.properties` |
| ABIs | arm64-v8a, armeabi-v7a | `abiFilters` |

**The two CMakes are different things and both are needed.** SDK `cmake;3.22.1`
builds the app's native code and is pinned by the project; the system CMake
4.4.4 is for the pybind11 binding of `rppg_core.cpp`. Do not conflate them.

**OpenCV needs no download.** The Android SDK build is vendored in the repo at
`sdk/`, wired in as the `:opencv` Gradle module by `settings.gradle.kts`.

SDK packages are installed with the **current** CLI (no Studio first-run
wizard). Command-line tools build 16111833, discovered from
`repository2-3.xml` rather than guessing a build number.

### Two traps that cost real time here

**`sdkmanager` is deprecated and fails silently.** Command-line tools 23.0.0
replaced it with an `android` binary, and package naming changed from
semicolons to slashes:

```bash
# old - prints "Package platforms not found" and then EXITS 0
sdkmanager --install "platforms;android-35"
# current
android sdk install platforms/android-35
```

It splits on the semicolon, reports both halves as not found, and returns
success. Only `platform-tools` installed, and the job looked clean. **Never
trust the exit code here — check for the actual file.** Verify with:

```bash
ls "$LOCALAPPDATA/Android/Sdk/ndk/28.2.13676358/ndk-build.cmd"
```

**`local.properties` must have no BOM.** Windows PowerShell 5.1 writes a UTF-8
BOM when given `-Encoding utf8`. Gradle reads `.properties` as ISO-8859-1, so
the BOM turns the first key into `\ufeffsdk.dir`, which never matches
`sdk.dir` — and the error you get is the thoroughly misleading
`SDK location not found`, pointing at an SDK that is installed perfectly.

Also note that a **single** backslash is an escape character in Java
properties, so `C:\Users\...` silently loses its separators. Use forward
slashes:

```properties
sdk.dir=C:/Users/saras/AppData/Local/Android/Sdk
```

## Still missing

**Visual Studio 2022 Build Tools**, "Desktop development with C++" workload.
Now known to be **required**, not optional, for two independent reasons:

1. `insightface==0.7.3` (a Deep-Live-Cam dependency) publishes **only a source
   distribution** on PyPI and builds a Cython extension
   (`face3d/mesh/cython/mesh_core_cython.pyx`). Without MSVC it cannot install
   at all.
2. The pybind11 binding of `rppg_core.cpp`.

Install with the workload selected explicitly:

```bash
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --source winget --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Do not be misled by the first failure mode here. The pip error read
`Failed to build 'insightface' when installing build dependencies` with an
underlying `SSLError`, which looks like a compiler problem but was actually the
DNS dropout preventing the *build dependencies* from downloading. Both causes
are real and both have to be fixed; fixing only the network would have hit the
compiler wall next.

## Backend

`backend/.env` was created from `.env.example`:

* `SECRET_KEY`, `TEMPLATE_KEY` — generated, written directly to the file.
* `ISSUER_KEY_PEM` — left empty on purpose. The backend generates a P-256
  issuer key into `backend/data/issuer_key.pem` on first start; this keeps a
  multi-line PEM out of the env file.
* `DATABASE_URL` — **SQLite is active.** The Postgres URL is present but
  commented out.

**Why not Postgres yet.** The repo's tests and scripts assume SQLite, and
switching buys nothing today. Postgres belongs with the schema rewrite for the
banking/insurance pivot, where the schema changes anyway. It is one uncommented
line when that lands.

Never commit `.env`. It is already covered by `.gitignore` line 37.

Verified: **75 backend tests pass.**

```bash
cd backend && python -m pytest -q
```

## Detector

New `detector/` package — see [detector/README.md](detector/README.md).
Verified: **29 tests pass.**

```bash
PYTHONPATH=. python -m pytest detector/tests -q
```

Needs no GPU, camera or OBS: it runs against synthetic multi-region scenarios.

## Red team

**Verified working.** `onnxruntime-gpu` 1.26.0 reports:

```
providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

so the RTX 3050 is visible and the swap will run on the GPU, not the CPU. This
is worth re-checking after any driver or package change: on CPU the swap drops
to a few frames per second and the demo is unusable.

`insightface` 0.7.3 compiled successfully once the MSVC C++ workload was
present. Model weights verified: `inswapper_128_fp16.onnx` 264.8 MB,
`gfpgan-1024.onnx` 130.2 MB.

Lives in **`D:\redteam\`, outside this repository.** Nothing from it is ever
committed, and no model weights belong in git.

* **Deep-Live-Cam** — cloned to `D:\redteam\Deep-Live-Cam`, with its own
  `.venv` on Python 3.12.
* Model weights (`inswapper_128_fp16.onnx`, `gfpgan-1024.onnx`) fetched from
  Hugging Face into `models/`.
* Run with `run-cuda.bat` (`python run.py --execution-provider cuda`).

Cloning needed `git -c http.version=HTTP/1.1` — the default HTTP/2 transport
reset mid-clone on this connection.

**DeepFaceLive is not installed and cannot be scripted.** Its Windows builds
are 2–5 GB archives on Mega.nz and Google Drive, neither of which is fetchable
from a shell. Deep-Live-Cam is equivalent for this demo: same real-time swap,
same OBS chain. It swaps at 128px and upscales, so it is visibly softer than a
trained `.dfm` model — which is irrelevant to detection, because the detector
measures blood-volume pulse, not visual fidelity. Adding DeepFaceLive later as
a *second* attack tool is worthwhile: it shows the detector is not tuned to one
tool's artifacts.

## Capture quality — the laptop webcam is not good enough

Measured on this machine, with the detector's camera front end:

| | |
|---|---|
| Frame rate | **19.9 fps**, identical at 1280×720 and 640×360; 16.5 fps on raw reads with no processing. The cap is the sensor, not the resolution and not the loop. rPPG wants 30. |
| Pulse recovered | **None.** Every region sat at −3 to −8 dB SNR, and the off-skin **background control showed the same periodicity at the same strength as skin** — so the rhythm present is lighting and sensor noise, not blood. |
| Face tracking | Flawless: 437/438 frames, never lost, all nine regions placed. |

The front end is fine. The camera is not.

**This threatens demo scene 2** — real human through OBS, no swap, liveness
still passes. That scene is the control that proves the detector reads
physiology rather than sniffing the camera name, and it needs a usable pulse
from the laptop capture path.

### Fix: use the phone as the laptop's camera

The phone sensor is far better and does 30 fps, and the injection chain is
unchanged:

```
phone camera -> connected camera -> OBS -> face swap -> Virtual Camera -> detector
```

**Option A — Windows 11 native (no laptop software).** This machine has
`MicrosoftWindows.CrossDevice` installed, so it is supported:

1. Install **Link to Windows** on the Android phone and sign in with the same
   Microsoft account.
2. On the laptop: Settings → Bluetooth & devices → **Mobile devices** → Manage
   devices → pair the phone.
3. Enable **"Use as a connected camera"**.
4. The phone appears as a normal Windows camera.

**Option B — Iriun Webcam** (installed as a fallback). Install *Iriun Webcam*
on the phone from the Play Store, keep both on the same Wi-Fi or connect over
USB, and it appears as a camera. Chosen over DroidCam because **DroidCam's free
tier caps at 640×480**, which is the limitation being escaped.

Then find it and capture:

```bash
PYTHONPATH=. python -m detector.capture --list-cameras
```

```bash
PYTHONPATH=. python -m detector.capture --source 1 --seconds 22 --save clips/real_01.npz
```

### Do not lock exposure on a camera that does not support it

Locking auto-exposure and white balance is correct in principle — both rewrite
the signal on the same timescale as a heartbeat, and the Android app locks them
before scanning. On this laptop's webcam it is destructive: the camera reports
`CAP_PROP_AUTO_EXPOSURE` as `-1.00` (unsupported), and setting it anyway drops
mean luma from 133 to 20 and face confidence from 0.94 to 0.20.

**These settings are sticky in the driver** and persist into other applications
until restored. The capture tool now restores auto control on every open and
prints the camera's real state, so check that line:

```
camera: 1280x720 codec=YUY2 auto_wb=1.00 auto_exposure=-1.00
```

`--lock-exposure` is opt-in, for a camera that genuinely supports manual control.

### Lighting is the cheapest variable

Even the phone app's own screenshots show 1.4 dB and 4.0 dB SNR against a
backend threshold of 3.0 dB. Pulse signal is marginal on good hardware too.
A steady, bright, even light on the face is the single highest-impact change
on demo day — and a *flickering* light is the one thing that will make the
detector report an illumination artifact instead of a verdict.

## Demo chain

```
                    ┌─ scene 1: phone, real sensor ──────────────► PASS
                    │
you, in front ──────┼─ scene 2: webcam → OBS Virtual Camera ─────► PASS liveness,
of the camera       │            (passthrough, swap OFF)           flagged virtual
                    │                                              source → REVIEW
                    │
                    └─ scene 3: webcam → Deep-Live-Cam → OBS ────► BLOCKED
                                 Virtual Camera (swap ON)
```

Scene 2 is not optional. Without it, scenes 1 and 3 differ in two ways at once
(real sensor vs virtual camera, *and* real face vs swapped face), so a reviewer
can argue you only detected the virtual camera. Scene 2 holds the camera path
fixed and changes only the face, which is what isolates the physiological
detector. It costs nothing: toggle the swap off in Deep-Live-Cam, run, toggle
it back on.

## Python environments

Keep these separate. A global CPU `onnxruntime` (1.31.0) is already installed
and **conflicts with `onnxruntime-gpu` in the same environment.**

| Environment | Python | Purpose |
|---|---|---|
| global | 3.12.10 | backend + detector tests (fastapi, numpy, scipy, sqlmodel present) |
| `D:\redteam\Deep-Live-Cam\.venv` | 3.12 | the swap tool, `onnxruntime-gpu` |
| `detector/.venv` (not yet created) | 3.12 | torch, mediapipe, opencv for the frame front end |
