# SignLink — Complete Problem and Improvement Log

Every problem encountered and every change made, with cause, fix and evidence.
Written as a durable record: for picking the work back up later, and as raw
material for the research paper.

Companion documents:
- `docs/FINDINGS.md` — the research-facing writeup (results and conditions)
- `docs/PROJECT_BRIEF_FOR_AI.md` — full project brief for literature research
- `docs/SETUP_WINDOWS.md` — cross-platform setup

Legend: **[FIXED]** in place · **[REMOVED]** built, measured, deleted ·
**[REVERTED]** built, tried, rolled back · **[OPEN]** known and unresolved

---

## A. Environment and platform (Windows → Linux port)

### A1. Python 3.14 cannot install the stack **[FIXED]**
CachyOS ships Python 3.14 as the system interpreter. `mediapipe` 0.10.14
publishes no wheels past 3.12, and `open3d` is similar — the venv could not be
created at all.
**Fix:** build the venv with `python3.11` (`pacman -S python311`).

### A2. numpy pin made the install unresolvable **[FIXED]**
The rebuild notes advised pinning `numpy==1.26.4` if mediapipe complained. But
`opencv-python` 4.13 requires `numpy>=2`, so that pin deadlocks pip.
**Fix:** keep numpy 2.4.4. mediapipe 0.10.14 runs against it fine — confirmed by
the Windows lock file, which had the same combination.

### A3. ONNX Runtime silently fell back to CPU **[FIXED]** — most serious
`get_available_providers()` listed `CUDAExecutionProvider`, so the GPU looked
available. Two independent causes:
1. `rtmlib` depends on the **CPU** `onnxruntime` package, which pip installs
   *over* `onnxruntime-gpu` in the same directory, stripping CUDA out entirely.
2. The CUDA provider links against `libcublas.so.12` / `libcudnn.so.9`, but the
   default torch wheel now ships **CUDA 13**.

**Critical behaviour:** ONNX Runtime does *not* raise. Even an explicit
`InferenceSession(model, providers=['CUDAExecutionProvider'])` succeeds on CPU
with only a warning, returning correct but slow results.
**Fix:** install torch from the cu128 index; reinstall `onnxruntime-gpu` after
any `pip install -r`; preload the CUDA libs bundled with torch via
`ctypes.CDLL(..., RTLD_GLOBAL)` before the first session (`src/cuda_setup.py`).
`LD_LIBRARY_PATH` cannot fix this from inside Python — glibc reads it once at
process start.
**Verification rule:** never trust `get_available_providers()`; build a session
and read back `session.get_providers()`. Result: RTMPose bound to CUDA, 31.7 fps.

### A4. opencv-contrib 5.0 shadowed opencv 4.13 **[FIXED]**
`pip` resolved `opencv-contrib-python` to 5.0.0.93, which overwrote the pinned
4.13 `cv2`. `cv2.__version__` reported 5.0.0.
**Fix:** pin both `opencv-python` and `opencv-contrib-python` to 4.13.0.92.

### A5. Camera backend hard-coded to Windows **[FIXED]**
`cv2.CAP_DSHOW` (DirectShow) is Windows-only; on Linux it stalls for seconds
before falling back.
**Fix:** select `CAP_V4L2` on Linux, `CAP_DSHOW` on Windows, `CAP_ANY`
elsewhere — in both `src/capture.py` and `scripts/compare_detectors.py`.

### A6. Windows-only DLL registration ran on Linux **[FIXED]**
`_add_nvidia_dlls()` used the Windows venv layout and `os.add_dll_directory`,
which does not exist on Linux.
**Fix:** guarded by `sys.platform`, later folded into `src/cuda_setup.py`.

### A7. Khmer font lookup only knew Debian paths **[FIXED]**
Arch/CachyOS use `/usr/share/fonts/noto/`, not
`/usr/share/fonts/truetype/noto/`.
**Fix:** added Arch paths. (The repo-bundled font is found first anyway.)

### A8. Open3D window never appeared on Wayland **[FIXED]**
Open3D renders through **GLFW**, whose Wayland path fails to initialize GLEW;
`create_window()` returned `False` silently.
**Fix:** steer GLFW to X11/XWayland before the first window is created.
Note: the older advice `QT_QPA_PLATFORM=xcb` does nothing here — Open3D uses
GLFW, not Qt.

### A9. Camera window rendered solid black **[FIXED]**
Two GUI toolkits in one process — Open3D (GLFW/OpenGL) and OpenCV (Qt5) — left
the Qt window painting black. Confirmed by dumping the frame handed to
`cv2.imshow()`: a correct image with skeleton overlay.
**Fix:** render the mannequin **offscreen** (`visible=False` +
`capture_screen_float_buffer`) and composite it beside the camera in one window.
Removes the conflict and gives a side-by-side view.

### A10. Camera window could not be resized **[FIXED]**
`cv2.imshow()` defaults to `WINDOW_AUTOSIZE`.
**Fix:** create it explicitly with `WINDOW_NORMAL`.

### A11. Hand overlay flickered **[FIXED]**
The hand drawing was cleared after 5 MediaPipe frames (~0.25 s at its ~20 fps
cap), so a single dropped detection blanked the hands.
**Fix:** hold for 15 frames (~0.75 s). Display only — recorded landmarks
unaffected.

### A12. CPU throttling on battery **[FIXED — operational]**
On battery all cores sat in `powersave` at 800–1300 MHz. MediaPipe Holistic is
CPU-bound and measured **64.5 ms/frame (15.5 fps ceiling)**, which was initially
mistaken for a camera or lighting fault.
**Fix:** record on mains power; `powerprofilesctl set performance`.
**Data implication:** capture rate roughly halves on battery; takes are
resampled to 60 frames regardless, so battery recordings are temporally coarser.

---

## B. Web control panel

### B1. Take counts drifted from reality **[FIXED]**
The recorder kept an in-memory "saved this session" counter that never
reconciled with disk, so deleting files in the file explorer left counts stale.
**Fix:** `webapp/library.py` derives every count by **scanning the folder**.
Deletes from anywhere are always reflected; orphaned sidecars are surfaced.

### B2. No way to delete a bad take **[FIXED]**
**Fix:** per-take delete (removes all 4 files: clean+noisy × npy+json), plus
delete-all-for-label and clear-synthetic.

### B3. Labels could not be deleted; "ghost" labels persisted **[FIXED]**
A label present only as a folder (no `labels.json` entry) could not be removed.
**Fix:** `delete_label()` removes the entry *and* the folder, and handles a label
existing on only one side.

### B4. Stop button was always enabled **[FIXED]**
It did nothing outside a take.
**Fix:** disabled unless state is `COUNTDOWN` or `RECORDING`.

### B5. Quit button invisible in dark mode **[FIXED]**
`[data-theme="dark"] .btn { color: var(--bg) }` outranked `.btn-secondary` by CSS
specificity, painting the text near-black on a dark background.
**Fix:** re-assert the text colour for `.btn-secondary` and `.btn-danger` under
`[data-theme="dark"]`.

### B6. Page could stall before painting **[FIXED]**
Google Fonts / Material Symbols were loaded as render-blocking stylesheets.
**Fix:** load with `media="print"` + `onload="this.media='all'"` so the page
paints immediately and upgrades when the fonts arrive.

### B7. Recording froze after ~30 takes **[FIXED]** — worst UX bug
`/api/state` called `list_languages()`, which re-scanned **every** language
folder (tens of thousands of files) on **every 800 ms poll**: measured
**219 ms per poll, ~27% duty cycle**. Because of the GIL, that Flask-thread work
stole time from the main camera thread, so recording stuttered and eventually
hung; a page refresh cleared the request backlog, which is why refreshing
"fixed" it.
**Fix:** cache `list_languages()` with a 5 s TTL and invalidate on every write
(9 invalidation points). Measured **219 ms → ~0 ms**; counts still exact.

### B8. Language dropdown counts went stale after deleting **[FIXED]**
`rebuildLangSelect` compared only language *names*, so count changes never
triggered a rebuild.
**Fix:** compare a signature of names *and* counts.

### B9. Record button sent the slug instead of the label text **[FIXED]**
Would have created junk labels (`sl_001` as a literal new label).
**Fix:** map slug → display text before posting.

### B10. UI controls did not initialize from server state **[FIXED]**
Sliders, signer and language showed defaults until touched.
**Fix:** initialize from `/api/state` on first poll.

### B11. Label counts did not refresh after a take saved **[FIXED]**
**Fix:** refresh when `session_take_count` increases.

### B12. Improvements added
- **Live view switch**: camera only / mannequin only / both, applied live.
- **Light/dark theme** with a sun–moon toggle, OS-preference default,
  `localStorage` persistence.
- **Language create/select** and label management from the browser, replacing
  CLI flags.
- **Live sliders** for mannequin count, synthetic-per-take and duration.
- **Mode isolation**: switching to Recognize tears down the recorder; record
  endpoints return HTTP 409 while in Recognize mode.
- CLI reduced to `./run_web.sh` (or `run_web.bat`).

---

## C. Data pipeline and evaluation methodology

### C1. Locally recorded data had no evaluation split **[FIXED]**
AUTSL encodes train/val/test in the signer-id suffix; local recordings have no
suffix, so everything landed in "train" and `load_split('val', ...)` raised.
With one signer, `--holdout` was also impossible (it would leave zero training
data).
**Fix:** `--split-random FRAC` holds out a fraction of the signer's own takes.

### C2. Random splitting would have leaked augmented data **[FIXED]**
Each real take produces N synthetic variants. A naive per-sample split puts
variants of a held-out take into training, leaking the test motion.
Synthetic filenames carry no parent reference, but variants are generated
N-per-take sequentially, so when counts divide evenly, synthetic variant `v`
belongs to real take `v // N`.
**Fix:** split on that group key. Verified: train 24 real + 144 synthetic per
class, eval 6 real, and the held-out takes' 36 synthetic copies correctly
excluded.

### C3. Deleting a take orphaned its synthetic children **[FIXED]**
The synth:real ratio stopped dividing evenly, the parent link became underivable,
and **all synthetic for that label was silently dropped from training** —
4 of 14 groups were affected before detection.
**Fix:** regenerate synthetic after deletions (`generate_synthetic.py --clean`).
Detectable via the ratio check in `_group_map`.

### C4. The deep-learning path had the same leak **[OPEN]**
`split_random` in `src/v2/train.py` splits **samples**, not takes, so synthetic
copies straddle train/val. This is why the TCN reported a perfect validation
score. The classical path is fixed; the deep path is not.

### C5. `train.py` had no CLI and trained on all languages **[FIXED]**
It required editing a config dataclass, and `discover_samples` was called with
no language filter — so it would have mixed Khmer with AUTSL.
**Fix:** added argparse (`--lang --model --epochs --holdout --save`), a language
filter, and bundle export.

### C6. `import_dataset.py` had a missing import **[FIXED]**
Used `deroll()` without importing it — would have crashed the AUTSL importer at
runtime (not caught by compilation).

---

## D. Capture quality and physical conditions

### D1. Camera roll was an uncorrected nuisance variable **[FIXED]**
`shoulder_normalize` removes position and scale but **not rotation**. Measured
roll across one session: **−10.5° to +19.7°**.

| eval tilt | 0° | 10° | 20° | 30° |
|---|---|---|---|---|
| without correction | 96.3 | 94.7 | 92.2 | **90.0** |
| with `deroll` | 96.7 | **96.7** | **96.7** | **96.7** |

**Fix:** `deroll()` rotates each frame so the shoulder line is horizontal.
Normalization (removing the variable) beat rotation *augmentation* (94–95%,
teaching tolerance). Existing khmer clean views were regenerated from the
untouched noisy source — verified first that `clean == shoulder_normalize(noisy)`
held for all sampled pairs, so regeneration was lossless.
Camera **pitch** costs <1 point and is left uncorrected.

### D2. Recognition broke after moving to a taller desk **[FIXED — operational]**
Diagnosis by measuring live geometry against the training distribution:

| metric | live | training range |
|---|---|---|
| shoulder width | 0.478 | 0.409–0.575 ✓ |
| shoulder tilt | −0.8° | corrected ✓ |
| **vertical position** | **0.883** | **0.361–0.719 ✗** |
| **wrists detected** | **13.5%** | ✗ |

The signer sat far too low in frame and the hands were mostly outside it. Not a
model problem — the landmarks were never captured.
**Fix:** reframe so shoulders sit near mid-frame with room above for the hands.

### D3. Hand detection is far weaker than body detection **[OPEN — design constraint]**
RTMPose body ~100%; MediaPipe hands **~57%** while signing.
**Consequence:** prefer signs distinguished by *arm trajectory*; signs relying on
fine handshape are unreliable at webcam distance.

### D4. Low-light enhancement **[REMOVED]**
Built: CLAHE on the LAB lightness channel, gray-world white balance, and camera
brightness/gamma control, with day/night presets and auto-detection.

| lighting | no enhancement | best preset |
|---|---|---|
| warm/orange bulb | 33.3% | **57.3%** |
| white bulb | 42.0% | **53.6%** |

Roughly doubled hand detection but ~57% was still insufficient — a phone
flashlight was still required. **Removed** as not worth the complexity.
**Hardware finding kept:** this webcam exposes **no `gain` control**. Capping
exposure to reduce motion blur dropped pose detection from **100% to 0%**,
because with no gain to compensate a shorter shutter only removes light. On such
sensors add brightness via `brightness`/`gamma`, never by shortening exposure.

---

## E. Recognition (Phase 2)

### E1. No model was ever saved **[FIXED]**
`run_baseline.py` trained, scored and discarded the model.
**Fix:** `src/v2/recognizer.py` saves a bundle containing the fitted model, the
label map, Khmer display text, and — importantly — `feature_mode` and `view`.
Those travel with the model because getting either wrong produces silently
garbage predictions (the arrays still have the right shape). `--save` retrains on
train+eval before saving, since the score already told us how it generalizes.

### E2. No live inference path **[FIXED]**
**Fix:** `LiveRecognizer` — rolling frame window, reusing the exact training
featurization. Verified by replaying 40 real clips frame-by-frame: **40/40**.

### E3. Live recognition flickered between signs **[PARTLY FIXED]**
Root cause: the model is trained on **complete** takes but classifies a
**continuously sliding window**, which usually holds a partial gesture it has
never seen — so it picks the nearest class and jumps between candidates.
**Fixes applied:** majority vote over recent predictions; a motion gate (a still
person otherwise gets confidently labelled); and **commit-on-pause** — when
motion stops, the whole segment is classified once as the answer.
Tuned by replaying real takes: `idle_frames_to_commit=6` avoided committing on
brief mid-sign pauses; the motion window was shortened to 6 frames because
averaging over 20 delayed end-of-sign detection by ~1.3 s.
Measured: flicker 2.0 → 0.90 label changes per sign, committed accuracy 42/42.

### E4. "Not a finished sign" negative class **[REVERTED]**
Built an 8th class from partial windows (<35% of a sign) and static holds, with
`__none__` masked at commit time. Offline this looked excellent — flicker
1.15 → 0.04 changes/sign at 96.7% accuracy — but felt **worse** in real use and
was reverted at the user's request.
**Lesson kept:** offline simulation of live behaviour is an unreliable proxy.
Available behind `--none-class N`, default 0.

### E5. Wrong model chosen from offline accuracy **[FIXED]**
Logistic regression led offline (98.1% vs RF 96.1%) but felt worse live. Measured
on sliding windows:

| model | offline | live | confidence on partial windows |
|---|---|---|---|
| logistic regression | **98.1%** | 90.5% | **0.90** |
| random forest | 96.1% | **100%** | **0.60** |
| TCN | 100% (leaky) | 88.1% | **0.91** |

Linear and neural models are **overconfident out-of-distribution** — ~0.9
confidence while wrong — which defeats confidence thresholding. The tree ensemble
cannot extrapolate and reports honest uncertainty, so the filter works.
**Fix:** random forest selected for live use.

### E6. Recognize mode was a stub **[FIXED]**
**Fix:** model picker with metadata, start/stop, live prediction display with
confidence bar and history; runs camera-only (no mannequin) to save GPU.

---

## F. Synthetic data generation

### F1. Anthropometrically impossible identities **[FIXED]**
Each bone was scaled independently, giving forearm/upper-arm ratios of
**0.67–1.48** across synthetic identities — ratios no human has.
**Fix:** `sample_identity()` draws one correlated *build* factor with ±3%
per-bone deviation → ratios **0.84–1.15**.

### F2. Retargeting moved the hand away from its sign location **[FIXED — opt-in]**
*Location is phonemic* in sign languages, but plain length-scaling moves the
wrist outward as the arm grows (mean displacement 0.14 units).
**Fix:** `retarget_ik()` keeps shoulder and wrist fixed and re-solves the elbow
by two-link IK. Bone lengths exact to 1e-4.
Two bugs found while building it: the degenerate straight-arm case (no plane to
preserve), and — larger — only the "too far" reachability bound was handled while
**294 frames** were "too close" (a two-link arm can only reach an *annulus*
between |L1−L2| and L1+L2).
**Status:** default remains `scale`; `method="ik"` is opt-in pending a fair
multi-signer comparison. Paired test on 14 identical splits: **+0.69 pts,
SEM ±0.57, t = 1.21** — not significant on single-signer data.

### F3. One augmentation parameter had no effect **[DOCUMENTED]**
`shoulder_normalize` divides by shoulder width, which **exactly cancels** the
`shoulder_s` scaling parameter. Three of four knobs were doing anything.

### F4. Extreme warping degrades class separability **[MEASURED]**
Distance from a variant to its source, as % of the mean between-sign distance:

| jitter | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|---|---|---|---|---|---|
| % | 32 | 57 | **91** | **124** | 198 |

Above ~0.35 a variant is as far from its source as a *different sign*, and
accuracy falls (RF 96.1 → 93.5 at 0.50). **Jitter 0.20 retained.**

### F5. Augmentation of the classical path **[ADDED, default off]**
`augment_clip` (time-warp, noise, rotation) existed but was wired only to the
deep-model path. Exposed to classical models via `--augment N`. Measured on
single-signer data: helped only k-NN (+6.3), cost 1–2 points for strong models.
Default 0.

---

## G. The central experimental finding

The same synthetic augmentation, evaluated two ways:

| evaluation | effect |
|---|---|
| held-out takes, **same signer** | mean **−7.3 pts**; helped 1 of 7 algorithms |
| held-out **signer** (leave-one-signer-out) | mean **+5.4 pts**; helped 5 of 8, hurt 0 |

Body-variation augmentation varies precisely the factor a same-signer test holds
constant, so such a test is **structurally incapable** of showing its benefit and
mildly penalizes it for trading specificity for generalization. The same trade
appears in reverse: synthetic costs ~1 point on *known* signers (96.6 → 95.6).

**Every negative augmentation result in sections D–F was produced under
single-signer evaluation and reversed once a second signer existed.** This is the
project's main methodological contribution.

Related measurements:
- Cross-signer accuracy **~63%** vs ~96% same-signer (a ~33-point gap).
- Combining two signers into one model: **96.6%** overall; signer A lost 2 points,
  signer B gained 37. One model serves multiple signers at negligible cost.
- Algorithm rankings do not transfer between datasets: LDA is best cross-signer
  on landmarks (62.4%) but **last** on Sign Language MNIST images (43.3%);
  SVM is 6th on landmarks and **1st** on images (83.7%).

---

## H. Open items

1. **C4** — take-aware splitting not applied to the deep-model path.
2. **D3** — hand detection ~57%; limits fine-handshape vocabulary.
3. **E3** — live flicker reduced but not eliminated (0.90 changes/sign).
4. **F2** — `scale` vs `ik` retargeting still undecided; needs the multi-signer
   comparison now that a second signer exists.
5. **Other language folders** (autsl etc.) still hold non-de-rolled clean views;
   do not mix them with khmer in one training run until migrated.
6. **Two signers only** — cross-signer results rest on one held-out person per
   direction. A third signer is the highest-value next step.
7. **2D body landmarks** (`z = 0`) — camera pitch cannot be distinguished from
   body proportion.

---

## I. Timeline of major work

1. Linux port: environment, GPU, cross-platform capture and rendering (§A)
2. Web control panel replacing CLI recording (§B)
3. Low-light investigation — built, measured, removed (§D4)
4. Recognition: model saving, live inference, Recognize mode (§E)
5. Evaluation methodology: take-aware splits, leakage fixes (§C)
6. Camera-geometry normalization: `deroll` (§D1)
7. Synthetic-generation improvements: anthropometry, IK (§F)
8. Second signer recorded → cross-signer evaluation → the central finding (§G)
9. Algorithm comparison studies (own landmarks + Sign Language MNIST)
10. Documentation: findings, project brief, Windows setup, this log
