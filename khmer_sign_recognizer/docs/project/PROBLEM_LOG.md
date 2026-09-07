# SignLink — Complete Problem and Improvement Log

Every problem encountered and every change made, with cause, fix and evidence.
Written as a durable record: for picking the work back up later, and as raw
material for the research paper.

Companion documents:
- `docs/results/FINDINGS.md` — the research-facing writeup (results and conditions)
- `docs/project/PROJECT_BRIEF_FOR_AI.md` — full project brief for literature research
- `docs/setup/SETUP_WINDOWS.md` — cross-platform setup

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

### C4. The deep-learning path had the same leak **[FIXED 2026-09-02]**
`split_random` in `src/v2/dataset.py` shuffled the flat sample list, so every
take's clean view, noisy view and synthetic copies were dealt independently
across train and val. This is why the TCN reported a perfect validation score.

**Measured before the fix**, holding out 15% at seed 42:

| corpus | val samples | with their own take also in train |
|---|---|---|
| `khmer_var` | 707 | **707 (100%)** |
| `khmer` | 882 | **882 (100%)** |

Not a partial leak — total. Every validation clip was a different view or a
warped copy of something in the training set, so no deep-model validation
number published before this date means anything.

**Fix:** `split_random` now holds out **takes**. `synthetic_ratios()` measures
the synth:real ratio per (signer, label) rather than assuming it — the same
defence C2 and C3 needed — and `take_id()` maps any sample back to the
recording it came from. Both are exported for reuse. After the fix: 0 leaked
samples, train/val take sets disjoint, all 7 classes on both sides.

The honest numbers this unlocked are in §J.

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

1. ~~**C4** — take-aware splitting not applied to the deep-model path.~~
   **Fixed 2026-09-02**; see §C4 and §J.
2. **D3** — hand detection ~57%; limits fine-handshape vocabulary. Now known to
   be the model's largest dependency: §J.4 measures 92.3% of feature importance
   on hand joints, 40.6% on hand *depth* alone.
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

11. Deep-path leak found and closed; classical vs deep compared under one
    protocol (§C4, §J)

---

## J. Classical vs deep learning, measured under one protocol

*2026-09-02. All numbers: 5-fold **take-aware** cross-validation
(`StratifiedGroupKFold` on take id), evaluated on **real clean takes only**,
folds asserted disjoint before use. Classical models get the 576 `summary`
features; deep models get the raw `(60, 144)` sequence. Same folds for both, so
any difference belongs to the models and not to the scoring.*

### J.1 The comparison

| corpus | best classical | best deep | verdict |
|---|---|---|---|
| `khmer_var` real | gboost **80.1** ±5.6 | tcn **85.9** ±2.9 | deep, +5.8 |
| `khmer_var` +synth | gboost **78.6** ±3.1 | transformer **86.8** ±1.0 | deep, +8.2 |
| `khmer` real | logreg **96.4** ±2.9 | tcn **98.3** ±1.8 | tie (gap 1.9 < spread 2.4) |
| `khmer` +synth | logreg **97.1** ±1.0 | tcn **98.3** ±1.8 | tie (gap 1.2 < spread 1.4) |

**Deep learning's advantage is a function of how hard the corpus is.** On
`khmer` — 30 takes per sign, 2 signers — every method saturates near 96–98% and
the two categories are indistinguishable. On `khmer_var` — 12 takes per sign,
4 signers, deliberate lighting/distance/position variation — deep wins clearly.
Reporting only the easy corpus would have shown deep learning as unnecessary.

Best configuration overall: **transformer + synthetic on `khmer_var`,
86.8 ±1.0** — the tightest spread of any model measured, on the hardest corpus.

### J.2 Within-category rank does not transfer either

`logreg` is 1st on `khmer` and 6th on `khmer_var`; `gboost` is 1st on
`khmer_var` and 3rd on `khmer`; `lda` is 4th on `khmer` real and **last** once
synthetic is added. This extends §G's cross-dataset instability to a second
axis: the same dataset, a different recording protocol.

### J.3 Synthetic data helps the highest-capacity model most

| model | real | +synthetic | Δ |
|---|---|---|---|
| transformer | 84.4 ±5.3 | **86.8 ±1.0** | **+2.4** |
| tcn | 85.9 ±2.9 | 85.4 ±2.9 | −0.5 |
| gboost | 80.1 | 78.6 | −1.5 |
| lda | 71.9 | 42.7 | **−29.2** |

Consistent with §G: the augmentation varies body geometry, and the `clean`
normalisation already divides body scale out, so a model reading order-invariant
summary statistics has little left to gain. A high-capacity sequence model
trained on 7× the data does gain — and its fold-to-fold spread collapses from
±5.3 to ±1.0, which matters more than the mean.

**LDA's collapse is reproducible and severe** (also 71.0 → 40.8 same-signer,
43.5 → 24.8 cross-signer). Near-duplicate rows destabilise the covariance
inversion. Untested fix: `solver="lsqr", shrinkage="auto"`.

### J.4 Where the signal lives

Random-forest importance over the 576 `summary` features:

| by body part | | by statistic | | by coordinate | |
|---|---|---|---|---|---|
| left hand | 49.9% | max | 36.4% | z | **40.6%** |
| right hand | 42.4% | min | 22.4% | x | 36.5% |
| body | **7.6%** | mean | 20.8% | y | 22.9% |
| | | std | 20.3% | | |

**Hands carry 92.3% of the signal.** Body `z` is identically zero across all
121,320 body-joint values (§H item 7), so the entire 40.6% z-importance sits on
*hand* depth — a monocular estimate, the least reliable channel in the
representation. The model's largest dependency is its weakest input, which is
§D3 restated as a measurement.

Half of all importance is in 112 of 576 features (19.4%).

### J.5 Order-invariance is a real defect, and only partly fixable

`summary` is mean/std/min/max over 60 frames: reverse a clip and the feature
vector is unchanged. Two signs visiting the same positions in a different order
are the same point in feature space.

The worst confusion is ជម្រាប់សួរ ↔ អរគុណ — 19 of 337 takes, with ជម្រាប់សួរ
recall at 54.2% against ≥72.9% for every other sign. Adding temporal features
(per-third means + frame-to-frame velocity) cut those pair errors from **19 to
13** and lifted recall to 60.4% / 81.2%.

But averaged over all nine classical algorithms the same features are worth
only **+0.7**. The order-invariance is genuinely part of the problem and
bolting time back on is genuinely a partial fix — the rest of the gap is what
the sequence models pick up, which is why they win by 5.8–8.2 on this corpus.

### J.6 The old split protocol understated its own uncertainty

`run_var_experiment.py` averages 8 independent random 75/25 take-aware draws.
Against 5-fold CV on the same data: **rankings are identical**, but

- 30 of 337 takes are never tested under the random draws; the rest are tested
  0–7 times. Under CV every take is tested exactly once.
- reported spread roughly **doubles** (rf ±2.5 → ±5.8, bagging ±3.1 → ±6.6,
  knn ±5.1 → ±7.9). Overlapping draws are correlated, so their variance is not
  an honest estimate of it.

The ± values in `Task_A_Report.docx` are therefore too tight. The means stand.

---

## K. The silent hand fill — the largest single defect found so far

*2026-09-02. Code: `src/v2/landmarks.py`. Runs: `.icm/experiments/runs/`.*

### K.1 What was wrong

`normalize.fill_nans` replaces an undetected joint with its last known
position. For a hand that means **all 21 joints collapse onto one point**, and
nothing anywhere records that it happened — the saved array holds a
plausible-looking coordinate where there was no measurement.

Measured on `khmer_var`, 150 real takes:

| | frames where the hand was not detected |
|---|---|
| left hand | **50.8%** |
| right hand | **35.9%** |
| both at once | 33.1% |

Since §J.4 puts **92.3% of feature importance on hand joints**, most of what the
models were reading was invented. This is §D3 — "hand detection ~57%" — but the
consequence was never traced: the number was treated as a capture limitation
rather than as corrupted input.

**Why nobody caught it:** an earlier check for missing landmarks found 0%
dropout, because `fill_nans` had already erased the NaNs. The pipeline destroys
the evidence of its own failure before anything downstream can see it.

### K.2 It is recoverable without re-recording

Twenty-one joints sharing one exact coordinate does not happen to a real hand,
so the mask can be derived from clips already on disk. It is genuine tracking
loss, not noise: **the median take has one contiguous dropout episode per
hand**, not scattered frames.

### K.3 The dropout is partly signal and partly the signer

| by sign | left missing | | by signer | left missing |
|---|---|---|---|---|
| ប៉ា | 75.2% | | Chingsan | **82.0%** |
| ម៉ាក់ | 61.7% | | Seng Menghong | 55.1% |
| ខុស | 60.6% | | Piseth | 31.5% |
| អរគុណ | 31.8% | | Mengly | **25.9%** |

Some signs are one-handed, so absence is real information. But the rate also
varies **3× between people**, which is camera and setup, not language. Presence
therefore partly encodes *who is signing*. **Every variant below was scored
both same-signer and leave-one-signer-out for exactly this reason** — a
signer-identity shortcut would improve the first and damage the second.

### K.4 What fixes it, and what does not

Classical, mean over nine algorithms:

| variant | same signer | unseen signer |
|---|---|---|
| A — as shipped | 70.0 | 45.4 |
| B — add presence features | 70.0 | 45.5 |
| **C — summarise each hand over real frames only** | **79.1 (+9.1)** | **55.6 (+10.3)** |
| D — B + C | 79.1 | 55.4 |

TCN:

| variant | same signer | unseen signer |
|---|---|---|
| A — as shipped | 86.2 ±2.4 | 57.8 ±18.4 |
| B — presence channels, coordinates untouched | 83.7 ±5.6 | **50.8 (−7.0)** |
| **C — fabricated coordinates zeroed + presence** | **87.4 (+1.3)** | **73.0 (+15.2)** |

**Two results worth keeping.**

1. **It helps more on a new person than on the same person** (+10.3 vs +9.1;
   +15.2 vs +1.3). That is the signature of a real fix. A shortcut would show
   the opposite, and given K.3 that was the outcome to rule out.
2. **Annotating the lie is not enough.** Telling the TCN a hand was missing
   while still feeding it the frozen coordinates made it *worse* (−7.0). The
   fabricated numbers have to be removed, not labelled.

### K.5 What changed in the code

- `src/v2/landmarks.py` — `hand_presence`, `zero_missing`, `deep_input`,
  `summary_valid`, `dropout_report`.
- `_featurize` gains `summary_valid`; it is now `run_baseline.py`'s **default**.
- `SignDataset(presence=True)` is the default, feeding `(60, 146)`.
  `train.py --no-presence` restores the old behaviour for reproducing old
  numbers. Input width follows the flag, so the model is now built from
  `dataset.n_features` rather than a constant.
- Presence is read from the **raw** clip, before augmentation — noise would
  break the identical-joints signature and silently mark every hand present.

`summary` and `presence=False` are both kept so pre-2026-09-02 results stay
reproducible.

### K.6 Still open

- **Fix it at the source.** `fill_nans` should record a validity channel at
  capture time instead of leaving it to be reconstructed. That changes the
  `(60, 48, 3)` contract, so it needs a migration plan, not a patch.
- **Why is dropout 3× worse for one signer?** If it is camera placement or
  lighting it belongs in the recording guide, and is cheaper than any modelling
  change.
- **§J's tables predate this fix** and understate every configuration.

---

## L. The canonicalizer — built, measured, not adopted

*2026-09-02. Code: `src/v2/canonical.py`. A negative result, kept because the
reasoning is reusable.*

### L.1 What was tried

A single interface layer — anything in, `(T, 48, 4)` out — carrying **per-joint**
visibility rather than K's per-hand pair, with a gap policy that interpolates
short dropouts and zeroes long ones, feeding a 738-feature classical vector and
a `(60, 192)` deep tensor.

### L.2 It did not beat the simpler fix

| khmer_var, macro-F1 | same signer | unseen signer |
|---|---|---|
| `summary` (original) | 70.0 | 45.4 |
| **`summary_valid` (§K)** | **79.1** | **55.4** |
| `canonical.summary` | 77.8 | 52.0 |
| TCN raw (60,144) | 85.4 | 57.8 |
| **TCN presence (60,146) (§K)** | **87.7** | 73.0 |
| TCN canonical (60,192) | 85.6 | 73.9 |

Classical is 1.3 / 3.4 points **worse**. The deep difference (−2.1 same-signer,
+0.9 unseen) sits well inside a ±15 fold spread and decides nothing.

### L.3 Why — two measurements, both worth keeping

**The gap policy is inert on this data.** Hand-dropout episode lengths across
200 takes:

| gap | share |
|---|---|
| 2–3 frames | 1.2% |
| 4–8 | 2.4% |
| 9–20 | 33.4% |
| 21–59 | 45.9% |
| all 60 | 17.2% |

Median **25 of 60 frames**, and **100% of episodes touch a clip edge** — the
hands are not up yet when a take starts. Only **0.1% of missing frames** sit in
gaps short enough to interpolate, so the policy reduces to zeroing, which §K
already did. Interpolating a 25-frame hole would be inventing half a sign.

**Per-joint visibility is not recoverable from `(T, 48, 3)`.** MediaPipe returns
a whole hand or nothing — there is no per-joint filter at `capture.py:437` — so
reconstruction cannot exceed per-hand resolution. The extra 46 channels carry
nothing the 2-channel version lacked, and on 337 samples the added
dimensionality costs the classical models 1–3 points.

**The lesson:** the design was sound and the data did not support it. Both
reasons are properties of *this corpus* — a longer take, or a tracker that
degrades per-finger, would change both.

### L.4 What was adopted

`normalize._pt` now returns visibility, and `frame_from_landmarks` takes
`with_visibility=True` for `(48, 4)`. `capture.py` has always computed a real
per-joint confidence (`sc[idx]` line 299, `lm.visibility` line 544) and used it
to *delete* joints; `_pt` then discarded it one step later. It is now plumbed.

**The default is still `(48, 3)` and nothing about the stored contract changed.**
Eight people are recording against that shape, and `schema.py`, `SignDataset`
and `verify_pool.py` all assert it. Switching is a migration, not a flag flip.

### L.5 The decision this leaves

Storing real per-joint confidence is the only thing that would make
`canonical.py` worth adopting, and it only helps **takes recorded after** the
change. The question is whether that is worth a mid-collection schema
migration — which is not a call to make from a measurement.

Until then `landmarks.summary_valid` and `landmarks.deep_input` stay the
defaults, and `canonical.py` is scaffolding with an honest sign on it.

---

## M. Bone vectors — telling the model it is looking at a body

*2026-09-02. Code: `src/v2/bones.py`. The largest gain measured on this project.*

### M.1 The gap

Nothing in the pipeline ever said the joints form a skeleton.
`nn.Linear(144, ...)` weights 144 slots with no notion that joint 7 hangs off
joint 6; a decision tree splits one coordinate at a time. **Shuffle all 48
joints consistently across the dataset and you train an identical model.** The
anatomy had to be inferred from 337 examples.

The field's answer is a graph network. The cheap one, tried here: replace a
joint's position with its offset from its parent, so adjacency lives in the
features — no new architecture, no schema change, and both categories gain at
once.

### M.2 Direction, not length — stated before measuring

A bone vector carries direction **and** length, and length is body size, which
is signer identity. The `clean` view divides by shoulder width, but "this
person has long fingers" survives it. So the unit-normalised variant should be
signer-invariant and the raw one should not.

Confirmed:

| khmer_var, macro-F1 | same signer | unseen signer |
|---|---|---|
| joints only (§K) | 79.1 | 55.4 |
| + bone *with length* | 86.2 | 64.8 |
| **+ bone direction (unit)** | **89.4** | **70.4** |
| bone direction only, no joints | 84.0 | 64.8 |
| TCN joints (§K) | 88.3 ±2.7 | 73.0 ±13.5 |
| TCN + bone with length | 89.4 ±2.7 | 72.4 ±14.0 |
| **TCN + bone direction** | **92.0 ±3.1** | **81.4 ±7.3** |

Raw length **costs 6 points cross-signer against direction** while gaining
almost nothing same-signer — the signature of a feature encoding *who* rather
than *what*. The TCN's fold spread also halves, ±13.5 → ±7.3.

Keeping joints alongside bones beats bones alone (89.4 vs 84.0): absolute
position still says *where in the signing space* a sign happens, which
direction cannot express.

### M.3 Where the day ended up

Unseen-signer macro-F1, the number that matters for a stranger at the camera:

| | |
|---|---|
| start of day (leaky split, TCN) | 57.8 |
| after the split fix + hand-fill fix (§C4, §K) | 73.0 |
| **after bone directions** | **81.4** |

**+23.6 points, entirely from fixing defects.** No new architecture, no extra
recordings, no change to the stored data. Every number is 5-fold take-aware CV
or 4-fold leave-one-signer-out with folds asserted disjoint.

Best configuration now: **TCN + presence + bone directions, 92.0 same-signer /
81.4 unseen-signer.**

### M.4 What changed in the code

- `src/v2/bones.py` — `PARENT` (the 48-joint skeleton), `bone_vectors`,
  `summary_bones`, `deep_input_bones`.
- `_featurize` gains `bones`; it is now `run_baseline.py`'s **default**.
- `SignDataset(bones=True)` by default → `(60, 290)`.
  `train.py --no-bones` restores the previous behaviour.
- Bones are computed **after** augmentation, unlike presence — they are a
  geometric function of the coordinates, so they must describe the clip the
  model actually sees. Presence must be read *before*, because augmentation
  noise would destroy the identical-joints signature it is derived from.
- `save_bundle` now stores `sequence_spec={"presence", "bones"}` and the live
  recognizer rebuilds its input from it. Without this the Recognize mode would
  have fed a 144-wide vector to a 290-wide model. Bundles with no spec are read
  as the old raw-144 recipe, so previously saved models still load — verified
  on both existing bundles.

### M.5 Limits

- Bones encode **adjacency**, not the graph. A real ST-GCN also shares weights
  across the skeleton and learns joint relationships beyond parent-child. This
  is the cheap 80%, not the thing itself.
- n = 4 signers. ±7.3 on the best configuration is still wide.
- Sections J and K predate this and understate every row.

---

## N. Full re-test with an adversarial audit

*2026-09-02. Everything in §J–M changed either the features or the splits, so
every earlier table is stale. This is the re-run, and it tries to break the
result before reporting it.*

### N.1 Why you should believe these numbers

This project has shipped two spuriously excellent results — the double
`generate_synthetic` run (+47 points) and the 100% split leak — both of which
looked fine until the right thing was measured. So the audit runs first.

| check | result |
|---|---|
| **A** duplicate clips across takes | 0 on both corpora |
| **B** fold disjointness | asserted per fold, 0 takes on both sides |
| **C** signer purity in LOSO | held-out signer contributes 0 training rows |
| **D** synthetic parentage | 337/420 groups, 0 orphans |
| **E** **label permutation** | **shuffled → 11.5 CV / 12.1 LOSO vs 14.3 chance** |
| **F** signer-identity probe | features predict *who* at 96.9 / 99.8 macro-F1 |

**E is the one that matters.** Labels are shuffled at take level and the whole
pipeline re-run: if anything were carrying the answer through the split, the
score would stay above chance. It lands *at or below* chance on both corpora.
That is the check that would have caught both historical leaks, and it is now
part of the harness rather than something to remember to do.

**F is not a failure — it is the argument for LOSO.** The features identify the
signer nearly perfectly. Any same-signer score therefore includes "can it
recognise this person", which is not the task. See N.5.

### N.2 The category winner flips between corpora

Unseen-signer macro-F1, `bones` features, folds asserted disjoint:

| corpus | best classical | best deep | winner |
|---|---|---|---|
| `khmer_var` (337 takes, **4 signers**) | rf **78.5** | bigru **85.1** | deep, +6.6 |
| `khmer` (420 takes, **2 signers**) | rf **92.9** | bigru 83.8 | classical, +9.0 |

Random forest is the best classical model on **both**, which is itself new —
`gboost` and `logreg` led the old tables and are now 3rd and 4th.

**Do not read the `khmer` row as strongly as the `khmer_var` row.** Two signers
means LOSO is two folds, each training on one person. That is the thinnest
possible cross-signer estimate and the 9-point gap is inside what a third
signer could move. `khmer_var` at four folds is the more trustworthy of the two,
and it says deep.

This extends §G and §J.2 to a third axis: category rank does not transfer
between corpora either.

### N.3 Six deep architectures, not two

`src/v2/model_rnn.py` adds GRU/LSTM and their bidirectional forms. A GRU had
existed since the earliest comparison but was defined *inline inside*
`algo_comparison/run_comparison.py`, where nothing else could import it — which
is why the deep category had only ever been compared two-wide.

`khmer_var`, ranked by unseen signer:

| model | unseen | same | params |
|---|---|---|---|
| **bigru** | **85.1 ±8.3** | 94.1 | 327k |
| bilstm | 84.7 ±6.3 | 94.7 | 435k |
| **gru** | **83.7 ±8.2** | 93.2 | **164k** |
| transformer | 81.8 ±10.2 | 93.2 | 2,201k |
| tcn | 81.4 ±7.3 | 92.0 | 729k |
| lstm | 79.4 ±4.2 | 93.8 | 217k |

**Recurrent models beat both the TCN and the transformer**, which were the only
two ever tested before. The transformer uses 13× the parameters of the GRU to
finish 2 points behind it.

**The bidirectional caveat is a deployment constraint, not a footnote.** A
bidirectional model reads the clip backwards as well as forwards, which is fine
for scoring a recorded take and impossible on a live rolling buffer — there is
no future to read. So `bigru`'s 85.1 is an **offline** number. For Recognize
mode the honest choice is `gru` at **83.7**, and the 1.4-point difference is
what going live costs.

### N.4 The classical side does not want temporal features

Asked directly: does the classical path need the temporal fix the sequence
models get for free? Adding per-third means and frame-to-frame velocity on top
of `bones`:

| | unseen signer, mean over 9 algorithms |
|---|---|
| bones | **70.4** |
| bones + time | 68.8 |

**It makes things worse**, on both corpora (71.5 → 69.7 on `khmer`). The
features triple the dimension against 337 samples, and the summary statistics
already capture most of what a per-third mean would say. Temporal structure is
worth having — it is most of why the sequence models win — but the way to get
it is a sequence model, not more columns.

### N.5 Should leave-one-signer-out be the headline? Yes

The signer probe settles it. The same features that classify signs predict
**which of four people is signing at 96.9 macro-F1** (99.8 on `khmer`, where
chance is 50). A same-signer split therefore lets a model answer partly by
recognising the person, and the gap is not small:

| | same signer | unseen signer | drop |
|---|---|---|---|
| bigru, `khmer_var` | 94.1 | 85.1 | **−9.0** |
| rf, `khmer_var` | 93.2 | 78.5 | **−14.7** |
| tree, `khmer_var` | 83.1 | 63.1 | −20.0 |

**Report both, lead with unseen-signer.** Same-signer is not wrong, it answers
a different question — "can it recognise signs from someone it has trained on"
— which is a real product question for a personal recogniser and the wrong one
for anything a stranger uses. Every table in §J–N gives both for that reason.

### N.6 Custom algorithms work under all of this

`custom_algos/bagging.py` is picked up automatically, gets the new default
features with no edit, and places **2nd of nine** on `khmer_var` unseen-signer
(75.6). A teammate adding a file needs to change nothing — the feature
pipeline sits upstream of the registry, so `--features bones` applies to
everyone's algorithm equally.

The one thing that does **not** carry over automatically is a custom *deep*
model: `run_baseline.py` and the registry are sklearn-shaped (`.fit`/`.predict`),
and the six architectures above are wired in `train.py` instead. A teammate
wanting to add an architecture has no drop-in folder for it. That is a real gap
and it is not yet closed.

### N.7 What is still not addressed

- **n = 4 signers.** Every unseen-signer number here rests on four folds, and
  the best models sit ±6–10 apart. §H item 6 has said a third signer was the
  highest-value next step since before there was a fourth; a fifth and sixth
  would do more for confidence than any further modelling.
- Deep hyperparameters remain untuned; so do the classical ones.
- `algo_comparison/results*/` and every `.docx` still hold pre-2026-09-02
  numbers and now understate everything.

---

## O. Live recognition felt worse than the offline scores — why

*2026-09-04. Reported from real use: dad/mum and the two greetings confused.
Four separate causes, three of them fixable.*

### O.1 First, a caveat on the numbers in this section

Every saved bundle was trained with **no held-out signer**, so replaying the
corpus through one tests it largely on its own training data. The absolute
figures below are therefore inflated and are **not** accuracy claims.

What they *do* support is the comparison between paths, because all three use
the same model on the same data. That comparison is the point of the section.

### O.2 The models in `models/recognizers/` were stale or undertrained

| bundle | state |
|---|---|
| `khmer__tcn` (Aug 4) | pre-fix, `sequence_spec=None`, **saved accuracy `1.0`** — the §C4 split leak |
| `khmer__rf` (Aug 6) | pre-fix, trained on `summary` features that average over fabricated hand positions |
| `khmer_var__tcn`, `khmer_var__gru` | 10-epoch smoke tests left behind by a verification run using `--save` |

**The old TCN reproduces the reported symptom exactly.** Replaying all 420
`khmer` takes through the live path:

    ប៉ា → ម៉ាក់              12 errors
    ជម្រាប់សួរ → អរគុណ        6 errors

A current GRU on the same corpus gives 3 for the worst pair. Nothing else in
the folder produced that pattern. **Fix:** every bundle retrained with current
code; the folder now holds gru/tcn/rf per corpus, all dated 2026-09-04.

### O.3 The sliding window is far worse than the committed answer

Three ways of classifying the same take with the same model:

| | khmer_var gru | khmer gru | old tcn |
|---|---|---|---|
| whole clip, direct | 96.1 | 100.0 | 92.1 |
| committed at end of sign | 94.4 | 98.6 | 88.9 |
| **sliding window, mid-sign** | **83.0** | **90.5** | **51.4** |

The live wrapper costs **11–13 points** relative to direct classification, and
the mid-sign window is far worse still — it holds a *partial* sign the model
never trained on. This is inherent to sliding-window inference, not a bug.

### O.4 The interface called a guess an answer **[FIXED]**

`webapp/static/index.html` labelled a mid-sign prediction that passed the vote
as **"Recognized"**, styled identically to a committed answer — same accent
colour, same landing animation. At 83% against 94% those are not the same
thing, and there was no way to tell them apart on screen.

**Fix:** mid-sign now always reads "Reading…" in muted secondary text.
"Recognized" and the accent styling are reserved for the committed answer.

### O.5 15% of takes never produced an answer at all **[FIXED]**

Only 287 of 337 `khmer_var` takes committed. Cause: the segment was discarded
after `idle_frames_to_commit` (6 frames, 0.2 s) **whether or not it was long
enough to classify**. Sign language holds handshapes for longer than that, so
any sign with a mid-gesture hold lost its accumulated frames and restarted.

**Fix:** `idle_frames_to_reset = 24` (~0.8 s) separates *commit* from
*discard*. Coverage 287 → 296 on `khmer_var`, 415 → 419 on `khmer`, at no cost
to accuracy.

### O.6 Two plausible fixes that measured worse — do not retry

**Absorbing hold frames into the segment.** A held handshape is part of the
sign, so collecting those frames looks obviously right. It is not: padding
with repeated still frames makes the clip mostly static once resampled to 60
frames.

| hold frames absorbed | khmer_var | khmer |
|---|---|---|
| none | **94.4** | **99.0** |
| up to 4 | 92.4 | 97.6 |
| up to 8 | 93.4 | 94.4 |
| unlimited | 93.1 | 94.4 |

Monotonic on `khmer`, no sweet spot. Reverted.

**Lowering `segment_min_frames`** to recover the remaining coverage:

| min frames | khmer_var F1 | answered | khmer F1 |
|---|---|---|---|
| 6 | 92.1 | 308 | 94.8 |
| 8 | 92.8 | 307 | 97.6 |
| 10 | 93.6 | 304 | 98.3 |
| **12 (kept)** | **94.4** | 296 | **99.0** |

Twelve extra takes on one corpus for 2.3 points there and 4.2 on the other.
Left at 12.

**The conclusion both sweeps reach:** for the ~12% of takes with fewer than 12
moving frames, **no answer is better than a forced one**. Every attempt to
convert them cost more than the coverage was worth.

### O.7 After retraining — the reported confusions are gone

All six bundles replayed through the live path (same caveat as O.1 — this is
their own training data, so these are not accuracy figures):

| bundle | committed F1 | answered | worst pair (>=3) |
|---|---|---|---|
| `khmer_var__rf` | 96.8 | 296/337 | **none** |
| `khmer_var__gru` | 94.4 | 296/337 | ម៉ាក់→ខុស 5 |
| `khmer_var__tcn` | 93.3 | 296/337 | ម៉ាក់→ខុស 6 |
| `khmer__gru` | 99.0 | 419/420 | **none** |
| `khmer__rf` | 99.0 | 419/420 | ម៉ាក់→ប៉ា 3 |
| `khmer__tcn` | 98.8 | 419/420 | **none** |

Against the stale bundle it replaces: **88.9, with ប៉ា→ម៉ាក់ 12 and
ជម្រាប់សួរ→អរគុណ 6.** Neither reported pair appears in any retrained model —
the greetings pair is absent entirely, and the worst dad/mum figure anywhere is
3.

**Do not read this table as a model ranking.** `khmer_var__rf` tops it because
a random forest memorises its training data almost perfectly, and this replay
is on training data. The honest unseen-signer ordering is the report's:
**gru 83.7, rf 78.5.** `gru` remains the recommendation for the live app.

### O.8 What to expect, honestly

The report's unseen-signer figure for `gru` is **83.7**, and §O.3 says the live
wrapper costs 11–13 points relative to direct classification. A stranger
signing at the camera should therefore expect noticeably less than 83.7, and
the mid-sign text will look worse still before it settles.

That is the state of the system, not a defect list. The fixes above remove the
stale models, stop the interface overstating a guess, and recover the takes
that silently produced nothing — they do not change what the model knows.

---

## P. Live recognition still poor after §O — the input, not the model

*2026-09-04. Reported as worse than before the retrain. Investigated; the
retrained models are not the cause.*

### P.1 The training corpus contains almost no mid-clip hand loss

`landmarks.hand_presence` flags a hand absent when its 21 joints sit on one
point. That is what `fill_nans` produces for a hand **never seen** in a clip.
A hand seen and then **lost mid-clip** is held at its last position — 21
*distinct* values — and is reported **present**.

Extending the detector to also catch frozen blocks (an exact repeat of the
previous frame, which a tracked hand never produces) changes almost nothing:

| detector | left hand flagged absent | mean same-signer | mean unseen-signer |
|---|---|---|---|
| never-seen only | 48.6% | 89.5 | 70.1 |
| + frozen frames | 48.8% | 89.6 | 70.1 |

**Only 0.2% of recorded frames are "hand seen, then lost."** In the lighting the
corpus was recorded under, a hand is either tracked throughout or never found.
The extension was therefore **not adopted** — it is correct but measures
nothing that exists in this data.

### P.2 Why that matters live

Poor lighting does not remove a hand cleanly; it makes tracking **flicker**.
That produces long runs of frozen fill in the middle of a sign — a pattern the
models have effectively never been trained on, and one the presence channel
labels `1.0` because the coordinates are distinct.

So a model can be correct, freshly trained, and verified, and still behave
badly in a dim room, with **no offline metric able to show it**. §D4 already
measured what dim light costs: hand detection **33% under a warm bulb, 42%
under a white bulb**, ~57% with enhancement that was judged insufficient and
removed. Feature importance puts **92% of the signal on the hands** (§J.4).

### P.3 What was added

`scripts/check_camera.py` — runs the tracker for N seconds and reports how
often each hand is actually found, graded against the D4 thresholds, plus a
count of mid-stream losses. It is the first thing to run when live behaviour
disagrees with the reported scores.

    python scripts/check_camera.py --seconds 20

**No code change fixes this.** Enhancement was already built, measured and
removed (§D4). More light on the hands is the fix.

### P.4 If the camera checks out and it is still bad

Then the honest explanation is the one in §O.8: the unseen-signer figure is
**83.7**, the live wrapper costs 11–13 points relative to direct
classification, and a 7-sign model trained on 4 people is simply not a solved
system. Re-recording the corpus **in the lighting it will actually be used in**
would do more than any further modelling — the training data currently encodes
one narrow set of conditions.

---

## Q. The khmer corpus grew from 2 signers to 7

*2026-09-07. Five people's Task B uploads imported. 420 → 1722 real takes.*

### Q.1 What arrived

A 1 GB zip, 38,429 files, one folder per person — and every person had
structured it differently: `Chingsan/sl_007/`, `Mengly/labels.json`,
`Mao Chhaiyanin/khmer_signs/sl_006/`, `Seng Menghong/khmer/sl_001/`.
`import_takes.py` handled all four layouts without configuration, which is
what it was built for.

| signer | real takes added |
|---|---|
| Chingsan | 210 |
| Mao Chhaiyanin | 292 |
| Mengly | 296 |
| reaj | 294 |
| Seng Menghong | 210 |

Plus the two already present (Piseth, Vichet, 210 each) = **7 signers, 1722
real takes**, and 10332 synthetic at exactly 6:1.

### Q.2 Four things that would have silently corrupted the corpus

**A folder named for one person containing another's files.** `Seth/` held 210
takes tagged `Piseth`. `import_takes.py` takes the signer from the **folder**,
not the filename, which is the only reason this was safe — the filename would
have merged a stranger's upload into an existing signer.

**A whole upload that was already present.** All 210 of `Seth/`'s takes were
byte-identical to data already in the corpus. Content-hash dedup dropped every
one. Without it the corpus would have gained 210 exact duplicates, which
inflates any same-signer score and is invisible afterwards.

**Two people using the default signer tag.** `me` appeared inside filenames
under two different folders. Had the importer trusted filenames, two people
would have merged into one fictional signer and leave-one-signer-out would
have been quietly wrong. Folder-based resolution made it a non-event.

**Uploaded synthetic with broken ratios.** Every upload contained synthetic at
ratios that did not divide evenly (186 against 42, 180 against 32, …). The
uploaded synthetic was discarded before import and regenerated once with
`--clean --per-take 6`. **Shipping synthetic is the mistake here** — see §C2;
it should stay out of the upload entirely (`export_recordings.py` now skips it
by default).

### Q.3 Incomplete takes break synthetic silently **[FIXED]**

15 takes arrived with a **clean view but no noisy view**. Nothing complains at
import — but `generate_synthetic.py` iterates over `*__real__noisy__*.npy`, so
a take with no noisy view gets **no synthetic at all**. The 6:1 ratio then
stops dividing evenly, and an uneven ratio is exactly what makes the
take-aware split group a clip under the wrong parent (§C2, the leak).

`verify_pool.py` caught it: 15 "missing its noisy view" plus 3 ratio failures.
The takes were quarantined rather than deleted, then synthetic regenerated —
after which verify passes with no problems.

All 15 were Seng Menghong's takes beyond variant 0030, so removing them left
him at exactly 30 per sign without any trimming decision being needed.

**Lesson:** a half-uploaded take is worse than a missing one, because it
corrupts an invariant several layers away from where it was introduced. Run
`verify_pool.py` after every import — the failure has no other symptom.

### Q.4 Uneven counts, handled without deleting anything

Four people recorded more than the 30 per sign Task B asked for (36–44).
Rather than trim the corpus, `run_task_a.py` gained `--cap-per-sign N`, which
selects the lowest variant numbers per (signer, sign) and drops their
synthetic children with them. The full corpus stays intact for training; the
report can quote a uniform protocol.

    python algo_comparison/run_task_a.py --lang khmer --grid 30 \
        --no-conditions --cap-per-sign 30 --tag cap30      # for the report
    python algo_comparison/run_task_a.py --lang khmer --grid 30 \
        --no-conditions --tag full                          # what ships

### Q.5 Seven signers changed the picture

Leave-one-signer-out over all seven, macro-F1:

| corpus | signers | best unseen-signer |
|---|---|---|
| `khmer`, before this import | 2 | 83.8 |
| `khmer_var` | 4 | 85.1 |
| **`khmer`, after** | **7** | **93.3** |

Two things fall out of it.

**All six architectures land within 1.5 points** (92.3–93.3) where on
`khmer_var` they spread over 6. **And classical nearly caught deep** — gboost
91.9 against tcn 93.3, a 1.4-point gap, where the 4-signer corpus showed 6.6.
The case for a sequence model weakens as the corpus grows, which is the
opposite of the usual expectation and worth stating in the paper.

§H item 6 has called a third signer the highest-value next step since before
there was a fourth. This is that claim being paid out.

### Q.6 Do the extra takes above 30 matter? Barely

252 extra real takes, +17% data, measured on identical folds:

| | capped at 30 | all takes | gain |
|---|---|---|---|
| tcn | 92.0 | 93.3 | +1.3 |
| bilstm | 92.4 | 93.1 | +0.7 |
| gru | 90.9 | 92.4 | +1.5 |
| gboost | 91.5 | 91.9 | +0.4 |
| **mean over 9 models** | | | **+0.9** |

**The gain is inside the fold-to-fold spread of ±5.8**, so no single model's
improvement is significant. But **all nine moved the same way**, between +0.5
and +1.5, which a coin would do about twice in a thousand tries. So the effect
is real and small: worth keeping for a shipped model, not worth claiming.

Hence two runs from one corpus — `--cap-per-sign 30` for anything quoted as
protocol, the full set for the models that actually ship. Nothing is
duplicated on disk and nothing is deleted.

### Q.7 The report generator invented a protocol Task B never had

`make_task_a_report.py` was written for `khmer_var` and hard-coded its
protocol. Run against Task B it stated as fact:

> "two lighting levels x two distances from the camera x three standing
> positions"

and printed a table of per-condition accuracy for the weakest sign:

| Condition | Correct | Rate |
|---|---|---|
| dim light, far | 82/84 | 98% |
| full light, near | 19/21 | 90% |

**None of that exists.** Task B was recorded freely — no lighting, distance or
position was ever prescribed. Those labels came from `variant % GRID`, an
index with no physical meaning on a freestyle corpus, relabelled as lighting
and distance by code that assumed every corpus was gridded.

This is the most dangerous kind of bug in a reporting tool: it produced a
plausible, well-formatted table that a reader has no way to question, and it
would have gone into a report handed to a teacher.

**Fix:** `--no-conditions` marks a corpus as freestyle. The slot arithmetic is
then skipped entirely rather than computed and ignored, section 8 becomes a
short note explaining there is no grid, and section 5 reports per-sign scores
without a condition breakdown. `khmer_var` keeps both tables, because there
the grid is real.

**Rule this implies:** a report generator must derive its claims about the
protocol from the data, never from the corpus it was first written against.
Every sentence describing how the data was collected is a factual claim.

### Q.8 Report generator, other corpus-awareness fixes

`make_task_a_report.py` was written for `khmer_var` and hard-coded its
protocol — "two lighting levels x two distances x three standing positions".
Run against Task B it printed that as fact, and section 8 computed condition
transfer from grid slots that do not exist. Both corpora now produce a report
describing the corpus they were actually run on: `--no-conditions` marks a
freestyle corpus, and section 8 becomes a short note saying so rather than a
fabricated table.
