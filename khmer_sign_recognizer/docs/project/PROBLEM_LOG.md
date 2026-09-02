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
