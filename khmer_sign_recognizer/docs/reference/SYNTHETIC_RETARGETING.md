# Synthetic Signer Generation — How the Skeleton Retargeting Works

How SignLink turns one recorded take into several takes of the *same sign*
performed by *differently-proportioned bodies*, and why the sign survives the
transformation.

Code: `src/v2/retarget.py`. Two methods exist: **`scale`** (default) and
**`ik`** (opt-in).

---

## 1. The idea

A sign is defined by **what the arms do**, not by how long they are. Two people
of very different builds performing "thank you" produce very different pixel
coordinates but the same *gesture*.

So: take a real recording, keep the motion, and rebuild the skeleton with
different bone lengths. Each variant is a new training sample labelled with the
same sign — no extra recording, no camera, no rendering. Pure geometry on the
`(60, 48, 3)` landmark array.

The arm is treated as a 2-link chain per side:

```
shoulder ──upper arm──> elbow ──forearm──> wrist ──> 21 hand landmarks
```

---

## 2. Method A — `scale` (the default)

Rebuild the skeleton outward from the shoulders, scaling each **bone length**
while keeping each **bone direction** untouched.

For the left arm, with scale factors `sh, ua, fa, hd`:

```python
mid       = (shoulder_L + shoulder_R) / 2          # shoulder midpoint
new_sh_L  = mid + (shoulder_L - mid) * sh          # shoulders move in/out

new_el_L  = new_sh_L + (elbow_L - shoulder_L) * ua # upper arm: same direction
new_wr_L  = new_el_L + (wrist_L - elbow_L) * fa    # forearm:  same direction

new_hand  = new_wr_L + (hand - wrist_L) * hd       # hand scales about its wrist
```

The key term is `(elbow − shoulder)`: that vector carries **both** the direction
and the length of the bone. Multiplying it by a scalar changes only its
magnitude. The direction is untouched, and each segment is re-anchored to the
already-moved joint before it, so the skeleton stays connected.

### Why the sign is preserved — provably

A joint angle is the angle between two bone vectors. Scaling a vector by a
positive scalar does not rotate it, so every joint angle is mathematically
invariant.

**Measured:** with deliberately extreme and inconsistent factors
(`ua=1.8, fa=0.2`), the maximum change in elbow angle across a whole clip was
**0.0004°** — floating-point noise. The gesture is identical; only the body
differs.

This is a real advantage over GAN or diffusion "different signer" augmentation,
which offers no guarantee that the generated sign still means what it did.

### The default parameters

`jitter = 0.20`, so each factor is drawn uniformly from `[0.80, 1.20]`,
6 variants per real take.

---

## 3. Two problems found in method A

### 3.1 One of the four knobs does nothing

The training view (`clean`) is produced by `shoulder_normalize`, which anchors
on the mid-shoulder and **divides every coordinate by the shoulder width**.

Scaling the shoulders by `sh` and then dividing by shoulder width cancels
exactly. So `sh` has **zero effect** on what the model sees. Only three of the
four parameters were ever doing anything, and what survives normalization is the
*ratio* of limb lengths to shoulder width — which is, fortunately, exactly the
anthropometric property that differs between people.

### 3.2 Bodies that no human has

Each bone was drawn independently, so a synthetic identity could have a long
upper arm with a short forearm. Measured **forearm/upper-arm ratio: 0.67 – 1.48**
across generated identities. Real human limb ratios barely move.

**Fix** — `sample_identity()` draws **one** `build` factor for the whole body,
then allows only ±3% independent deviation per bone:

```python
build = rng.uniform(1 - spread, 1 + spread)        # this person's overall size
bone  = build * rng.normal(1.0, 0.03)              # small per-bone variation
```

Result: ratio range **0.84 – 1.15**. Same amount of between-person variation,
but each synthetic person is internally consistent.

---

## 4. How far can the warping go?

More extreme warping does **not** mean more useful variation. Measured: the
distance a synthetic variant moves from its source, expressed as a percentage of
the mean distance between *different signs*:

| jitter | distance vs between-sign distance |
|---|---|
| 0.20 | 32% |
| 0.35 | 57% |
| 0.50 | **91%** |
| 0.65 | **124%** |
| 0.80 | 198% |

At jitter 0.50 a variant sits almost as far from its own source as a completely
different sign does; at 0.65 it is **further**. The augmentation has stopped
producing "the same sign on another body" and started walking into another
class's territory. Accuracy follows: random forest fell from 96.1% to 93.5% at
jitter 0.50.

**General principle worth stating in a paper:** augmentation magnitude should be
bounded by a fraction of the inter-class distance. Measure it rather than guess.

**Jitter 0.20 is retained** as the default.

---

## 5. Method B — `ik` (anatomically constrained, opt-in)

### The motivation: location is phonemic

In sign languages, **where** a sign is made is part of its meaning — a sign at
the forehead is a different sign at the chest. Linguistically this is a phonemic
parameter, alongside handshape and movement.

Method A scales the arm, so a longer-armed synthetic signer's hand **drifts
outward**, away from the body location the sign is supposed to touch.
**Measured mean wrist displacement: 0.14 units.** A real tall person still
touches their own forehead; they bend the elbow more.

### The fix: keep the hand, move the elbow

Hold the shoulder **and the wrist** fixed, and re-solve the elbow position for
the new bone lengths using standard **two-link inverse kinematics**.

Given shoulder `S`, target wrist `W`, new bone lengths `L1` (upper arm) and
`L2` (forearm):

```
d = |W − S|                      distance to reach
u = (W − S) / d                  unit vector along the reach
v                                unit vector perpendicular to u, in the plane
                                 the ORIGINAL elbow occupied

a = (d² + L1² − L2²) / (2d)      distance along u
h = sqrt(L1² − a²)               distance along v

E = S + a·u + h·v                the new elbow
```

The perpendicular direction `v` is taken from the original elbow's position, so
the arm keeps swinging the same way (inward/outward) rather than flipping.

Verified algebraically and numerically: `|E − S| = L1` and `|W − E| = L2` exactly
(to 1e-4, float32 precision), while the wrist does not move at all.

### The reachability annulus — the subtle part

A two-link arm cannot place its wrist anywhere. The reachable set is an
**annulus**:

```
|L1 − L2|  ≤  d  ≤  L1 + L2
    ↑                  ↑
 fully folded      fully extended
```

Both bounds matter, and missing one is easy:

- `d > L1 + L2` — **too far**: a shorter-armed person cannot reach that point.
- `d < |L1 − L2|` — **too close**: the arm cannot fold that tightly.

The first implementation handled only "too far". Measurement found **294 frames**
in the "too close" case, which silently produced wrong bone lengths.

When the target is outside the annulus, the code **keeps the bone lengths honest**
and moves the wrist to the nearest reachable point — which is what a real person
with those proportions would do. The alternative (stretching bones to reach)
would produce anatomically impossible skeletons.

One further degenerate case: if the original arm is perfectly straight there is
no plane to preserve, `v` collapses to zero, and the bone lengths come out wrong.
The code detects this and picks an arbitrary perpendicular direction.

---

## 6. Does `ik` beat `scale`?

**Not established.** Paired test over 14 identical splits, random forest:

> **+0.69 points, SEM ±0.57, t = 1.21** — not statistically significant.

But that test was run on **single-signer data**, where body-variation
augmentation cannot demonstrate its value in principle (see §7). `scale` remains
the default precisely so the comparison stays clean once multi-signer data is
available:

```bash
python scripts/generate_synthetic.py --language khmer --clean --method scale
python scripts/run_baseline.py --algo rf --lang khmer --mode both --holdout <signer>
# then repeat with --method ik and compare
```

Synthetic data is always regenerable from the real `.npy` files, so nothing is
locked in by recording.

---

## 7. Why this augmentation appeared useless for months

Measured effect of synthetic body-variant augmentation:

| evaluation | effect |
|---|---|
| held-out takes, **same signer** | mean **−7.3 points**; helped 1 of 7 algorithms |
| held-out **signer** (leave-one-signer-out) | mean **+5.4 points**; helped 5 of 8, hurt 0 |

Per-algorithm, cross-signer:

| holdout | algorithm | real only | + synthetic | Δ |
|---|---|---|---|---|
| signer B | random forest | 60.2% | **72.0%** | **+11.9** |
| signer A | logistic regression | 37.0% | **48.0%** | **+11.0** |
| signer A | SVM | 19.6% | 29.0% | +9.4 |
| signer A | random forest | 48.3% | 54.4% | +6.2 |

**The technique varies body proportions. A same-signer test holds body
proportions constant. Such a test is structurally incapable of showing the
benefit** — and mildly penalizes it, because generalization is bought at the cost
of a little specificity (adding synthetic costs ~1 point on *known* signers:
96.6% → 95.6%).

This is the project's main methodological finding, and it applies to any
augmentation aimed at invariance: **the evaluation must vary the factor the
augmentation varies, or the measurement is meaningless.**

---

## 8. Practical notes

- Synthetic samples keep the **original signer id**, so leave-one-signer-out
  evaluation stays leak-free.
- Deleting a real take **orphans its synthetic children**. The synth:real ratio
  stops dividing evenly, the parent link becomes underivable, and the affected
  synthetic data is then silently dropped from training. Regenerate synthetic
  after deletions.
- Splits must be **by take, not by sample**: variant `v` belongs to real take
  `v // N` when the counts divide evenly. Otherwise a copy of a test take lands
  in training.
- Evaluation splits are **always real only** — never grade a model on synthetic
  data.
