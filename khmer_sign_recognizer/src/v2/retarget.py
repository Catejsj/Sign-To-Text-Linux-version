"""Parametric mannequin — retarget recorded motion onto varied body shapes.

Pure math on landmark arrays. No video, no camera, no rendering. It takes a
recorded (60, 48, 3) clip and rebuilds the skeleton with different bone
lengths — same motion (every joint angle preserved), different body.

Used by:
  - scripts/record_session.py   (auto-generates synthetic takes on save)
  - scripts/generate_synthetic.py  (batch-generates from existing recordings)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .schema import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    LEFT_HAND, RIGHT_HAND,
    Source, View, SampleMeta, save_sample, next_variant,
)
from .normalize import shoulder_normalize, deroll


def sample_identity(rng: np.random.Generator, spread: float = 0.15,
                    ratio_sd: float = 0.03) -> dict:
    """Sample ONE synthetic person with anatomically consistent proportions.

    The original sampler drew every bone independently, so a synthetic identity
    could have a long upper arm with a short forearm — a forearm/upper-arm ratio
    anywhere in 0.67-1.48, which no real human has. Limb lengths in people are
    strongly correlated with overall build, and the ratios between them barely
    move (a few percent).

    So: draw one `build` factor for the whole body, then allow only a few percent
    of independent deviation per bone. Same amount of between-person variation,
    but each synthetic person is internally plausible.
    """
    build = float(rng.uniform(1.0 - spread, 1.0 + spread))
    dev = lambda: float(rng.normal(1.0, ratio_sd))
    return {"shoulder": build * dev(), "upperarm": build * dev(),
            "forearm": build * dev(), "hand": build * dev(), "build": build}


def retarget_ik(clip: np.ndarray, upperarm_s: float, forearm_s: float,
                hand_s: float) -> np.ndarray:
    """Retarget while KEEPING THE HAND WHERE IT IS relative to the body.

    Why this differs from `retarget`: in sign language, *location* is phonemic —
    "at the forehead" is part of the sign's meaning. Plain length-scaling moves
    the wrist outward as the arm grows, so a longer-armed signer ends up reaching
    past the place the sign is supposed to touch.

    Here the shoulder and the wrist stay put and only the ELBOW is re-solved, via
    standard two-link inverse kinematics, for the new bone lengths. The result is
    a different-bodied person performing the sign at the same body-relative
    locations — which is what a real different-bodied signer does.

    The elbow is placed in the same plane it originally occupied, so the arm's
    swing direction (inward/outward) is preserved rather than flipped.
    """
    f = clip.copy()
    for sh_i, el_i, wr_i, hand_idx in (
            (L_SHOULDER, L_ELBOW, L_WRIST, LEFT_HAND),
            (R_SHOULDER, R_ELBOW, R_WRIST, RIGHT_HAND)):
        S, E, W = clip[:, sh_i], clip[:, el_i], clip[:, wr_i]
        L1 = np.linalg.norm(E - S, axis=1) * upperarm_s      # new upper arm
        L2 = np.linalg.norm(W - E, axis=1) * forearm_s       # new forearm

        SW = W - S
        d = np.linalg.norm(SW, axis=1)
        safe = d > 1e-6
        u = np.zeros_like(SW)
        u[safe] = SW[safe] / d[safe, None]

        # Component of the ORIGINAL elbow perpendicular to S->W fixes the plane.
        SE = E - S
        perp = SE - (SE * u).sum(1)[:, None] * u
        pn = np.linalg.norm(perp, axis=1)
        v = np.zeros_like(perp)
        ok = pn > 1e-6
        v[ok] = perp[ok] / pn[ok, None]
        # Arm already straight (elbow collinear with shoulder->wrist): there is
        # no original plane to preserve, so pick any perpendicular direction —
        # otherwise v stays zero and the new bone lengths come out wrong.
        if (~ok).any():
            bad = ~ok & safe
            if bad.any():
                ref = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (bad.sum(), 1))
                alt = np.cross(u[bad], ref)
                an = np.linalg.norm(alt, axis=1)
                flat = an <= 1e-6
                if flat.any():        # u parallel to z -> use x instead
                    alt[flat] = np.cross(u[bad][flat],
                                         np.array([1.0, 0.0, 0.0], np.float32))
                    an = np.linalg.norm(alt, axis=1)
                v[bad] = alt / np.maximum(an, 1e-9)[:, None]

        # Circle intersection: elbow sits at distance `a` along u, `h` along v.
        # A two-link arm can only place its wrist in the annulus between
        # |L1-L2| (fully folded) and L1+L2 (fully extended) — BOTH bounds matter.
        far = L1 + L2
        near = np.abs(L1 - L2)
        reachable = (d <= far) & (d >= near) & safe

        a = (d**2 + L1**2 - L2**2) / (2 * np.maximum(d, 1e-6))
        h = np.sqrt(np.maximum(L1**2 - a**2, 0.0))
        newE = S + a[:, None] * u + h[:, None] * v
        newW = W.copy()

        # Outside the annulus the target is physically impossible for this arm.
        # Keep the bone lengths honest and put the wrist at the closest point the
        # arm can actually reach — a real signer with different proportions does
        # the same. Stretching bones instead would produce impossible anatomy.
        out = (~reachable) & safe
        if out.any():
            too_far = out & (d > far)
            too_close = out & (d < near)
            if too_far.any():
                newE[too_far] = S[too_far] + L1[too_far][:, None] * u[too_far]
                newW[too_far] = S[too_far] + far[too_far][:, None] * u[too_far]
            if too_close.any():
                # folded flat: elbow out along u, wrist folded back toward S
                newE[too_close] = S[too_close] + L1[too_close][:, None] * u[too_close]
                newW[too_close] = S[too_close] + near[too_close][:, None] * u[too_close]

        f[:, el_i] = np.where(safe[:, None], newE, E)
        f[:, wr_i] = newW
        # Hand rides with the (possibly moved) wrist; only finger scale changes.
        f[:, hand_idx] = (newW[:, None, :]
                          + (clip[:, hand_idx] - W[:, None, :]) * hand_s)
    return f.astype(np.float32)


def retarget(clip: np.ndarray, shoulder_s: float, upperarm_s: float,
             forearm_s: float, hand_s: float) -> np.ndarray:
    """Retarget a (60, 48, 3) clip onto a body with different bone lengths.

    Bone DIRECTIONS are untouched (so every joint angle — the sign itself —
    is preserved). Only bone LENGTHS are scaled. The skeleton is rebuilt
    outward from the shoulders so it stays connected.
    """
    f = clip
    sh_L, sh_R = f[:, L_SHOULDER], f[:, R_SHOULDER]   # each (60, 3)
    el_L, el_R = f[:, L_ELBOW],    f[:, R_ELBOW]
    wr_L, wr_R = f[:, L_WRIST],    f[:, R_WRIST]

    # Shoulders move in/out from the shoulder midpoint.
    mid = (sh_L + sh_R) / 2.0
    new_sh_L = mid + (sh_L - mid) * shoulder_s
    new_sh_R = mid + (sh_R - mid) * shoulder_s

    # Upper arm: same direction, scaled length, anchored to the new shoulder.
    new_el_L = new_sh_L + (el_L - sh_L) * upperarm_s
    new_el_R = new_sh_R + (el_R - sh_R) * upperarm_s

    # Forearm: same direction, scaled length, anchored to the new elbow.
    new_wr_L = new_el_L + (wr_L - el_L) * forearm_s
    new_wr_R = new_el_R + (wr_R - el_R) * forearm_s

    out = f.copy()
    out[:, L_SHOULDER], out[:, R_SHOULDER] = new_sh_L, new_sh_R
    out[:, L_ELBOW],    out[:, R_ELBOW]    = new_el_L, new_el_R
    out[:, L_WRIST],    out[:, R_WRIST]    = new_wr_L, new_wr_R

    # Hands: scale each hand around its wrist, then re-anchor to the new wrist.
    out[:, LEFT_HAND] = (new_wr_L[:, None, :]
                         + (f[:, LEFT_HAND] - wr_L[:, None, :]) * hand_s)
    out[:, RIGHT_HAND] = (new_wr_R[:, None, :]
                          + (f[:, RIGHT_HAND] - wr_R[:, None, :]) * hand_s)
    return out.astype(np.float32)


def generate_variants(root: Path, noisy_clip: np.ndarray, *,
                      label: str, signer_id: str, fps: int,
                      n: int, jitter: float,
                      rng: np.random.Generator,
                      language: str = "khmer",
                      method: str = "scale") -> int:
    """Generate `n` synthetic body-variant takes from one real noisy clip.

    Each variant is saved as a clean+noisy pair tagged source=SYNTHETIC.
    The ORIGINAL signer_id and language are kept so leave-one-signer-out
    stays leak-free and synthetic data lands in the right language folder.
    Returns the number of synthetic takes saved.

    method:
      "scale" (default) — the original: scale each bone independently. Bone
              lengths change, so the WRIST MOVES; body proportions between
              synthetic identities can be anatomically inconsistent.
      "ik"    — sample one anatomically-consistent identity, then re-solve the
              elbow so the hand stays at the same body-relative LOCATION
              (location is phonemic in sign language).

    "scale" stays the default deliberately: the two have not been compared on
    real multi-signer data yet, and changing it mid-collection would confound
    that comparison. Synthetic can be regenerated either way at any time.
    """
    lo, hi = 1.0 - jitter, 1.0 + jitter
    made = 0
    for _ in range(n):
        if method == "ik":
            idn = sample_identity(rng, spread=jitter)
            noisy = retarget_ik(noisy_clip, idn["upperarm"],
                                idn["forearm"], idn["hand"])
        else:
            sh = float(rng.uniform(lo, hi))
            ua = float(rng.uniform(lo, hi))
            fa = float(rng.uniform(lo, hi))
            hd = float(rng.uniform(lo, hi))
            noisy = retarget(noisy_clip, sh, ua, fa, hd)
        clean = deroll(shoulder_normalize(noisy))

        # One variant index shared by this synthetic take's clean + noisy.
        variant = max(
            next_variant(root, label, signer_id, Source.SYNTHETIC,
                         View.CLEAN, language),
            next_variant(root, label, signer_id, Source.SYNTHETIC,
                         View.NOISY, language),
        )
        note = f"retarget sh{sh:.2f} ua{ua:.2f} fa{fa:.2f} hd{hd:.2f}"
        for clip_v, view in ((clean, View.CLEAN), (noisy, View.NOISY)):
            m = SampleMeta(label=label, signer_id=signer_id,
                           source=Source.SYNTHETIC, view=view,
                           language=language, fps=fps,
                           variant=variant, notes=note)
            save_sample(root, clip_v.astype(np.float32), m)
        made += 1
    return made
