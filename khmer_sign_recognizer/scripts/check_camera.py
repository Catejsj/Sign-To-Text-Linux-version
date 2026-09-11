"""Is your camera actually seeing your hands well enough to recognise signs?

Run this BEFORE blaming the model. Feature importance puts **92% of the
signal on the hands** (docs/project/PROBLEM_LOG.md J.4), so hand tracking is
the ceiling on everything downstream — and hand detection is the first thing
poor lighting destroys.

Measured previously on this hardware (PROBLEM_LOG D4):

    warm / orange bulb      33% of frames
    white bulb              42%
    with image enhancement  ~57%   (still needed a phone flashlight)

Those are recognition-breaking numbers. The recorded training corpus, by
contrast, has the hand tracked essentially throughout whenever it is tracked at
all — only 0.2% of its frames are "hand seen, then lost". **A model trained on
that has never seen the flickering pattern bad light produces**, which is why
live recognition can feel far worse than any offline score suggests.

    python scripts/check_camera.py                 # 15 seconds, sign normally
    python scripts/check_camera.py --seconds 30
    python scripts/check_camera.py --no-preview    # headless / over SSH

A preview window shows the camera with the tracker's own skeleton and hand
overlays, and flags each hand GREEN or RED as it is found and lost. Watch it
while you sign: *where in the movement* the hands drop out is the most useful
thing this script can tell you, and no summary line can. The view is mirrored,
so your left hand is on the left.

Sign as you normally would while it runs — it is measuring the tracker, not you.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PREVIEW_WINDOW = "check_camera — is the tracker seeing your hands?"

# BGR. Green reads as "tracked", red as "lost", at a glance and from across a
# room, which is the distance you will actually be signing from.
_OK = (90, 210, 90)
_LOST = (60, 60, 235)
_INK = (245, 245, 245)
_DIM = (170, 170, 170)


def _draw_hud(cv2, frame, *, remaining, body, left, right,
              has_l, has_r, lost_l, lost_r) -> None:
    """Overlay the live state on the preview frame, in place.

    `frame` already carries the tracker's own skeleton and hand overlays —
    `LandmarkCapture.read_frame` composites them — so this adds only what the
    tracker cannot show: whether each hand is being found *right now*, the
    running percentages, and how many times each has dropped out.

    The frame is mirrored first, so it reads like a mirror and your left hand
    is on the left. The skeleton overlay mirrors with it and stays aligned.
    """
    frame[:] = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dark strip behind the text, so white stays legible over a bright room.
    strip = frame[0:74, :].copy()
    cv2.rectangle(strip, (0, 0), (w, 74), (20, 18, 16), -1)
    cv2.addWeighted(strip, 0.62, frame[0:74, :], 0.38, 0, frame[0:74, :])

    def hand_block(x_left: int, label: str, pct: float,
                   present: bool, lost: int) -> None:
        colour = _OK if present else _LOST
        cv2.putText(frame, f"{label} {pct:4.1f}%", (x_left, 28),
                    font, 0.62, _INK, 1, cv2.LINE_AA)
        cv2.putText(frame, "TRACKED" if present else "LOST", (x_left, 52),
                    font, 0.62, colour, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{lost} dropout" + ("" if lost == 1 else "s"),
                    (x_left, 68),
                    font, 0.42, _DIM, 1, cv2.LINE_AA)

    hand_block(14, "LEFT", left, has_l, lost_l)          # mirrored: user's left
    hand_block(w - 150, "RIGHT", right, has_r, lost_r)

    cv2.putText(frame, f"body {body:4.1f}%", (w // 2 - 54, 28),
                font, 0.56, _INK if body > 90 else _LOST, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{max(0.0, remaining):4.1f}s left", (w // 2 - 54, 52),
                font, 0.56, _DIM, 1, cv2.LINE_AA)

    # A red frame while either hand is missing: peripheral vision picks this up
    # while you are looking at your hands rather than at the text.
    if not (has_l and has_r):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _LOST, 4)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.json"))
    ap.add_argument("--no-preview", dest="preview", action="store_false",
                    help="do not open the camera window (headless, or SSH)")
    args = ap.parse_args()

    # This script exists to be run when something is already wrong, so a bare
    # ModuleNotFoundError traceback is the worst possible greeting. The usual
    # cause is the system interpreter rather than the project's venv.
    try:
        import cv2
        from src.capture import LandmarkCapture
        from src.utils import load_config
    except ModuleNotFoundError as exc:
        venv = ROOT / "venv" / "bin" / "python"
        if not venv.exists():                      # Windows layout
            venv = ROOT / "venv" / "Scripts" / "python.exe"
        hint = (f"\n  Run it with the project's interpreter:\n"
                f"      {venv} scripts/{Path(__file__).name}\n"
                f"  or activate the environment first:\n"
                f"      source venv/bin/activate\n"
                if venv.exists() else
                "\n  No venv found — see docs/setup/ for your platform.\n")
        raise SystemExit(
            f"\n  {exc.name!r} is not installed in this Python "
            f"({sys.executable}).{hint}")

    cfg = load_config(args.config)
    cap = LandmarkCapture(cfg)
    if not cap.start():
        raise SystemExit("could not open the camera")

    print(f"\nSign normally for {args.seconds:.0f} seconds...\n")
    if args.preview:
        print("  A preview window is open — watch WHERE in a sign the hands go")
        print("  red. Press q to stop early.\n")

    frames = 0
    body_seen = left_seen = right_seen = 0
    left_runs = right_runs = 0          # how often a hand is LOST mid-stream
    prev_l = prev_r = False
    preview = args.preview
    t0 = time.time()
    next_tick = t0
    try:
        while time.time() - t0 < args.seconds:
            ok, frame = cap.read_frame()
            if not ok:
                continue
            frames += 1
            pose = dict(cap.latest_pose)
            lh = dict(cap.latest_left_hand)
            rh = dict(cap.latest_right_hand)
            has_l, has_r = bool(lh), bool(rh)
            body_seen += bool(pose)
            left_seen += has_l
            right_seen += has_r
            if prev_l and not has_l:
                left_runs += 1
            if prev_r and not has_r:
                right_runs += 1
            prev_l, prev_r = has_l, has_r
            if frames % 30 == 0:
                print(f"  {frames:5d} frames   body {body_seen/frames*100:5.1f}%"
                      f"   left {left_seen/frames*100:5.1f}%"
                      f"   right {right_seen/frames*100:5.1f}%", end="\r")

            if preview and frame is not None:
                try:
                    _draw_hud(cv2, frame,
                              remaining=args.seconds - (time.time() - t0),
                              body=body_seen / frames * 100,
                              left=left_seen / frames * 100,
                              right=right_seen / frames * 100,
                              has_l=has_l, has_r=has_r,
                              lost_l=left_runs, lost_r=right_runs)
                    cv2.imshow(PREVIEW_WINDOW, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(
                            PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break                      # window closed
                except cv2.error:
                    # No display (SSH, headless, a Wayland/GLFW refusal). The
                    # measurement is the point; the window is a convenience,
                    # so drop it and keep going rather than abort the run.
                    preview = False
                    print("\n  (no display available — continuing without the "
                          "preview)")

            # The capture thread owns the camera; read_frame() hands back
            # whatever it last decoded, so polling faster than the camera just
            # samples the same frame repeatedly. Pace at ~30 Hz so the
            # percentages are per unit TIME rather than per loop iteration —
            # drawing the preview costs milliseconds, so sleep the REMAINDER of
            # the tick rather than a fixed interval, or the window would bias
            # the very numbers it is there to explain.
            next_tick += 1.0 / 30.0
            slack = next_tick - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.time()            # fell behind; resynchronise
    finally:
        cap.stop()
        if args.preview:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass

    if not frames:
        raise SystemExit("no frames captured")

    b = body_seen / frames * 100
    l = left_seen / frames * 100
    r = right_seen / frames * 100
    best = max(l, r)

    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print(f"  {frames} samples over {elapsed:.0f}s "
          f"({frames/elapsed:.0f}/s)")
    print("=" * 62)
    print(f"  body detected      {b:5.1f}% of frames")
    print(f"  left hand          {l:5.1f}%")
    print(f"  right hand         {r:5.1f}%")
    print(f"  hand lost mid-use  {left_runs} times (left), "
          f"{right_runs} times (right)")

    print("\n  " + "-" * 58)

    # Refuse to diagnose lighting from a run that cannot support the claim.
    # Missing hands only mean "too dark" if a PERSON was reliably detected;
    # hands and body both absent means nobody was in frame, and a short run
    # measures MediaPipe warming up rather than the room.
    if elapsed < 10:
        print(f"  RUN TOO SHORT ({elapsed:.0f}s) to judge anything — MediaPipe")
        print("  needs several seconds to warm up. Use --seconds 15 or more.")
        return
    if b < 40:
        print(f"  NO PERSON DETECTED ({b:.0f}% of frames had a body).")
        print("  This is not a lighting verdict — the camera may be covered,")
        print("  pointed elsewhere, or you may not have been in frame.")
        print("  Stand in view, then run it again.")
        return

    if best >= 85:
        verdict = ("GOOD — hand tracking is close to the conditions the "
                   "training data was recorded in.")
    elif best >= 70:
        verdict = ("USABLE, but below the recorded corpus. Expect noticeably "
                   "worse live recognition than the offline scores.")
    elif best >= 50:
        verdict = ("POOR — this is roughly the 'with enhancement' level from "
                   "D4, which was still judged insufficient. Add light.")
    else:
        verdict = ("BAD — comparable to the warm-bulb measurement (33%). "
                   "Recognition cannot work reliably from this. Add light.")
    for line in verdict.split(" — "):
        print(f"  {line}")
    print(f"\n  (body detected {b:.0f}% of the time, so a person WAS in frame "
          f"— \n   the hands are what the tracker is losing.)")

    if best < 85:
        print("\n  What actually helps, in order:")
        print("    1. More light on your HANDS, from the front. A desk lamp")
        print("       pointed at you beats a brighter room.")
        print("    2. Avoid backlight — a window or lamp behind you makes the")
        print("       camera expose for the background and darken you.")
        print("    3. Plain, contrasting background behind your hands.")
        print("    4. Mains power, not battery (D4/A12: the CPU throttles and")
        print("       MediaPipe is CPU-bound, halving the capture rate).")
        print("\n  Note: image enhancement was already built for this and")
        print("  REMOVED — it reached ~57% and that was still not enough")
        print("  (PROBLEM_LOG D4). Light is the fix, not code.")

    if max(left_runs, right_runs) > elapsed:      # more than ~1 loss/second
        print(f"\n  Hands are FLICKERING in and out ({left_runs}/{right_runs} "
              f"losses). That pattern barely exists in the training data")
        print("  (0.2% of frames), so the model has effectively never seen it.")
        print("  This hurts more than a steadily-missing hand does.")


if __name__ == "__main__":
    main()
