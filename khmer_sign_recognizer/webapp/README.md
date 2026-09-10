# webapp/

The control panel. `./run_web.sh` → browser on port 8000.

| | |
|---|---|
| `__main__.py` | Supervisor loop. **Owns the main thread.** |
| `app.py` | Flask routes. |
| `engine.py` | Capture / record / recognize state machine. |
| `library.py` | Language and take scanning; counts always come from disk. |
| `static/index.html` | The entire UI. No build step, no npm. |

## The threading contract

The browser is **controls only**. Flask runs in a daemon thread and does
nothing but flip shared state; **every OpenCV and Open3D call happens on the
main thread** in `__main__.py`. Both libraries crash or hang if driven from a
worker. Keep it that way.

## Three things that break the page

1. The mode switch sets `style.display = 'grid'` on `#mode-record` — that
   layout must stay a grid.
2. ~25 classes are generated at runtime by `loadLabels()`. No CSS rule means
   unstyled output.
3. Engine states are exactly `IDLE`, `COUNTDOWN`, `RECORDING`, `RECOGNIZING`;
   the badge class is built as `'badge badge-' + state.toLowerCase()`.

**Screenshot it after any CSS change.** A dark-mode contrast bug shipped twice
here and was invisible in the source both times.

## The Body toggle

**Configuration → Body** switches the 3D figure between the capsule mannequin
and a rigged VRM from `assets/avatars/`. It is a display setting — it changes
nothing that gets recorded or trained.

Two things here are easy to get wrong:

1. **The state poll is faster than the config debounce.** A poll landing
   between the click and the POST used to echo the server's stale skin back
   over the user's choice, so the request went out with the old value.
   `skinGuardUntil` holds the poll off for 4 s, ending early if the engine
   reports an error. Any other optimistic control needs the same guard.
2. **A failed avatar load must not take the session down.** `_build_mannequins`
   catches `AvatarUnavailable`, falls back to `classic`, and puts the reason in
   `engine.skin_error`, which the hint line under the toggle renders in the
   warning colour. That hint is the only place the user learns it happened.

## Recognize mode

Mid-sign predictions come from a sliding window holding a *partial* sign and
measure ~11 points below the answer committed when motion stops. They render as
muted "Reading…"; only the committed answer gets "Recognized". Do not restyle
them alike.
