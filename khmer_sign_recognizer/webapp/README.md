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

## Two 3D viewers, one rig description

| | where | skinning | quality |
|---|---|---|---|
| **desktop** | Open3D, in the OpenCV window | numpy, main thread | 40k vertices, colours baked per vertex, no alpha |
| **browser** | three.js, in this page | GPU | full detail, real textures and alpha |

The browser one is better and is the one to use. The desktop one stays because
it is what the recorder has always drawn beside the camera feed, and because
it works with no WebGL.

**They share what matters.** Both consume `engine.scene_from_pose()` — the
same scene coordinates, including the forward push on the wrists — and the
same `<model>.rig.json`. Two viewers deriving coordinates separately would
drift apart the first time either was touched.

**They duplicate the pose maths**, in `src/avatar_pose.py` and
`static/avatar3d.js`. That is a real cost and it was taken deliberately:
sending 900 bone matrices per frame over HTTP to avoid it would be far worse
than sending seven joints. Change one, change the other, and re-run
`scripts/check_avatar.py --selftest` for the Python side. Both implement:

```
bone.matrixWorld = S · T(p_new) · R · T(-p_rest) · rest_world
```

three.js binds skins with an identity bindMatrix, so its shader computes the
same product the Python does. `S` — the avatar→scene similarity — can be
folded into each bone matrix because skinning is linear in them.

### Things that bit, in the browser specifically

- **GLTFLoader renames nodes.** `PropertyBinding.sanitizeNodeName` strips
  `. [ ] : /`, so the sidecar's `c_arm_twist_offset.l_0116` arrives as
  `c_arm_twist_offsetl_0116`. Every name lookup tries the sanitised form.
- **Bones must not auto-update.** We write `bone.matrixWorld` ourselves; with
  `matrixWorldAutoUpdate` left on, three recomputes it from the parent chain
  and undoes the flattened-rig repair.
- **Skinned meshes must sit at identity.** glTF ignores a skinned mesh's own
  transform, and GLTFLoader binds with an identity bindMatrix, so a leftover
  node transform would be applied a second time.
- **WebGL ignores `linewidth`.** The hands are instanced spheres and capsules,
  not lines — a LineSegments rig would be the same invisible 1px as Open3D's.

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
