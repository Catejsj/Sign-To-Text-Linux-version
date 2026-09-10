# Avatars

Drop a `.vrm` here and the 3D view can wear it.

The newest file in this folder is the one that gets used, so adding a new
model makes it the active one. Pick a specific file instead with
`--avatar path/to/model.vrm`.

## Getting a model

Any of these produce a file that works:

| source | notes |
|---|---|
| **[VRoid Studio](https://vroid.com/en/studio)** | free, makes anime characters, exports `.vrm` directly. The easiest path. |
| **[VRoid Hub](https://hub.vroid.com/)** | thousands of models; check each one's licence before downloading |
| **Booth / VRChat avatar shops** | usually `.vrm` or `.fbx`; the `.fbx` ones need converting |
| **Blender** | export glTF 2.0 (`.glb`) with *Skinning* enabled and a standard humanoid rig |

**Check the licence.** Most VRoid Hub models restrict redistribution, and some
forbid use in recordings or research output. Using one locally to preview your
own motion is normally fine; putting rendered frames in a paper may not be.
That is your call to make, not the code's.

## Two places it can render

**The browser** — `./run_web.sh`, the *3D view* card. GPU skinning, full
detail, real textures. Use this one.

**The desktop window** — *Configuration → Body → Anime*. CPU skinning, so the
model is thinned to 40k vertices and its colours baked per vertex. Kept
because it draws beside the camera feed and needs no WebGL.

Both read this folder and both honour the sidecar below.

## Check it before wondering why it looks wrong

```bash
./venv/bin/python scripts/check_avatar.py assets/avatars/your_model.vrm
```

It prints the bone map it found, the vertex count, and how long one frame
takes to skin. If it says `UNUSABLE`, the message says what is missing.

## What has to be in the file

* **a skin** — a rigged mesh. A static sculpt cannot be posed.
* **six arm bones** — upper arm, lower arm and hand, both sides. These are
  what the capture actually drives; everything else is optional.
* **no mesh compression.** Draco and meshopt are detected and reported, not
  decoded. Re-export with compression off.

## Two things the loader does to toon models on purpose

**Outline shells are dropped, in both viewers.** Anime models ship a second
copy of the body painted near-black — the cartoon outline, drawn as an
"inverted hull": inflate it slightly, show only its inside surface, and what
remains visible is a dark rim around the silhouette. Left in, it renders as an
opaque black skin and the character disappears inside it. Materials named
`*_Line` or `*outline*` are skipped; on a typical model that is half the
vertices and half the per-frame cost.

Drawing them properly was tried in the browser, where back-face-only rendering
*is* available, and it still failed. Two measurable reasons on this model: the
hull is inflated by 0.0075 units on a 4.7-unit figure — 0.16%, too little to
clear the surface reliably — and much of a character like this is thin cloth
(sleeves, hat brim, cape), an open surface with no inside for a hull to hide
in, so its back faces land in front of the body. A real toon outline needs the
vertex shader to widen the hull along the normal, which is what MToon does and
what this export dropped.

**Colour is read from the emissive channel when base colour is black.** Toon
and unlit materials set base colour to pure black and put the artwork in
emissive so a lit renderer cannot darken it. Read as base colour you get a
black silhouette.

VRM files carry a humanoid bone table, so they map automatically. A plain
`.glb` has no such table and the loader falls back to matching bone *names* —
Mixamo, Blender and VRoid naming all work, but `check_avatar.py` will show you
what it guessed.

## When the automatic matcher fails: `<model>.rig.json`

Some exports need to be described by hand. Write a sidecar next to the model
— `elaina.glb` → `elaina.glb.rig.json` — and it overrides everything the
loader guessed. Start one with:

```bash
./venv/bin/python scripts/check_avatar.py assets/avatars/model.glb --write-rig-template
```

It has two maps, both keyed by node names exactly as `check_avatar.py` prints
them:

| key | what it fixes |
|---|---|
| `bones` | the matcher picked the wrong node, or none. Rigs that call the upper arm `arm.l`, or bury it among 70 IK/FK controls, land here. |
| `chain_parents` | the export **flattened** the skeleton — the bones carrying skin weight are all parented to one root instead of to each other, so rotating the upper arm leaves the forearm behind. List `child: parent` to rejoin the chain. |
| `mirror` | defaults to `true`, and should almost never be changed. The capture mirrors you left-for-right on purpose, so the avatar faces you like a reflection. Set `false` only if a model somehow ends up facing away. |

`chain_parents` changes only what *inherits* motion; rest positions are read
from the file and are unaffected.

**`check_avatar.py` tells you which one you need.** It poses the rig and
measures whether each arm segment actually points where it was aimed. A
flattened hierarchy reports `BROKEN — the arms do not follow`, with 90°-ish
errors; a correct one reports `0.00° error`. That check exists because a
flat rig poses without any error at all and is only wrong once it moves.

`elaina_-_the_witchs_journey.glb.rig.json` in this folder is a worked example
— an Auto-Rig Pro rig exported through Sketchfab, needing both maps.

## What is driven, and what is not

The capture gives six body joints and a nose. So:

* **arms follow you exactly** — the shoulder, elbow and wrist targets are
  aimed at directly
* **the head turns** toward where your nose is
* **the torso does not bend.** The avatar is aligned to your shoulder line as
  a whole, which already puts the chest where your chest is
* **the model's own fingers are hidden.** Its hands collapse and the 21-point
  MediaPipe landmark rig is drawn instead — the same cyan wireframe the
  classic mannequin uses. A VRM's fifteen finger bones per hand cannot be
  recovered from our data, and a frozen open palm during fingerspelling reads
  worse than an honest wireframe.

## This folder is not in git

Model files are tens of megabytes and carry their own licences, so
`.gitignore` keeps them local. This README is tracked; the models are not.
