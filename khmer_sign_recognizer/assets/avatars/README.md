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

VRM files carry a humanoid bone table, so they map automatically. A plain
`.glb` has no such table and the loader falls back to matching bone *names* —
Mixamo, Blender and VRoid naming all work, but `check_avatar.py` will show you
what it guessed.

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
