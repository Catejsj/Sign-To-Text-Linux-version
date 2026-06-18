# SignLink — Team Setup & Recording Guide

Everything you need to record signs and run the experiment. Read once,
top to bottom. **Use the current code** — old copies cause the label
problems we keep hitting.

---

## 1. Get the code (everyone, first time)

```powershell
git clone https://github.com/Catejsj/Sign-to-Text.git
cd Sign-to-Text\khmer_sign_recognizer
```

Already have it? Just update:
```powershell
cd Sign-to-Text\khmer_sign_recognizer
git pull
```

---

## 2. One-time install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If PowerShell blocks `Activate.ps1`, run once (admin PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Every new terminal:** `cd` into `khmer_sign_recognizer` and run
`.\venv\Scripts\Activate.ps1` before any python command.

---

## 3. The 4 signs we're recording

Turkish family signs (from the family video in the group). When recording,
**type these EXACT words** — lowercase, nothing else:

| Type this | Means |
|---|---|
| `aile` | family |
| `anne` | mother |
| `baba` | father |
| `cocuk` | child |

> ⚠️ **Do NOT type English** (`family`, `mother`...) and do NOT add anything
> (`aile (family)`). If everyone types the same 4 words, all recordings merge
> automatically. Typing different words = your data won't combine with ours.

---

## 4. Record (the main task)

```powershell
python scripts\record_session.py --signer YOURNAME --lang autsl --synthetic 1
```

- Replace `YOURNAME` with your own short lowercase tag (e.g. `menghong`).
  **Use the same tag every time** — it identifies your recordings.
- `--lang autsl` → records into the shared Turkish-signs folder
- `--synthetic 1` → auto-makes 1 synthetic body-variant per take

**How to record each sign** (in the browser tab that opens):
1. Type the label (e.g. `anne`) → press **ENTER**
2. Camera shows **COUNTDOWN** → get ready
3. Camera shows **RECORDING** → perform the sign
4. Auto-stops, or press **SPACE** to stop early
5. Click the sign in "Recently used" to do another take

**Do this for each of the 4 signs:**
- **3 takes in normal light**, then **3 takes in dim light** (close the curtains a bit)
- That's 6 takes per sign = 24 takes total

Quit: press **`q`** in the camera window, or **Quit** in the browser.

---

## 5. Send me your recordings

```powershell
python scripts\export_recordings.py --signer YOURNAME
```

This makes a folder `exports\YOURNAME\` with only YOUR takes (it skips the
shared Turkish base data so you don't upload gigabytes).

Then **upload that `exports\YOURNAME\` folder** to the shared Drive:
```
SignLink/recordings/YOURNAME/
```

That's it. I'll merge everyone's recordings and run the experiment.

---

## 6. Common problems

**"My label shows `family` / `aile_1` / something weird"**
You typed the wrong word. Don't panic, don't delete, don't re-record — just
send me your files and I'll fix the label with one command. Next time type
exactly `aile` `anne` `baba` `cocuk`.

**"Khmer/labels.json looks confusing"**
Ignore `labels.json` — it's auto-made, you never edit it. You only ever type
the sign name in the recording box.

**"Activate.ps1 cannot be loaded"**
Run the `Set-ExecutionPolicy` line in section 2.

**"Camera won't open"**
Close other apps using the webcam (Zoom, Teams, browser tabs), then re-run.

**"It's slow / the 3D mannequin lags"**
Add `--mannequin 0` to turn off the 3D window:
```powershell
python scripts\record_session.py --signer YOURNAME --lang autsl --synthetic 1 --mannequin 0
```

---

## 7. The golden rules

1. **Use the current code** (`git pull` first) — old copies cause label chaos
2. **Type exactly:** `aile` `anne` `baba` `cocuk` — lowercase, Turkish
3. **Same signer tag** every session (your name)
4. **3 normal-light + 3 dim-light** takes per sign
5. **Never hand-edit labels.json** — if a label is wrong, send me the files
6. Record + export + upload to `SignLink/recordings/YOURNAME/` — I do the rest
