# SignLink — commands cheat sheet

Everything you need, organized by which terminal it goes in.
Print this. Tape it next to your monitor.

> **Notation:** `Piseth` is used as the example signer tag — replace it with
> your own short tag (and use the **same** tag every time you record).

## Three places code runs

| Terminal       | What runs there                          |
|----------------|------------------------------------------|
| **PowerShell** | recorder, drive sync, Colab pull/push, `update_ips.py` |
| **WSL bash**   | `start_wsl.sh` (the WSL processing layer) |
| **Godot GUI**  | open `khmer-sign-mannequin2/project.godot`, press **F5** |

`cd` once at the start of each terminal session:

```powershell
# PowerShell:
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
```

```bash
# WSL bash:
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
```

---

## ONE-TIME SETUP (do these once, ever)

### A. Allow PowerShell to run venv activate (admin once)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### B. Install rclone for Drive sync
1. Download `rclone-current-windows-amd64.zip` from https://rclone.org/downloads/
2. Extract `rclone.exe` into `C:\Tools\rclone\`
3. Add to PATH (admin PowerShell, once):
   ```powershell
   [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Tools\rclone", [EnvironmentVariableTarget]::Machine)
   ```
4. Reopen PowerShell, then configure Google Drive:
   ```powershell
   rclone config
   ```
   Walk through the prompts (see `docs/FIRST_TIME_COLAB.md` Step 3 for exact answers).

### C. Create the Drive folder structure (PowerShell)
```powershell
rclone mkdir ksldrive:SignLink
rclone mkdir ksldrive:SignLink/data
rclone mkdir ksldrive:SignLink/data/sequences_v2
rclone mkdir ksldrive:SignLink/models
rclone mkdir ksldrive:SignLink/models/weights_v2
rclone mkdir ksldrive:SignLink/logs
rclone mkdir ksldrive:SignLink/logs/v2
python scripts\drive_sync.py doctor
```

---

## EVERY-TIME-YOU-REBOOT (because WSL gets a new IP)

**PowerShell:**
```powershell
python update_ips.py
```

You should see ✅ for both WSL IP and Windows host IP.

---

## DAILY: RECORD SIGNS WITH LIVE MANNEQUIN

Open three things in this order:

### 1. WSL bash terminal
```bash
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
./start_wsl.sh
```
Wait for: `Listening on 0.0.0.0:9999` and `Forwarding to Godot at <ip>:8888`.

### 2. Godot 4.6
Open `khmer-sign-mannequin2/project.godot`, press **F5**.
Expect: `UDP listening on port 8888` in Godot's console.

### 3. PowerShell — the recorder
```powershell
python scripts\record_session.py --signer Piseth
```

A camera window opens. Y-Bot in Godot should now mirror your movement.

### Inside the session
```
hello   [ENTER]   →  GET READY 1.5s → REC 2.0s → "saved [1] hello: ..."
        [ENTER]   →  another take of "hello"
        [ENTER]   →  another take of "hello"
thanks  [ENTER]   →  switches to "thanks", saves take 0
yes     [ENTER]
no      [ENTER]
...
quit    [ENTER]   →  end session (or focus camera window, press q)
```

Stand 1.5–2 m from the camera. If a take captured fewer than 5 frames the
recorder warns and discards it — fix lighting or framing and redo.

---

## RECORD WITHOUT THE LIVE MANNEQUIN (faster, less ceremony)

Skip WSL and Godot. Just one terminal:

```powershell
python scripts\record_session.py --signer Piseth --no-stream
```

Same session UI. Same files saved. Just no Y-Bot animation.

---

## PUSH DATA → DRIVE → COLAB → WEIGHTS BACK

```powershell
# After a recording session
python scripts\drive_sync.py push-data

# Commit + push code if you changed any
git add -A
git commit -m "your message"
git push origin main

# In Colab: open notebooks/colab_train_v2.py, run cells top to bottom

# After training finishes, back in PowerShell:
python scripts\drive_sync.py pull-weights
```

---

## VERIFY YOUR DATA AFTER A SESSION

```powershell
ls data\sequences_v2
ls data\sequences_v2\hello
```

Expect 4 files per take per sign:
```
Piseth__real__clean__0000.npy
Piseth__real__clean__0000.json
Piseth__real__noisy__0000.npy
Piseth__real__noisy__0000.json
```

Quick count of total takes recorded:
```powershell
(Get-ChildItem data\sequences_v2 -Recurse -Filter "*.npy").Count
```

---

## STOP / RESTART CHEAT SHEET

| Want to stop…       | How                                              |
|---------------------|--------------------------------------------------|
| The recorder        | Type `quit` + ENTER in its PowerShell, or press `q` in the camera window |
| WSL processing      | `Ctrl+C` in the WSL bash terminal                |
| Godot               | Stop button in the editor, or close the run window |

If anything misbehaves, restart in this order:
1. quit recorder
2. `Ctrl+C` in WSL
3. stop Godot
4. then re-launch them in the order WSL → Godot → recorder.

---

## TROUBLESHOOTING (fast)

| Symptom                                         | Most likely cause / fix                          |
|------------------------------------------------|--------------------------------------------------|
| `'<' is reserved for future use` in PowerShell | You typed angle brackets from a placeholder. Drop the `< >`. |
| `Cannot find path 'C:\d\...'`                   | Used bash-style `/d/...` in PowerShell. Use `D:\...` instead. |
| `'source' not recognized`                       | Use `.\venv\Scripts\Activate.ps1` in PowerShell. |
| `cd: /d/...: No such file or directory` in WSL  | WSL uses `/mnt/d/...`, not `/d/...`.             |
| `python: not found` in WSL                      | Use `python3` in WSL.                            |
| `update_ips.py: 'wsl' not found`                | You ran it inside WSL. It must run from PowerShell. |
| Camera shows you but mannequin doesn't move    | You used `--no-stream`, or `update_ips.py` wasn't refreshed this boot. |
| `<' operator reserved` again                    | Same — drop angle brackets, they're notation not syntax. |
