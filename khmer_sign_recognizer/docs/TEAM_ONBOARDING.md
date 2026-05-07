# SignLink — how to join the training setup

Two things to do before you can run the notebook the lead sends you.

---

## 1. Add the shared Drive folder (one-time, 2 minutes)

The lead shares a Google Drive folder called **SignLink** with your Google account.

1. Open the shared link from the group.
2. In the top bar of the folder, click **Add shortcut to Drive**.
3. Choose **My Drive** and click **Add**.

**Verify**: go to https://drive.google.com — you should see `SignLink/` sitting in My Drive.

That's it. The Colab notebook will find the data and save the weights there automatically.

---

## 2. Open the Colab notebook (one-time)

Open the Colab link the lead sends you.

- **Runtime → Change runtime type → T4 GPU → Save** (do this once, Colab remembers it).

---

## 3. Run training (every session)

Run the cells **in order**. Do not skip any.

| Cell | What it does |
|---|---|
| **1** | Mounts your Drive — click through the permission popup |
| **2** | Downloads the latest code |
| **3** | Installs dependencies |
| **4** | Copies training data from Drive to Colab |
| **5** | Trains the model — watch loss and accuracy per epoch |
| **6** | Saves the trained weights back to the shared Drive |

After Cell 6, everyone on the team has the new weights automatically.

---

## 4. Contribute your own recordings (optional)

More signers = better model. If you want to add your own takes:

### Record on your machine

Install deps and run the recorder:

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python scripts\record_session.py --signer yourname --no-stream
```

Use a short lowercase tag you'll use every time (e.g.`dara`).

```
hello   [ENTER]  →  GET READY 1.5s  →  REC 2.0s  →  "saved [1] hello: ..."
        [ENTER]  →  another take of hello
thanks  [ENTER]  →  switch sign
quit    [ENTER]  →  end session
```

Stand 1.5–2 m from camera, arms visible from the waist up.
**Target: 10 signs × 3 takes minimum.**

Signs to record (don't add new ones — it means retraining from scratch):
`hello`, `thanks`, `yes`, `no`, `please`, `sorry`, `good`, `bad`, `family`, `name`

### Push your recordings to the shared Drive

**You need two windows open side by side: File Explorer and your browser.**

**Window 1 — File Explorer on your machine:**
Open this folder:
```
D:\Projects\Sign to Text\khmer_sign_recognizer\data\sequences_v2\
```
You will see one folder per sign you recorded, e.g. `hello\`, `thanks\`.
Inside each folder are 4 files per take, like:
```
dara__real__clean__0000.npy
dara__real__clean__0000.json
dara__real__noisy__0000.npy
dara__real__noisy__0000.json
```

**Window 2 — Google Drive in your browser:**
1. Go to https://drive.google.com
2. Click **My Drive** in the left sidebar
3. Open **SignLink** → **data** → **sequences_v2**

You are now inside the shared training data folder.

**For each sign you recorded (e.g. hello):**
1. Check if a folder called `hello` already exists in Drive
   - If it does → open it
   - If it does not → right-click → **New folder** → name it `hello` → open it
2. Go back to File Explorer, open your local `hello\` folder
3. Select all 4 files (Ctrl+A), drag them into the Drive browser tab and drop them

Repeat for every sign folder.

**Verify**: each sign folder in Drive should contain 4 files per take ending in `.npy` and `.json`.

> No Drive edit access yet? Zip your `data/sequences_v2/` folder and send to the lead on Telegram.

### Delete a bad take before uploading

```powershell
# See what you recorded
ls data\sequences_v2\hello

# Delete one specific take (e.g. take 0001 — removes all 4 files)
Remove-Item data\sequences_v2\hello\*__0001.*

# Delete all takes of a sign and start over
Remove-Item -Recurse data\sequences_v2\hello

# Wipe everything and start fresh
Remove-Item -Recurse data\sequences_v2\*
```

Quit the recorder first before deleting. Variant numbers don't reset unless you delete the whole label folder.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Cell 4 says 0 samples found | The lead hasn't pushed data yet — ask them to run `push-data` |
| Drive not mounting | Re-run Cell 1, make sure you added the shortcut to **My Drive** (not just opened the share) |
| `No module named 'src'` | Re-run Cell 2 |