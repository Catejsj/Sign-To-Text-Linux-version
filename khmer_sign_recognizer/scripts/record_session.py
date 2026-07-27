"""Streaming session recorder — browser-based multi-language label input.

The flow:
  1. Camera + RTMPose (body) + MediaPipe (hands) run live in this process.
  2. A tiny local HTTP server on 127.0.0.1 serves a single-page label UI.
  3. Your default browser opens to it automatically. Type a sign label in
     ANY language — Khmer, English, ASL gloss, anything — and press ENTER.
     Browsers handle every IME and Unicode script correctly (unlike tk).
  4. The camera window shows a 1.5 s countdown, then records for 2 s.
  5. The take is saved under  data/sequences_v2/<lang>/<slug>/  with the
     original Unicode label preserved in <lang>/labels.json.
  6. The page keeps a "Recently used" list — click to add another take of
     the same sign without retyping.

Why a browser, not tk? Tk on Windows mangles non-ASCII keyboard input
(Khmer types as '?????' even when the OS IME is producing real Khmer
codepoints — confirmed by typing the same characters in Notepad
working fine). Browsers do not have that bug.

Usage:
  python scripts/record_session.py --signer piseth
  python scripts/record_session.py --signer piseth --lang english
  python scripts/record_session.py --signer piseth --lang khmer --synthetic 6
  python scripts/record_session.py --signer piseth --mannequin 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.capture import LandmarkCapture                # noqa: E402
from src.utils import load_config, setup_logging       # noqa: E402
from src.v2.normalize import (                          # noqa: E402
    frame_from_landmarks, clean_clip_from_frames, noisy_clip_from_frames,
)
from src.v2.schema import save_pair                    # noqa: E402
from src.v2.retarget import generate_variants          # noqa: E402

if HAS_OPEN3D:
    from scripts.mannequin_local import (              # noqa: E402
        Mannequin, retarget_scene, body_to_3d, hand_to_3d,
    )


COUNTDOWN_S = 1.5
DEFAULT_RECORD_S = 3.0        # plenty of room for slow signs by default
MAX_RECORD_S     = 8.0        # hard safety cap so a stuck take can't run forever

CAM_WINDOW = "SignLink — Camera + Mannequin (q to quit)"

# Offscreen size of the mannequin render; it is scaled to the camera height
# before being pasted next to the camera view.
MANNEQUIN_W, MANNEQUIN_H = 1100, 780

ASCII_SAFE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ─────────────────────────────────────────────────────────────────────
#  Camera-overlay font (Pillow, reads .ttf directly — works for any
#  language regardless of Windows-installed fonts).
# ─────────────────────────────────────────────────────────────────────
def find_khmer_font_path() -> str | None:
    candidates = [
        ROOT / "fonts" / "NotoSansKhmer-Regular.ttf",
        Path(r"C:\Windows\Fonts\KhmerUI.ttf"),
        Path(r"C:\Windows\Fonts\Khmerui.ttf"),
        # Arch/CachyOS (noto-fonts) vs Debian/Ubuntu layouts.
        Path("/usr/share/fonts/noto/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/TTF/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"),
        Path("/System/Library/Fonts/KhmerSangamMN.ttc"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


class OverlayFont:
    def __init__(self):
        self.path = find_khmer_font_path()
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}
        if not self.path:
            print("⚠ no Khmer font found. Camera-overlay Khmer text will "
                  "render as boxes. Drop NotoSansKhmer-Regular.ttf into "
                  f"{ROOT / 'fonts'}")

    def _font(self, size: int):
        if size not in self._cache:
            if self.path:
                self._cache[size] = ImageFont.truetype(self.path, size)
            else:
                self._cache[size] = ImageFont.load_default()
        return self._cache[size]

    def draw(self, frame: np.ndarray, text: str, xy: tuple[int, int],
             size: int, color_bgr: tuple[int, int, int]) -> None:
        rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(xy, text, font=self._font(size), fill=rgb)
        np.copyto(frame, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────
#  Label slug + labels.json
# ─────────────────────────────────────────────────────────────────────
def label_to_slug(label: str, labels_path: Path) -> str:
    labels: dict[str, str] = {}
    if labels_path.exists():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            labels = {}
    for slug, existing in labels.items():
        if existing == label:
            return slug
    if ASCII_SAFE.match(label):
        base = label.lower().replace("-", "_")
        slug = base
        i = 1
        while slug in labels and labels[slug] != label:
            slug = f"{base}_{i}"
            i += 1
    else:
        i = 1
        while f"sl_{i:03d}" in labels:
            i += 1
        slug = f"sl_{i:03d}"
    labels[slug] = label
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return slug


# ─────────────────────────────────────────────────────────────────────
#  Browser-based label UI
# ─────────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SignLink — Recording</title>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans Khmer", sans-serif;
       background: #0d1117; color: #cdd9e5; padding: 28px;
       max-width: 720px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 6px; color: #79c2ff; }
.row { color: #8899a6; font-size: 13px; margin-bottom: 18px; }
#status { background: #161b22; padding: 14px 16px; border-left: 3px solid #4ade80;
          border-radius: 4px; margin: 18px 0; font-size: 15px; min-height: 28px; }
#status.busy { border-left-color: #fb923c; }
#status.rec   { border-left-color: #ef4444; }
#timer { font-size: 28px; font-weight: bold; font-variant-numeric: tabular-nums;
         color: #4ade80; display: inline-block; min-width: 60px; }
#status.busy #timer { color: #fb923c; }
#status.rec  #timer { color: #ef4444; }
#next-banner { background: #1f2937; padding: 8px 12px; border-left: 3px solid #79c2ff;
               border-radius: 4px; margin: 8px 0; font-size: 13px; color: #79c2ff;
               display: none; }
.muted { color: #8899a6; font-size: 13px; margin: 14px 0 4px; }
#label-input { width: 100%; padding: 14px 16px; font-size: 24px;
               background: #161b22; color: #e7eef6; border: 1px solid #30363d;
               border-radius: 6px; margin-top: 4px;
               font-family: "Noto Sans Khmer", system-ui, sans-serif; }
#label-input:focus { outline: 2px solid #4ade80; border-color: transparent; }
.btn { padding: 11px 22px; font-size: 15px; background: #21262d; color: #cdd9e5;
       border: 1px solid #30363d; border-radius: 6px; cursor: pointer;
       margin-top: 10px; margin-right: 6px; }
.btn:hover { background: #2d333b; border-color: #4ade80; }
.btn-quit { color: #fb7185; }
.btn-stop { background: #7f1d1d; border-color: #ef4444; color: #fee;
            font-weight: bold; font-size: 16px; padding: 14px 28px;
            display: none; }     /* shown via JS only while RECORDING */
.btn-stop:hover { background: #991b1b; }
.list { margin-top: 8px; }
.label-item { padding: 12px 14px; background: #161b22; margin: 5px 0;
              border: 1px solid #30363d; border-radius: 6px; cursor: pointer;
              font-size: 18px; display: flex; justify-content: space-between;
              align-items: center;
              font-family: "Noto Sans Khmer", system-ui, sans-serif; }
.label-item:hover { background: #21262d; border-color: #4ade80; }
.count { color: #8899a6; font-size: 13px; }
</style></head>
<body>
<h1>SignLink &mdash; recording</h1>
<div class="row">signer: <b id="signer">…</b> &nbsp; · &nbsp;
                 lang: <b id="lang">…</b></div>

<div id="status">connecting…</div>
<div id="next-banner"></div>

<p class="muted">Type sign label (Khmer, English, ASL gloss, …) and press ENTER:</p>
<input id="label-input" type="text" placeholder="សួស្តី / hello / HELLO" autofocus />
<div>
  <button class="btn" onclick="submit()">Record  (ENTER)</button>
  <button class="btn btn-stop" id="stop-btn" onclick="stop()">Stop recording (SPACE)</button>
  <button class="btn btn-quit" onclick="quit()">Quit</button>
</div>

<p class="muted">Recently used (click to record another take):</p>
<div id="label-list" class="list"></div>

<script>
const $ = s => document.querySelector(s);
// Escape EVERY HTML-attribute-unsafe char (incl. quotes — important so
// labels containing " or ' don't break the data-lbl attribute).
function esc(s){
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

let serverState = null;
let serverStateRcvdAt = 0;       // performance.now() when status was received

async function submit(label) {
  const v = (typeof label === 'string') ? label : $('#label-input').value.trim();
  if (!v) return;
  try {
    await fetch('/record', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({label: v}) });
    if (typeof label !== 'string') {
      $('#label-input').value = '';
      $('#label-input').focus();
    }
  } catch (e) {}
}

async function quit() {
  if (!confirm('Quit the recording session?')) return;
  try { await fetch('/quit', {method:'POST'}); } catch(e) {}
}

async function stop() {
  try { await fetch('/stop', {method:'POST'}); } catch(e) {}
}

$('#label-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); submit(); }
});

// Server poll — slow loop, 4 Hz. Keeps the page synced to truth.
async function poll() {
  try {
    serverState = await (await fetch('/status')).json();
    serverStateRcvdAt = performance.now();
  } catch (e) { /* ignore */ }
  setTimeout(poll, 250);
}

// Render loop — 60 Hz. Smoothly counts down between server polls using
// the local clock, so the timer looks continuous instead of jumping
// every 250 ms.
function tick() {
  requestAnimationFrame(tick);
  if (!serverState) return;
  const s = serverState;
  const elapsedSinceServer = (performance.now() - serverStateRcvdAt) / 1000;
  const remaining = Math.max(0, (s.remaining || 0) - elapsedSinceServer);

  $('#signer').textContent = s.signer;
  $('#lang').textContent   = s.language;

  const st = $('#status');
  st.className = '';
  if (s.state === 'COUNTDOWN') st.className = 'busy';
  else if (s.state === 'RECORDING') st.className = 'rec';

  // Stop button only meaningful while recording.
  $('#stop-btn').style.display = (s.state === 'RECORDING') ? 'inline-block' : 'none';

  let html = '';
  if (s.state === 'COUNTDOWN' || s.state === 'RECORDING') {
    html = `<b>${s.state}</b> &mdash; ${esc(s.current_label || '')}` +
           ` &nbsp; <span id="timer">${remaining.toFixed(1)}s</span>`;
  } else {
    html = `status: <b>idle</b> &nbsp;·&nbsp; ` +
           `saved this session: <b>${s.take_count}</b>`;
  }
  st.innerHTML = html;

  // "Next:" banner appears only if a queued label is waiting.
  const nb = $('#next-banner');
  if (s.next_label && (s.state === 'COUNTDOWN' || s.state === 'RECORDING')) {
    nb.style.display = 'block';
    nb.innerHTML = `next take queued: <b>${esc(s.next_label)}</b>`;
  } else {
    nb.style.display = 'none';
  }

  // Re-render the recently-used list (cheap, but only if labels changed).
  const list = $('#label-list');
  const labels = s.labels || [];
  const counts = s.counts || {};
  const signature = labels.map(l => `${l}|${counts[l]||0}`).join(',');
  if (list._sig !== signature) {
    list._sig = signature;
    list.innerHTML = labels.map(lbl => {
      const c = counts[lbl] || 0;
      return `<div class="label-item" data-lbl="${esc(lbl)}">
              <span>${esc(lbl)}</span>
              <span class="count">${c} takes</span></div>`;
    }).join('');
    for (const el of list.querySelectorAll('.label-item')) {
      el.onclick = () => submit(el.dataset.lbl);
    }
  }
}

poll();
tick();
</script>
</body></html>
"""


class BrowserUI:
    """Replaces the old tk label dialog. A local HTTP server on 127.0.0.1
    serves an HTML/JS page that opens in the user's default browser.
    Browsers handle Unicode/IME input correctly for every script (tk on
    Windows mangles Khmer typing). Exposes the same interface the
    Recorder used to expect from the dialog (label_order, take_counts,
    add_take, set_status, closed)."""

    def __init__(self, language: str, signer: str, on_record, on_stop=None):
        self.language = language
        self.signer = signer
        self.on_record = on_record
        self.on_stop = on_stop or (lambda: None)
        self.lock = threading.Lock()
        self.closed = False

        # State the page reads via GET /status
        self.label_order: list[str] = []
        self.take_counts: dict[str, int] = {}
        self.state_label = "idle"
        self.current_display: str | None = None
        self.take_count = 0
        # phase_end_at is the wall-clock time when the current COUNTDOWN
        # or RECORDING phase finishes. The /status endpoint reports the
        # remaining seconds so the browser can display a live timer.
        self.phase_end_at: float | None = None
        self.next_label: str | None = None    # what will be recorded next

        handler = self._make_handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        url = f"http://127.0.0.1:{self.port}/"
        print(f"[browser] label UI: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # ── public interface (Recorder calls these) ──
    def _register(self, label: str) -> None:
        with self.lock:
            if label not in self.label_order:
                self.label_order.append(label)
                self.take_counts.setdefault(label, 0)

    def add_take(self, label: str) -> None:
        with self.lock:
            if label not in self.label_order:
                self.label_order.append(label)
            self.take_counts[label] = self.take_counts.get(label, 0) + 1
            self.take_count += 1

    def set_status(self, msg: str) -> None:
        with self.lock:
            self.state_label = msg

    def set_current(self, label: str | None) -> None:
        with self.lock:
            self.current_display = label

    def set_phase(self, state_label: str, duration: float | None = None) -> None:
        """Set the current phase and (optionally) when it ends, so the
        browser can show a live countdown."""
        with self.lock:
            self.state_label = state_label
            self.phase_end_at = (time.time() + duration) if duration else None

    def set_next(self, label: str | None) -> None:
        """A label submitted while the recorder is busy is shown as
        'next' in the browser so the user knows their click registered."""
        with self.lock:
            self.next_label = label

    def shutdown(self) -> None:
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass

    # ── HTTP handlers ──
    def _make_handler(self):
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: D401
                pass   # quiet

            def _json(self, code, body):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type",
                                 "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    with ui.lock:
                        remaining = (max(0.0, ui.phase_end_at - time.time())
                                     if ui.phase_end_at else 0.0)
                        snap = {
                            "language": ui.language,
                            "signer": ui.signer,
                            "state": ui.state_label,
                            "current_label": ui.current_display,
                            "next_label": ui.next_label,
                            "remaining": round(remaining, 2),
                            "take_count": ui.take_count,
                            "labels": list(ui.label_order),
                            "counts": dict(ui.take_counts),
                        }
                    self._json(200, snap)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b""
                if self.path == "/record":
                    try:
                        data = json.loads(raw.decode("utf-8"))
                    except Exception:
                        data = {}
                    label = (data.get("label") or "").strip()
                    if not label:
                        self._json(400, {"error": "empty label"})
                        return
                    ui._register(label)
                    try:
                        ui.on_record(label)
                    except Exception as e:
                        print(f"[browser] on_record error: {e}")
                    self._json(200, {"ok": True})
                elif self.path == "/stop":
                    try:
                        ui.on_stop()
                    except Exception as e:
                        print(f"[browser] on_stop error: {e}")
                    self._json(200, {"ok": True})
                elif self.path == "/quit":
                    ui.closed = True
                    self._json(200, {"ok": True})
                else:
                    self.send_response(404)
                    self.end_headers()

        return Handler


# ─────────────────────────────────────────────────────────────────────
#  Session header / new-language confirmation
# ─────────────────────────────────────────────────────────────────────
def list_existing_languages(root: Path) -> list[tuple[str, int]]:
    if not root.exists():
        return []
    out = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = sum(1 for _ in lang_dir.rglob("*__real__noisy__*.npy"))
        out.append((lang_dir.name, count))
    return out


def print_session_header(root: Path, lang: str, signer: str) -> None:
    print("=" * 60)
    print("  SignLink — recording session")
    print(f"  signer:  {signer}")
    print(f"  lang:    {lang}")
    print("=" * 60)
    existing = list_existing_languages(root)
    if existing:
        print("\nExisting languages:")
        for name, count in existing:
            mark = "  ←" if name == lang else ""
            print(f"  • {name:14s} ({count} real takes){mark}")
    else:
        print("\n(no languages recorded yet)")
    if any(name == lang for name, _ in existing):
        print(f"\n[OK] Using existing '{lang}' folder. New takes will be added here.\n")
    else:
        print(f"\n[!] '{lang}' is a NEW language. Folder will be created on first save.")
        if existing:
            print("    If you meant an existing one, Ctrl+C and re-run with the right --lang.")
            try:
                input(f"    Press ENTER to continue with '{lang}', or Ctrl+C: ")
            except (EOFError, KeyboardInterrupt):
                print("\n(quit)"); sys.exit(0)
        print()


# ─────────────────────────────────────────────────────────────────────
#  Recorder — state machine + camera + browser UI integration
# ─────────────────────────────────────────────────────────────────────
class Recorder:
    def __init__(self, args, cfg, root: Path, lang: str,
                 capture: LandmarkCapture, rng: np.random.Generator):
        self.args = args
        self.cfg = cfg
        self.root = root
        self.lang = lang
        self.capture = capture
        self.rng = rng
        self.labels_path = root / lang / "labels.json"
        self.img_w = cfg["capture"]["width"]
        self.img_h = cfg["capture"]["height"]
        self.fps = cfg["capture"].get("fps", 30)

        self.state = "IDLE"
        self.current_label: str | None = None
        self.pending_label: str | None = None
        self.countdown_start = 0.0
        self.record_start = 0.0
        self.take_buffer: list[np.ndarray] = []
        self.take_count = 0
        # When True, the in-progress recording finalises on the next tick.
        # Set by SPACE in the camera window or by the browser's Stop button.
        self.stop_now = False

        self.dialog = BrowserUI(
            language=lang, signer=args.signer,
            on_record=self._on_picked,
            on_stop=self._on_stop_requested,
        )
        self.overlay = OverlayFont()
        self._preload_known_labels()

        # WINDOW_NORMAL (rather than imshow's default WINDOW_AUTOSIZE) is what
        # makes the camera window resizable / maximizable.
        cv2.namedWindow(CAM_WINDOW, cv2.WINDOW_NORMAL)
        win_w = self.img_w
        if args.mannequin > 0 and HAS_OPEN3D:
            win_w += int(MANNEQUIN_W * self.img_h / MANNEQUIN_H)
        cv2.resizeWindow(CAM_WINDOW, win_w, self.img_h)

        # Optional Open3D mannequin window
        self.synths: list[tuple] = []
        self.vis = None
        if args.mannequin > 0 and HAS_OPEN3D:
            N = args.mannequin
            spacing = 2.6
            xs = (np.arange(N) - (N - 1) / 2.0) * spacing
            for i in range(N):
                body = dict(
                    sh=float(rng.uniform(0.80, 1.20)),
                    ua=float(rng.uniform(0.80, 1.20)),
                    fa=float(rng.uniform(0.80, 1.20)),
                    hd=float(rng.uniform(0.80, 1.20)),
                    xoff=float(xs[i]),
                )
                self.synths.append((Mannequin(), body))
            # visible=False: the mannequin renders OFFSCREEN and is composited
            # into the camera window (see _mannequin_image). Two on-screen GUI
            # toolkits in one process — Open3D's GLFW/OpenGL window next to
            # OpenCV's Qt window — leaves the Qt window painting solid black.
            # Rendering offscreen sidesteps that and puts both views side by
            # side in a single window.
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                "SignLink — Synthetic Signers (live)",
                width=MANNEQUIN_W, height=MANNEQUIN_H, visible=False)
            for mq, _ in self.synths:
                for g in mq.geometries():
                    self.vis.add_geometry(g)
            opt = self.vis.get_render_option()
            opt.background_color = np.array([0.05, 0.06, 0.09])
            opt.light_on = True
            vc = self.vis.get_view_control()
            vc.set_front([0.0, 0.0, 1.0]); vc.set_up([0.0, 1.0, 0.0])
            vc.set_lookat([0.0, 0.0, 0.0]); vc.set_zoom(0.75 + 0.14 * N)
            print(f"[mannequin] {N} synthetic signer(s) — same motion, "
                  f"different bodies, live.")

    def _on_picked(self, label: str) -> None:
        """Always accept a submission. If we're idle it runs immediately;
        if we're already recording it queues (last-click wins), so the
        next take auto-starts when the current one finishes."""
        self.pending_label = label
        if self.state != "IDLE":
            self.dialog.set_next(label)

    def _on_stop_requested(self) -> None:
        """Stop the current recording on the next tick (saves what we have).
        Triggered by the browser's Stop button OR SPACE in the camera."""
        if self.state == "RECORDING":
            self.stop_now = True

    def _preload_known_labels(self) -> None:
        if not self.labels_path.exists():
            return
        try:
            labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        lang_dir = self.labels_path.parent
        n = 0
        for slug in sorted(labels.keys()):
            display = labels[slug]
            slug_dir = lang_dir / slug
            count = (len(list(slug_dir.glob("*__real__noisy__*.npy")))
                     if slug_dir.exists() else 0)
            with self.dialog.lock:
                self.dialog.label_order.append(display)
                self.dialog.take_counts[display] = count
            n += 1
        if n:
            print(f"[labels] pre-loaded {n} {self.lang} signs from "
                  f"{self.labels_path.relative_to(self.root.parent)}")

    # ── main loop ──
    def run(self) -> None:
        print("\nReady. Use the browser tab to enter labels. Camera window: "
              "press q to quit (or click Quit in the browser).\n")
        try:
            while not self.dialog.closed:
                self._tick()
                # ~60 Hz cap — capture/inference dominates real fps
                time.sleep(0.015)
        except KeyboardInterrupt:
            pass

    def _tick(self) -> None:
        ret, frame = self.capture.read_frame()
        if not ret or frame is None:
            cv2.waitKey(1)
            return

        with self.capture.result_lock:
            pose = dict(self.capture.latest_pose)
            lh = dict(self.capture.latest_left_hand)
            rh = dict(self.capture.latest_right_hand)
        has_pose = bool(pose)
        now = time.time()

        if self.state == "IDLE" and self.pending_label is not None:
            self.current_label = self.pending_label
            self.pending_label = None
            self.state = "COUNTDOWN"
            self.countdown_start = now
            self.dialog.set_phase("COUNTDOWN", COUNTDOWN_S)
            self.dialog.set_current(self.current_label)
            self.dialog.set_next(None)

        if self.state == "COUNTDOWN" and now - self.countdown_start >= COUNTDOWN_S:
            self.state = "RECORDING"
            self.record_start = now
            self.take_buffer = []
            self.dialog.set_phase("RECORDING", self.args.duration)

        if self.state == "RECORDING":
            if has_pose:
                self.take_buffer.append(
                    frame_from_landmarks(pose, lh, rh, self.img_w, self.img_h)
                )
            elapsed = now - self.record_start
            # End if the user pressed SPACE / Stop, the configured duration
            # is up, or the hard safety cap is hit.
            if self.stop_now or elapsed >= self.args.duration or elapsed >= MAX_RECORD_S:
                self._finalize_take()
                self.state = "IDLE"
                self.stop_now = False
                self.dialog.set_phase("idle", None)
                self.dialog.set_current(None)

        self._draw_overlay(frame, now)

        if self.vis is not None:
            if has_pose:
                self._update_mannequins(pose, lh, rh)
            self.vis.poll_events()
            self.vis.update_renderer()
            mann = self._mannequin_image(frame.shape[0])
            if mann is not None:
                frame = np.hstack([frame, mann])

        cv2.imshow(CAM_WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.dialog.closed = True
        elif key == ord(" ") and self.state == "RECORDING":
            # SPACE = stop the current recording early ("done signing now").
            self.stop_now = True

    def _finalize_take(self) -> None:
        if len(self.take_buffer) >= 5 and self.current_label:
            slug = label_to_slug(self.current_label, self.labels_path)
            clean = clean_clip_from_frames(self.take_buffer)
            noisy = noisy_clip_from_frames(self.take_buffer)
            p_clean, p_noisy = save_pair(
                self.root, clean, noisy,
                label=slug, signer_id=self.args.signer,
                language=self.lang, fps=self.fps,
            )
            self.take_count += 1
            print(f"saved [{self.take_count}] {self.current_label} -> {slug}: "
                  f"{p_clean.name} + {p_noisy.name} "
                  f"({len(self.take_buffer)} raw frames)")
            if self.args.synthetic > 0:
                n = generate_variants(
                    self.root, noisy,
                    label=slug, signer_id=self.args.signer,
                    language=self.lang, fps=self.fps,
                    n=self.args.synthetic, jitter=0.20, rng=self.rng,
                )
                print(f"         + {n} synthetic body-variants")
            self.dialog.add_take(self.current_label)
        else:
            print(f"WARN: only {len(self.take_buffer)} frames — discarded.")

    def _update_mannequins(self, pose: dict, lh: dict, rh: dict) -> None:
        scene_joints: dict[str, np.ndarray] = {}
        name_map = {
            "left_shoulder": "l_shoulder", "right_shoulder": "r_shoulder",
            "left_elbow": "l_elbow",       "right_elbow": "r_elbow",
            "left_wrist": "l_wrist",       "right_wrist": "r_wrist",
            "left_hip": "l_hip",           "right_hip": "r_hip",
            "nose": "nose",
        }
        for src_name, dst_name in name_map.items():
            if src_name in pose:
                scene_joints[dst_name] = body_to_3d(
                    pose[src_name], self.img_w, self.img_h)
        if "l_shoulder" in scene_joints and "r_shoulder" in scene_joints:
            sw = float(np.linalg.norm(
                scene_joints["l_shoulder"] - scene_joints["r_shoulder"]))
            fwd = np.array([0.0, 0.0, sw * 0.32])
            if "l_wrist" in scene_joints:
                scene_joints["l_wrist"] = scene_joints["l_wrist"] + fwd
            if "r_wrist" in scene_joints:
                scene_joints["r_wrist"] = scene_joints["r_wrist"] + fwd
        scene_lhand = scene_rhand = None
        if "l_wrist" in scene_joints:
            scene_lhand = hand_to_3d(lh, self.img_w, self.img_h,
                                     scene_joints["l_wrist"])
        if "r_wrist" in scene_joints:
            scene_rhand = hand_to_3d(rh, self.img_w, self.img_h,
                                     scene_joints["r_wrist"])
        for mq, body in self.synths:
            sj, sl, sr = retarget_scene(
                scene_joints, scene_lhand, scene_rhand,
                body["sh"], body["ua"], body["fa"], body["hd"], body["xoff"])
            mq.update(sj, sl, sr)
        for mq, _ in self.synths:
            for g in mq.geometries():
                self.vis.update_geometry(g)

    def shutdown(self) -> None:
        if self.vis is not None:
            try: self.vis.destroy_window()
            except Exception: pass
            self.vis = None
        self.dialog.shutdown()

    def _mannequin_image(self, height: int) -> np.ndarray | None:
        """Grab the offscreen Open3D render as a BGR image scaled to `height`,
        so it can sit next to the camera view in the same window."""
        try:
            buf = self.vis.capture_screen_float_buffer(do_render=True)
        except Exception:
            return None
        img = np.asarray(buf)
        if img.size == 0:
            return None
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        width = max(1, int(img.shape[1] * height / img.shape[0]))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _draw_overlay(self, frame: np.ndarray, now: float) -> None:
        color = {
            "IDLE":      (200, 200, 200),
            "COUNTDOWN": (0, 200, 255),
            "RECORDING": (0, 0, 255),
        }[self.state]
        cv2.putText(frame, self.state, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"signer: {self.args.signer}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"lang: {self.lang}", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"saved: {self.take_count}", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if self.current_label and self.state == "COUNTDOWN":
            rem = max(0.0, COUNTDOWN_S - (now - self.countdown_start))
            self.overlay.draw(frame, f"{self.current_label}  ({rem:.1f}s)",
                              (15, 160), 36, color)
        elif self.current_label and self.state == "RECORDING":
            rem = max(0.0, self.args.duration - (now - self.record_start))
            self.overlay.draw(frame, f"REC: {self.current_label}  ({rem:.1f}s)",
                              (15, 160), 36, color)
        elif self.current_label:
            self.overlay.draw(frame, f"last: {self.current_label}",
                              (15, 160), 28, (180, 180, 180))


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True,
                    help="Your short signer tag — used in filenames.")
    ap.add_argument("--lang", default="khmer",
                    help="ASCII language code (khmer, english, asl, ...).")
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    ap.add_argument("--duration", type=float, default=DEFAULT_RECORD_S)
    ap.add_argument("--synthetic", type=int, default=0, metavar="N",
                    help="auto-generate N synthetic body-variants per real take.")
    ap.add_argument("--mannequin", type=int, default=1, metavar="N",
                    help="show N synthetic mannequins live in an Open3D "
                         "window. Default 1. Pass 0 to disable the 3D window.")
    ap.add_argument("--no-stream", action="store_true",
                    help="(legacy no-op)")
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.json"))
    args = ap.parse_args()

    if not ASCII_SAFE.match(args.lang):
        print(f"ERROR: --lang must be ASCII (got '{args.lang}').", file=sys.stderr)
        sys.exit(2)
    if args.mannequin > 0 and not HAS_OPEN3D:
        print("WARN: --mannequin > 0 but open3d not installed.")
        args.mannequin = 0

    root = Path(args.root)
    cfg = load_config(args.config)
    setup_logging(cfg)
    rng = np.random.default_rng()

    print_session_header(root, args.lang, args.signer)

    capture = LandmarkCapture(cfg)
    if not capture.start():
        print("ERROR: camera failed to start", file=sys.stderr)
        return

    recorder = Recorder(args, cfg, root, args.lang, capture, rng)
    try:
        recorder.run()
    finally:
        recorder.shutdown()
        capture.stop()
        cv2.destroyAllWindows()
        target = root / args.lang
        print(f"\nsession ended. {recorder.take_count} takes saved under {target}.")


if __name__ == "__main__":
    main()
