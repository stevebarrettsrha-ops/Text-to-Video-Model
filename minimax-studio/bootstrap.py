"""
bootstrap.py - first-launch setup for MiniMax Studio.

MiniMax H3 is a big model and the weights are the floor: the INT8 DiT is 21 GB
and the Qwen3-VL-32B text encoder is 27 GB, and ComfyUI has to hold one of those
at a time. preflight() measures the machine and says plainly what that means,
before fifty-odd GB is downloaded.

Steps: Python -> ComfyUI -> custom nodes -> dependencies -> weights -> launch.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

APP_DIR = Path(__file__).resolve().parent
# Config, gallery and finished clips. MINIMAX_STUDIO_DATA moves the lot, which
# is what lets the tests run against a throwaway folder — as in the sibling apps.
DATA_DIR = Path(os.environ.get("MINIMAX_STUDIO_DATA") or (APP_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
HF_BASE = "https://huggingface.co"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
TURBO_REPO = "lightx2v/Minimax-h3-Turbo"
UPSCALER_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"

CUSTOM_NODES = [
    {"id": "kjnodes", "dir": "ComfyUI-KJNodes", "label": "ComfyUI-KJNodes",
     "repo": "https://github.com/kijai/ComfyUI-KJNodes.git", "fallback": "",
     "why": "Live preview while a clip renders. Optional.", "optional": True},
    # Comfy-Org publishes these (registry id comfyui_nvidia_rtx_nodes); the
    # NVIDIA org has no such repo and two guessed URLs there 404ed every
    # install until a real machine hit it.
    {"id": "rtx", "dir": "Nvidia_RTX_Nodes_ComfyUI", "label": "NVIDIA RTX nodes",
     "repo": "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git",
     "fallback": "",
     "why": "RTX Video Super Resolution — upscales a finished clip on the "
            "NVIDIA SDK rather than on diffusion weights, so it is comfortable "
            "on a small card.",
     "optional": True},
    {"id": "manager", "dir": "ComfyUI-Manager", "label": "ComfyUI-Manager",
     "repo": "https://github.com/Comfy-Org/ComfyUI-Manager.git",
     "fallback": "https://github.com/ltdrdata/ComfyUI-Manager.git",
     "why": "Installs and updates other nodes from inside ComfyUI.",
     "optional": True},
]

# Published file sizes, so the totals the app shows are real — measured from
# files on disk after a real install, not guessed. 0 means "ask HuggingFace
# when it is time to download".
PRECISIONS = {
    "int8": {"label": "INT8 — widest support",
             "note": "Any recent NVIDIA card. 21 GB DiT, 27 GB text encoder.",
             "dit": {"name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                     "size": 21_000_000_000},
             "clip": {"name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                      "size": 27_140_000_000}},
    "fp8": {"label": "FP8 — faster on Ada",
            "note": "RTX 40 and newer. 21 GB DiT (same size, faster fp8 "
                    "kernels), same 27 GB text encoder.",
            "dit": {"name": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                    "size": 20_960_000_000},
            "clip": {"name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                     "size": 27_140_000_000}},
    "nvfp4": {"label": "NVFP4 text encoder — Blackwell only",
              "note": "RTX 50 series. Emulated and slow on anything older.",
              "dit": {"name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                      "size": 21_000_000_000},
              "clip": {"name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                       "size": 0}},
}

VIDEO_VAE = {"name": "minimax_h3_video_vae_fp16.safetensors", "size": 5_210_000_000}
AUDIO_VAE = {"name": "minimax_h3_audio_vae_fp32.safetensors", "size": 605_000_000}

TURBO_LORAS = {
    "8step": {"name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
              "repo": MODEL_REPO, "folder": "loras", "steps": 8,
              "label": "8-step turbo", "size": 1_960_000_000},
    "4step": {"name": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
              "repo": TURBO_REPO, "folder": "", "steps": 4,
              "label": "4-step turbo (768p)", "size": 0},
    "ref2v4": {"name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
               "repo": MODEL_REPO, "folder": "loras", "steps": 4,
               "label": "4-step turbo (ref2v)", "size": 0},
}

PREVIEW_TAE = {"name": "taeh3.safetensors", "repo": MODEL_REPO,
               "folder": "vae_approx", "path": "vae_approx/taeh3.safetensors"}

DEFAULT_CONFIG = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_dir": "", "models_dir": "", "python": "",
    "managed": True, "auto_start_comfy": True, "torch_index": "",
    "hf_token": "", "hf_endpoint": HF_BASE, "hf_repo": MODEL_REPO,
    "precision": "int8", "turbo": "8step",
    "want_kjnodes": True, "want_rtx": True, "want_manager": True,
    "lowvram": True, "setup_complete": False,
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# model set
# --------------------------------------------------------------------------- #
def model_set(cfg: dict) -> list[dict]:
    p = PRECISIONS.get(cfg.get("precision") or "int8", PRECISIONS["int8"])
    lora = TURBO_LORAS.get(cfg.get("turbo") or "8step", TURBO_LORAS["8step"])
    return [
        {"folder": "diffusion_models", "repo": MODEL_REPO,
         "path": f"diffusion_models/{p['dit']['name']}", "name": p["dit"]["name"],
         "size": p["dit"]["size"], "role": "required",
         "why": "The video model — reference images and prompt in, video out."},
        {"folder": "text_encoders", "repo": MODEL_REPO,
         "path": f"text_encoders/{p['clip']['name']}", "name": p["clip"]["name"],
         "size": p["clip"]["size"], "role": "required",
         "why": "Qwen3-VL-32B. Reads the prompt and the reference images."},
        {"folder": "vae", "repo": MODEL_REPO, "path": f"vae/{VIDEO_VAE['name']}",
         "name": VIDEO_VAE["name"], "size": VIDEO_VAE["size"], "role": "required",
         "why": "Decodes the picture."},
        {"folder": "vae", "repo": MODEL_REPO, "path": f"vae/{AUDIO_VAE['name']}",
         "name": AUDIO_VAE["name"], "size": AUDIO_VAE["size"], "role": "required",
         "why": "Decodes the sound — H3 generates audio with the video."},
        {"folder": "loras", "repo": lora["repo"],
         "path": (f"{lora['folder']}/{lora['name']}" if lora["folder"]
                  else lora["name"]),
         "name": lora["name"], "size": lora["size"], "role": "required",
         "why": f"{lora['label']} — {lora['steps']} steps instead of forty."},
    ]


def model_path(models_dir: Path, item: dict) -> Path:
    return models_dir / item["folder"] / item["name"]


def missing_models(models_dir: Path, cfg: dict) -> list[dict]:
    return [m for m in model_set(cfg) if not model_path(models_dir, m).exists()]


def node_installed(comfy_dir: Path, node: dict) -> bool:
    return (comfy_dir / "custom_nodes" / node["dir"]).is_dir()


def wanted_nodes(cfg: dict) -> list[dict]:
    keys = {"kjnodes": "want_kjnodes", "rtx": "want_rtx", "manager": "want_manager"}
    return [n for n in CUSTOM_NODES if cfg.get(keys.get(n["id"], ""), True)]


# --------------------------------------------------------------------------- #
# human-readable numbers
# --------------------------------------------------------------------------- #
def fmt_size(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


def fmt_eta(seconds: float) -> str:
    s = int(max(seconds or 0, 0))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m left"
    if s >= 60:
        return f"{s // 60}m {s % 60}s left"
    return f"{s}s left"


def fmt_transfer(got: float, total: float, speed: float, eta: float) -> str:
    """One line of download state: how much, how fast, how much longer."""
    bits = [f"{fmt_size(got)} of {fmt_size(total)}" if total
            else f"{fmt_size(got)} so far"]
    if speed > 0:
        bits.append(f"{speed / 1e6:.1f} MB/s")
    if total and speed > 0:
        bits.append(fmt_eta(eta))
    return " · ".join(bits)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def _ram_bytes() -> int:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MS()
            stat.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 0
    return 0


def _vram_bytes(python: str) -> tuple[int, str]:
    if not python or not Path(python).exists():
        return 0, ""
    code = ("import torch,json;d=torch.cuda.is_available();"
            "print(json.dumps({'v':(torch.cuda.get_device_properties(0)"
            ".total_memory if d else 0),"
            "'n':(torch.cuda.get_device_name(0) if d else '')}))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=90)
        if out.returncode != 0:
            return 0, ""
        d = json.loads(out.stdout.strip().splitlines()[-1])
        return int(d["v"]), d["n"]
    except Exception:
        return 0, ""


def assess(vram: int, ram: int, free_disk: int, download: int,
           peak: int) -> tuple[str, list[str]]:
    """The verdict from the measurements — calibrated against a real run.

    An RTX 4060 (8 GB) with 32 GB of RAM renders H3 shots in minutes with
    exactly this app's defaults: INT8 weights, the 8-step turbo LoRA,
    low-VRAM mode, 0.2 MP, the latent upscale pass, RTX upscale after. So
    8 GB is "tight — proven workable", not "hard". "Hard" is kept for what
    genuinely blocks or breaks: less VRAM than that proven floor, too little
    RAM to page through, no disk for the download.
    """
    notes, verdict = [], "ok"

    def worse(level: str) -> None:
        nonlocal verdict
        order = ("ok", "tight", "hard")
        if order.index(level) > order.index(verdict):
            verdict = level

    if vram and vram < 7e9:
        worse("hard")
        notes.append(f"{vram/1e9:.0f} GB of VRAM is under the 8 GB this app "
                     "is tuned for. It will install and queue, but every "
                     "step swaps weights and a shot can take an hour.")
    elif vram and vram < 12e9:
        worse("tight")
        notes.append(f"{vram/1e9:.0f} GB of VRAM — proven workable: with "
                     "low-VRAM mode the INT8 weights stream from system RAM "
                     "and the 8-step turbo keeps a shot to minutes on an "
                     "RTX 4060. Start at 0.2 MP and upscale afterwards.")
    elif vram and vram < 20e9:
        worse("tight")
        notes.append(f"{vram/1e9:.0f} GB of VRAM — comfortable with "
                     "offloading; higher megapixels are on the table.")
    if ram and peak and ram < peak * 0.7:
        worse("hard")
        notes.append(f"{ram/1e9:.0f} GB of system RAM against a "
                     f"{peak/1e9:.0f} GB peak — the larger of DiT or text "
                     "encoder, plus the VAEs. That is not enough to page "
                     "through; expect out-of-memory stops.")
    elif ram and peak and ram < peak * 1.15:
        worse("tight")
        notes.append(f"{ram/1e9:.0f} GB of system RAM against a "
                     f"{peak/1e9:.0f} GB peak — the bigger loads page "
                     "through disk. Slower, not impossible; a fast SSD "
                     "matters more than the number here.")
    if free_disk and download and free_disk < download * 1.1:
        worse("hard")
        notes.append(f"{free_disk/1e9:.0f} GB free where the models go, and "
                     f"the set needs {download/1e9:.0f} GB.")
    if not vram:
        notes.append("Could not read the GPU — install PyTorch, then recheck.")
    if verdict == "tight":
        notes.append("If a shot is still too slow, render small and press "
                     "RTX upscale ×2 after — that pass runs on the NVIDIA "
                     "SDK over decoded frames and costs almost no VRAM.")
    if verdict == "hard":
        notes.append("It will still install and queue; it may simply be too "
                     "slow to use. Pointing Settings at a ComfyUI on a "
                     "rented 24 GB box is the way round it.")
    return verdict, notes


def preflight(cfg: dict) -> dict:
    """Measure the machine and say what that means for this model."""
    vram, gpu = _vram_bytes(comfy_python(cfg))
    ram = _ram_bytes()
    try:
        free_disk = shutil.disk_usage(cfg.get("models_dir") or str(APP_DIR)).free
    except Exception:
        free_disk = 0

    items = model_set(cfg)
    download = sum(i["size"] for i in items)
    dit = next((i["size"] for i in items if i["folder"] == "diffusion_models"), 0)
    clip = next((i["size"] for i in items if i["folder"] == "text_encoders"), 0)
    peak = max(dit, clip) + VIDEO_VAE["size"] + AUDIO_VAE["size"]

    verdict, notes = assess(vram, ram, free_disk, download, peak)
    return {"vram": vram, "gpu": gpu, "ram": ram, "free_disk": free_disk,
            "download": download, "peak": peak, "verdict": verdict,
            "notes": notes, "precision": cfg.get("precision", "int8"),
            "turbo": cfg.get("turbo", "8step")}


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class Progress:
    STEPS = [("python", "Check Python"), ("comfyui", "Install ComfyUI"),
             ("nodes", "Install the custom nodes"),
             ("deps", "Install dependencies"),
             ("models", "Download the MiniMax H3 weights"),
             ("launch", "Start ComfyUI")]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.done = False
        self.error: str | None = None
        self.step = ""
        self.steps = {k: {"key": k, "label": v, "state": "pending",
                          "detail": "", "pct": None} for k, v in self.STEPS}

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 4000:
                del self.lines[:2000]
        print(f"[setup] {msg}", flush=True)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.step = key
            self.steps[key]["state"] = "running"
            self.steps[key]["detail"] = detail
            self.steps[key]["pct"] = None

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def track(self, key: str, pct: float | None, detail: str = "") -> None:
        """Move a step's bar. `pct` None means running with no number yet —
        the front end shows an indeterminate bar rather than a fake 0%."""
        with self._lock:
            self.steps[key]["pct"] = (None if pct is None
                                      else round(max(0.0, min(100.0, pct)), 1))
            if detail:
                self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            self.steps[key]["pct"] = None
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["pct"] = None
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            # A list, in the order the steps actually happen: jsonify sorts
            # dict keys, which listed Check Python last on the one screen
            # where order is the whole point.
            steps = [dict(self.steps[k]) for k, _ in self.STEPS]
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step, "steps": steps,
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _stream(cmd: list[str], on_line, cwd: str | None = None,
            env: dict | None = None) -> int:
    r"""Run `cmd` and hand every line of its output to `on_line` as it appears.

    Splits on carriage returns as well as newlines: git writes its progress by
    rewriting one line with \r, so a plain line iterator would hold all of it
    back until the clone finished — which is exactly the silence this is meant
    to fill. Reads with read1() so a partial block is delivered straight away.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=cwd, env=env)
    assert proc.stdout
    buf = b""
    while True:
        block = proc.stdout.read1(8192)
        if not block:
            break
        buf += block
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                on_line(text)
    if buf.strip():
        on_line(buf.decode("utf-8", "replace").strip())
    return proc.wait()


# git reports each phase as its own 0-100%; "Receiving objects" is the download.
GIT_PHASE = re.compile(r"(Counting objects|Compressing objects|Receiving objects"
                       r"|Resolving deltas|Updating files):\s+(\d+)%")


def git_run(cmd: list[str], log, on_pct=None) -> tuple[int, str]:
    """A git command with its progress forwarded. Returns (code, last output)."""
    tail: list[str] = []

    def line(text: str) -> None:
        m = GIT_PHASE.search(text)
        if m:
            if on_pct:
                on_pct(m.group(1), float(m.group(2)))
            return
        tail.append(text)
        log(text[:200])

    # C locale: the phase names above are what git prints in English, and a
    # translated git would otherwise report no progress at all.
    code = _stream(cmd, line, env=dict(os.environ, LC_ALL="C"))
    return code, "\n".join(tail[-8:])


def git_clone(url: str, target: Path, log, on_pct=None,
              depth: int = 1) -> tuple[int, str]:
    return git_run(["git", "clone", "--depth", str(depth), "--progress",
                    url, str(target)], log, on_pct)


def find_python(prog: Progress | None = None) -> str:
    candidates: list[list[str]] = [[sys.executable]]
    if platform.system() == "Windows":
        candidates += [["py", "-3.12"], ["py", "-3.11"], ["py", "-3.10"],
                       ["py", "-3"], ["python"]]
    else:
        candidates += [["python3.12"], ["python3.11"], ["python3.10"],
                       ["python3"], ["python"]]
    for cand in candidates:
        try:
            out = _run(cand + ["-c", "import sys;print(sys.executable);"
                                     "print('%d.%d' % sys.version_info[:2])"],
                       timeout=25)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        parts = [p.strip() for p in out.stdout.strip().splitlines() if p.strip()]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            major, minor = (int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        if (major, minor) >= (3, 10):
            if prog:
                prog.log(f"Using Python {parts[1]} at {parts[0]}")
            return parts[0]
    raise RuntimeError("No Python 3.10 or newer found. Install it from "
                       "python.org, tick 'Add to PATH', and run setup again.")


def portable_python(comfy_dir: Path) -> Path | None:
    for base in (comfy_dir.parent, comfy_dir):
        cand = base / "python_embeded" / "python.exe"
        if cand.exists():
            return cand
    return None


def venv_python(comfy_dir: Path) -> Path:
    venv = comfy_dir.parent / "comfy-venv"
    return venv / ("Scripts/python.exe" if platform.system() == "Windows"
                   else "bin/python")


def comfy_python(cfg: dict) -> str:
    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir:
        p = portable_python(comfy_dir)
        if p:
            return str(p)
        v = venv_python(comfy_dir)
        if v.exists():
            return str(v)
    return cfg.get("python") or ""


def have_git() -> bool:
    return shutil.which("git") is not None


def detect_comfy_dirs() -> list[str]:
    home = Path.home()
    cands = [APP_DIR / "ComfyUI", home / "ComfyUI",
             home / "Documents" / "ComfyUI", home / "Desktop" / "ComfyUI",
             Path("C:/ComfyUI"), Path("C:/ComfyUI_windows_portable/ComfyUI"),
             Path("D:/ComfyUI"), Path("D:/ComfyUI_windows_portable/ComfyUI")]
    appdata, local = os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")
    if appdata:
        cands.append(Path(appdata) / "ComfyUI")
    if local:
        cands.append(Path(local) / "Programs" / "@comfyorgcomfyui-electron"
                     / "resources" / "ComfyUI")
    out, seen = [], set()
    for c in cands:
        try:
            if ((c / "main.py").exists() or (c / "models").is_dir()) \
                    and str(c) not in seen:
                seen.add(str(c))
                out.append(str(c))
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")


def hf_tree(cfg: dict, repo: str, revision: str = "main") -> list[dict]:
    base = hf_endpoint(cfg)
    last = ""
    for kind in ("models", "datasets"):
        url = f"{base}/api/{kind}/{repo}/tree/{revision}?recursive=1"
        try:
            r = requests.get(url, headers=hf_headers(cfg), timeout=30)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            continue
        if r.status_code == 401:
            raise RuntimeError("This repo needs a HuggingFace token. Add one on "
                               "the Models page, then try again.")
        if r.status_code == 403:
            raise RuntimeError("Your token cannot read this repo. MiniMax H3 is "
                               "under a community licence — accept it on the "
                               "model page first.")
        if r.status_code == 404:
            continue
        r.raise_for_status()
        files = []
        for e in r.json():
            if e.get("type") != "file":
                continue
            size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
            files.append({"path": e["path"], "size": size})
        return files
    raise RuntimeError(f"Could not find '{repo}' on {base}. "
                       + (last or "Check the spelling, or add a token."))


def download_file(cfg: dict, repo: str, path: str, dest: Path,
                  on_progress=None, should_cancel=None,
                  revision: str = "main") -> None:
    import urllib.parse
    url = (f"{hf_endpoint(cfg)}/{repo}/resolve/{revision}/"
           + urllib.parse.quote(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(hf_headers(cfg))
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            part.replace(dest)
            return
        if r.status_code in (401, 403):
            raise RuntimeError("HuggingFace refused the download. Accept the "
                               "MiniMax H3 licence on the model page, then add "
                               "a token on the Models page.")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0
        got, last, started = have, 0.0, time.time()
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if should_cancel and should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.6:
                    last = now
                    speed = (got - have) / max(now - started, .1)
                    eta = (total - got) / speed if speed > 0 and total else 0
                    on_progress(got, total, speed, eta)
    part.replace(dest)


# --------------------------------------------------------------------------- #
# ComfyUI process
# --------------------------------------------------------------------------- #
class ComfyProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def note(self, msg: str) -> None:
        """An app-side line in the engine console — what the app is doing TO
        the engine belongs next to what the engine itself says."""
        with self._lock:
            self.lines.append(f"[MiniMax Studio] {msg}")
            if len(self.lines) > 2000:
                del self.lines[:1000]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, python: str, comfy_dir: Path, port: int, prog: Progress,
              lowvram: bool = True) -> None:
        if self.alive():
            return
        cmd = [python, "main.py", "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch"]
        if lowvram:
            # Weights stream from system RAM rather than sitting in VRAM, and
            # nothing is cached between runs. Slower, but it is what makes a
            # 21 GB DiT possible on a small card at all.
            cmd += ["--lowvram", "--cache-none"]
        prog.log("Launching ComfyUI: " + " ".join(cmd))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if platform.system() == "Windows" else 0
        self.proc = subprocess.Popen(cmd, cwd=str(comfy_dir),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, creationflags=flags)
        threading.Thread(target=self._pump, args=(prog,), daemon=True).start()

    def _pump(self, prog: Progress) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            with self._lock:
                self.lines.append(line)
                if len(self.lines) > 2000:
                    del self.lines[:1000]
            if any(k in line for k in ("Error", "Traceback", "error:",
                                       "IMPORT FAILED", "Starting server",
                                       "out of memory")):
                prog.log(f"ComfyUI: {line}")

    def tail(self, n: int = 40) -> list[str]:
        with self._lock:
            return self.lines[-n:]

    def stop(self) -> None:
        if self.alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def comfy_online(url: str) -> bool:
    try:
        return requests.get(f"{url}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def _pids_from_proc_net(port: int) -> list[int]:
    """Linux, no tools needed: the socket inode from /proc/net/tcp*, then the
    process whose fd table holds it."""
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and local.rsplit(":", 1)[-1] == f"{port:04X}":
                inodes.add(inode)
    pids = set()
    if not inodes:
        return []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if any(f"socket:[{i}]" == target for i in inodes):
                    pids.add(int(proc.name))
                    break
        except OSError:
            continue
    return sorted(pids)


def port_pids(port: int) -> list[int]:
    """Whoever is listening on the port."""
    if platform.system() == "Windows":
        pids = set()
        try:
            out = _run(["netstat", "-ano", "-p", "TCP"], timeout=25).stdout
        except Exception:
            return []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" \
                    and parts[3] == "LISTENING" \
                    and parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
        return sorted(pids)
    found = _pids_from_proc_net(port)
    if found:
        return found
    if shutil.which("lsof"):
        try:
            out = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                       timeout=25).stdout
            return sorted({int(t) for t in out.split() if t.strip().isdigit()})
        except Exception:
            pass
    return []


def pid_cmdline(pid: int) -> str:
    try:
        if platform.system() == "Windows":
            out = _run(["wmic", "process", "where", f"processid={pid}",
                        "get", "commandline"], timeout=25).stdout
            lines = [ln.strip() for ln in out.splitlines()
                     if ln.strip() and "CommandLine" not in ln]
            return lines[0] if lines else ""
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists():
            return cmd.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
        return _run(["ps", "-p", str(pid), "-o", "command="],
                    timeout=25).stdout.strip()
    except Exception:
        return ""


def kill_pid(pid: int) -> str:
    """Stop a process: politely first, firmly if it lingers. Returns what the
    system said about it, so a refusal (access denied, already gone) can be
    shown instead of guessed at."""
    if platform.system() == "Windows":
        try:
            out = _run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=30)
            return (out.stdout or out.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            return str(exc)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        return "access denied"
    for _ in range(25):
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "stopped"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "access denied"
    return "sent SIGKILL"


def comfy_stats(url: str) -> dict | None:
    """What is actually answering on the address — argv says which install."""
    try:
        r = requests.get(f"{url}/system_stats", timeout=3)
        if r.status_code == 200:
            return r.json().get("system") or {}
    except Exception:
        pass
    return None


def wait_for_comfy(url: str, timeout: int = 900, on_wait=None) -> bool:
    """Poll until ComfyUI answers. `on_wait(elapsed, timeout)` runs each pass —
    there is no percentage to give here, only how long it has been waiting."""
    started = time.time()
    deadline = started + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        if on_wait:
            on_wait(time.time() - started, timeout)
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip / nodes
# --------------------------------------------------------------------------- #
# pip's own progress bar is a terminal animation and vanishes when its output
# is a pipe, which is why a 2.4 GB torch wheel looks like a hang. `--progress-bar
# raw` makes it print "Progress <done> of <total>" lines instead, which survive
# the pipe. Older pips do not have it, so ask before using it.
PIP_RAW = re.compile(r"^Progress (\d+) of (\d+)$")
PIP_GET = re.compile(r"^\s*(?:Downloading|Using cached)\s+(\S+)")
_PIP_RAW_OK: dict[str, bool] = {}


def pip_has_raw_progress(python: str) -> bool:
    if python not in _PIP_RAW_OK:
        ok = False
        try:
            out = _run([python, "-m", "pip", "install", "--help"], timeout=60)
            at = out.stdout.find("--progress-bar")
            ok = at >= 0 and "raw" in out.stdout[at:at + 300]
        except Exception:  # noqa: BLE001
            ok = False
        _PIP_RAW_OK[python] = ok
    return _PIP_RAW_OK[python]


def pip_install(python: str, args: list[str], log, on_pct=None) -> None:
    """Install with pip. `on_pct(pct|None, detail)` is called as it downloads."""
    import urllib.parse
    cmd = [python, "-m", "pip", "install"]
    if on_pct and pip_has_raw_progress(python):
        cmd += ["--progress-bar", "raw"]
    cmd += args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))

    # Per-wheel transfer state: pip restarts the counter for every file.
    cur = {"name": "", "base": 0, "started": 0.0, "last": 0.0}

    def line(text: str) -> None:
        m = PIP_RAW.match(text)
        if m:
            if not on_pct:
                return
            got, total = int(m.group(1)), int(m.group(2))
            now = time.time()
            if got < cur["base"] or not cur["started"]:
                cur["base"], cur["started"] = got, now       # a new file
            if now - cur["last"] < 0.4 and not (total and got >= total):
                return
            cur["last"] = now
            speed = (got - cur["base"]) / max(now - cur["started"], .1)
            eta = (total - got) / speed if speed > 0 and total else 0
            label = cur["name"] or "package"
            on_pct((got / total * 100) if total else None,
                   f"{label} — {fmt_transfer(got, total, speed, eta)}")
            return
        m = PIP_GET.match(text)
        if m:
            cur.update(name=urllib.parse.unquote(
                m.group(1).rsplit("/", 1)[-1])[:60],
                base=0, started=0.0, last=0.0)
        if text.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(text[:200])
            if on_pct and text.startswith(("Installing", "Building")):
                # Unpacking and byte-compiling: no byte count to report, and
                # torch takes minutes over it, so say what is happening.
                on_pct(None, text[:120])

    if _stream(cmd, line) != 0:
        raise RuntimeError("pip install failed — see the log.")
    if "pip" in args:
        # pip just upgraded itself, so whether it can report progress may have
        # changed. The very first install in a new venv is that upgrade.
        _PIP_RAW_OK.pop(python, None)


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    if shutil.which("nvidia-smi"):
        try:
            if _run(["nvidia-smi"], timeout=20).returncode == 0:
                return "https://download.pytorch.org/whl/cu128"
        except Exception:
            pass
    return "https://download.pytorch.org/whl/cpu"


def clone_node(node: dict, comfy_dir: Path, log, on_pct=None) -> Path:
    target = comfy_dir / "custom_nodes" / node["dir"]
    if target.exists():
        log(f"Updating {node['label']}")
        git_run(["git", "-C", str(target), "pull", "--ff-only", "--progress"],
                log, on_pct)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [node["repo"]] + ([node["fallback"]] if node.get("fallback") else [])
    errors = []
    for url in urls:
        log(f"git clone {url}")
        code, out = git_clone(url, target, log, on_pct)
        if code == 0:
            return target
        errors.append(out[-300:])
        shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError(f"Could not download {node['label']}: " + " | ".join(errors))


# --------------------------------------------------------------------------- #
# progress adapters
# --------------------------------------------------------------------------- #
# Each git phase is its own 0-100%, so stack them into one bar that only ever
# moves forwards. Receiving objects is the transfer and takes nearly all of it.
GIT_WEIGHT = {"Counting objects": (0.00, 0.02), "Compressing objects": (0.02, 0.03),
              "Receiving objects": (0.05, 0.90), "Resolving deltas": (0.95, 0.04),
              "Updating files": (0.95, 0.05)}


def _git_pct(prog: Progress, key: str, head: str = "",
             base: float = 0.0, span: float = 100.0):
    """A git on_pct callback that drives `key`'s bar between base and base+span."""
    def on_pct(phase: str, pct: float) -> None:
        start, width = GIT_WEIGHT.get(phase, (0.0, 0.0))
        line = f"{phase} — {pct:.0f}%"
        prog.track(key, base + (start + width * pct / 100) * span,
                   f"{head} · {line}" if head else line)
    return on_pct


def _pip_pct(prog: Progress, head: str, key: str = "deps"):
    """A pip on_pct callback: pct is None while pip is not transferring bytes."""
    def on_pct(pct: float | None, detail: str) -> None:
        prog.track(key, pct, f"{head} · {detail}" if head else detail)
    return on_pct


# --------------------------------------------------------------------------- #
# setup run
# --------------------------------------------------------------------------- #
def run_setup(cfg: dict, prog: Progress, comfy: ComfyProcess,
              chosen_dir: str = "", mode: str = "auto") -> None:
    prog.running = True
    prog.done = False
    prog.error = None
    try:
        prog.begin("python")
        if mode == "external":
            prog.finish("python", "Not needed — you run ComfyUI yourself")
            py = cfg.get("python") or sys.executable
        else:
            py = find_python(prog)
            cfg["python"] = py
            prog.finish("python", py)

        prog.begin("comfyui")
        comfy_dir = None
        if mode == "external":
            if not comfy_online(cfg["comfy_url"]):
                raise RuntimeError(f"Nothing is answering at {cfg['comfy_url']}.")
            if not cfg.get("models_dir"):
                raise RuntimeError("Set the ComfyUI models folder in Settings.")
            cfg["managed"] = False
            if cfg.get("comfy_dir"):
                comfy_dir = Path(cfg["comfy_dir"])
            prog.finish("comfyui", cfg["comfy_url"])
        else:
            if chosen_dir:
                comfy_dir = Path(chosen_dir)
                cfg["managed"] = False
                prog.log(f"Using existing ComfyUI at {comfy_dir}")
            else:
                comfy_dir = APP_DIR / "ComfyUI"
                cfg["managed"] = True
                if not (comfy_dir / "main.py").exists():
                    if not have_git():
                        raise RuntimeError("Git is not installed. Install it "
                                           "from the Engine page first.")
                    prog.track("comfyui", None, "Downloading ComfyUI…")
                    code, out = git_clone(COMFY_REPO, comfy_dir, prog.log,
                                          _git_pct(prog, "comfyui"))
                    if code != 0:
                        raise RuntimeError("git clone failed: " + out[-600:])
                else:
                    prog.track("comfyui", None, "Updating ComfyUI…")
                    git_run(["git", "-C", str(comfy_dir), "pull", "--ff-only",
                             "--progress"], prog.log,
                            _git_pct(prog, "comfyui"))
            if not (comfy_dir / "main.py").exists():
                raise RuntimeError(f"No main.py in {comfy_dir}.")
            cfg["comfy_dir"] = str(comfy_dir)
            cfg["models_dir"] = str(comfy_dir / "models")
            prog.finish("comfyui", str(comfy_dir))

        models_dir = Path(cfg["models_dir"])

        prog.begin("nodes")
        node_paths: list[Path] = []
        if comfy_dir is None:
            prog.finish("nodes", "Install the nodes in your own ComfyUI")
        else:
            wanted = wanted_nodes(cfg)
            if wanted and not have_git():
                raise RuntimeError("Git is needed to install the custom nodes.")
            for index, node in enumerate(wanted):
                # one bar across all the nodes: each clone gets its slice
                prog.track("nodes", index / len(wanted) * 100,
                           f"Installing {node['label']}…")
                try:
                    node_paths.append(clone_node(
                        node, comfy_dir, prog.log,
                        _git_pct(prog, "nodes", node["label"],
                                 base=index / len(wanted) * 100,
                                 span=100 / len(wanted))))
                except Exception as exc:  # noqa: BLE001
                    if node.get("optional"):
                        prog.log(f"Skipped {node['label']}: {exc}")
                    else:
                        raise
            prog.finish("nodes", ", ".join(n["label"] for n in wanted) or "none")

        prog.begin("deps")
        if mode == "external":
            prog.finish("deps", "Handled by your own ComfyUI install")
        else:
            target = portable_python(Path(cfg["comfy_dir"]))
            if target:
                prog.log(f"Portable ComfyUI detected — installing into {target}")
            else:
                vpy = venv_python(Path(cfg["comfy_dir"]))
                if not vpy.exists():
                    prog.detail("deps", "Creating the Python environment…")
                    res = _run([py, "-m", "venv",
                                str(Path(cfg["comfy_dir"]).parent / "comfy-venv")])
                    if res.returncode != 0:
                        raise RuntimeError("venv creation failed: " +
                                           (res.stderr or res.stdout)[-600:])
                target = vpy
                prog.track("deps", None, "Upgrading pip…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"],
                            prog.log, _pip_pct(prog, "pip"))
                args = ["torch", "torchvision", "torchaudio"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                prog.track("deps", None, "Installing PyTorch — the long one…")
                pip_install(str(target), args, prog.log,
                            _pip_pct(prog, "PyTorch"))
                prog.track("deps", None, "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(Path(cfg["comfy_dir"]) / "requirements.txt")],
                            prog.log, _pip_pct(prog, "ComfyUI requirements"))
            cfg["python"] = str(target)
            for path in node_paths:
                reqs = path / "requirements.txt"
                if reqs.exists():
                    prog.track("deps", None, f"Requirements for {path.name}…")
                    try:
                        pip_install(str(target), ["-r", str(reqs)], prog.log,
                                    _pip_pct(prog, path.name))
                    except Exception as exc:  # noqa: BLE001
                        prog.log(f"Skipped {path.name} requirements: {exc}")
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        prog.begin("models")
        todo = missing_models(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            for note in preflight(cfg)["notes"]:
                prog.log("Preflight: " + note)
            # Ask each repo for the real sizes so one bar can cover the whole
            # set. The hard-coded sizes are rough and some are zero, and a bar
            # that only knows the current file jumps back to 0% five times.
            sizes: dict[str, int] = {}
            prog.track("models", None, "Checking what is on the repo…")
            for repo in {m["repo"] for m in todo}:
                try:
                    for f in hf_tree(cfg, repo):
                        sizes[f"{repo}/{f['path']}"] = f["size"]
                except Exception as exc:  # noqa: BLE001
                    prog.log(f"Could not read {repo}'s file list ({exc}). The "
                             "bar will follow one file at a time instead.")
            plan = [(item, int(sizes.get(f"{item['repo']}/{item['path']}")
                               or item.get("size") or 0)) for item in todo]
            grand = sum(s for _, s in plan)
            prog.log(f"{len(plan)} file(s) to download "
                     f"({cfg.get('precision', 'int8')} weights"
                     + (f", {fmt_size(grand)}" if grand else "") + ")")
            done_bytes = 0
            for i, (item, size) in enumerate(plan, 1):
                dest = model_path(models_dir, item)
                head = f"{item['name']} ({i} of {len(plan)})"

                def on_prog(got, total, speed, eta, _head=head):
                    whole = ((done_bytes + got) / grand * 100) if grand else (
                        (got / total * 100) if total else None)
                    prog.track("models", whole,
                               f"{_head} — " + fmt_transfer(got, total,
                                                            speed, eta))

                prog.track("models",
                           (done_bytes / grand * 100) if grand else None,
                           f"{head} — starting…")
                download_file(cfg, item["repo"], item["path"], dest, on_prog)
                done_bytes += size or (dest.stat().st_size
                                       if dest.exists() else 0)
                prog.log(f"Downloaded {item['name']}")
            prog.finish("models", f"{len(plan)} file(s) ready"
                        + (f" · {fmt_size(grand)}" if grand else ""))

        prog.begin("launch")
        url = cfg["comfy_url"]
        if mode == "external" or not cfg.get("auto_start_comfy", True):
            if not comfy_online(url):
                raise RuntimeError(f"ComfyUI is not answering at {url}.")
        elif comfy_online(url):
            prog.log("ComfyUI is already running — restart it so it picks up the "
                     "new nodes and weights.")
        else:
            port = int(url.rsplit(":", 1)[-1])
            comfy.start(cfg["python"], Path(cfg["comfy_dir"]), port, prog,
                        cfg.get("lowvram", True))

            def waiting(elapsed: float, limit: int) -> None:
                s = int(elapsed)
                been = f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"
                prog.track("launch", None, "Waiting for ComfyUI — "
                           f"{been} so far. The first start is slow.")

            prog.track("launch", None, "Waiting for ComfyUI…")
            if not wait_for_comfy(url, timeout=900, on_wait=waiting):
                raise RuntimeError("ComfyUI did not start within 15 minutes.\n"
                                   + "\n".join(comfy.tail(25)))
        prog.finish("launch", url)

        cfg["setup_complete"] = True
        save_config(cfg)
        prog.done = True
        prog.log("Setup complete.")
    except Exception as exc:  # noqa: BLE001
        prog.error = str(exc)
        if prog.step:
            prog.fail(prog.step, str(exc))
        prog.log(f"FAILED: {exc}")
    finally:
        prog.running = False
