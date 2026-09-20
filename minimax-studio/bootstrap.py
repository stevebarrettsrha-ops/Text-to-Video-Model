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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
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
    {"id": "rtx", "dir": "comfyui_nvidia_rtx_nodes", "label": "NVIDIA RTX nodes",
     "repo": "https://github.com/NVIDIA/ComfyUI-Nvidia-RTX-Nodes.git",
     "fallback": "https://github.com/nvidia/comfyui_nvidia_rtx_nodes.git",
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

# Published file sizes, so the totals the app shows are real. 0 means "ask
# HuggingFace when it is time to download".
PRECISIONS = {
    "int8": {"label": "INT8 — widest support",
             "note": "Any recent NVIDIA card. 21 GB DiT, 27 GB text encoder.",
             "dit": {"name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                     "size": 21_000_000_000},
             "clip": {"name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                      "size": 27_100_000_000}},
    "fp8": {"label": "FP8 — smaller DiT",
            "note": "Ada (RTX 40) and newer. Same 27 GB text encoder.",
            "dit": {"name": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                    "size": 0},
            "clip": {"name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                     "size": 27_100_000_000}},
    "nvfp4": {"label": "NVFP4 text encoder — Blackwell only",
              "note": "RTX 50 series. Emulated and slow on anything older.",
              "dit": {"name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                      "size": 21_000_000_000},
              "clip": {"name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                       "size": 0}},
}

VIDEO_VAE = {"name": "minimax_h3_video_vae_fp16.safetensors", "size": 4_850_000_000}
AUDIO_VAE = {"name": "minimax_h3_audio_vae_fp32.safetensors", "size": 564_000_000}

TURBO_LORAS = {
    "8step": {"name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
              "repo": MODEL_REPO, "folder": "loras", "steps": 8,
              "label": "8-step turbo", "size": 0},
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


def preflight(cfg: dict) -> dict:
    """Measure the machine and say what it means, without softening it."""
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

    notes, verdict = [], "ok"
    if vram and vram < 12e9:
        verdict = "hard"
        notes.append(f"{vram/1e9:.0f} GB of VRAM. The DiT wants about 12 GB "
                     "resident even in its smallest form, so every step streams "
                     "weights from system memory.")
    elif vram and vram < 20e9:
        verdict = "tight"
        notes.append(f"{vram/1e9:.0f} GB of VRAM — workable with offloading, "
                     "but keep the resolution low.")
    if ram and peak and ram < peak * 1.15:
        verdict = "hard"
        notes.append(f"{ram/1e9:.0f} GB of system RAM against a {peak/1e9:.0f} GB "
                     "peak — the larger of DiT or text encoder, plus the VAEs. "
                     "Expect heavy paging, or an out-of-memory stop.")
    if free_disk and download and free_disk < download * 1.1:
        verdict = "hard"
        notes.append(f"{free_disk/1e9:.0f} GB free where the models go, and the "
                     f"set needs {download/1e9:.0f} GB.")
    if not vram:
        notes.append("Could not read the GPU — install PyTorch, then recheck.")
    if verdict == "hard":
        notes.append("It will still install and queue; it may simply be too slow "
                     "to use. Pointing Settings at a ComfyUI on a rented 24 GB "
                     "box is the way round it.")
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
        self.steps = {k: {"label": v, "state": "pending", "detail": ""}
                      for k, v in self.STEPS}

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

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step,
                    "steps": json.loads(json.dumps(self.steps)),
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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


def wait_for_comfy(url: str, timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip / nodes
# --------------------------------------------------------------------------- #
def pip_install(python: str, args: list[str], log) -> None:
    cmd = [python, "-m", "pip", "install"] + args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(line[:200])
    if proc.wait() != 0:
        raise RuntimeError("pip install failed — see the log.")


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


def clone_node(node: dict, comfy_dir: Path, log) -> Path:
    target = comfy_dir / "custom_nodes" / node["dir"]
    if target.exists():
        log(f"Updating {node['label']}")
        _run(["git", "-C", str(target), "pull", "--ff-only"])
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [node["repo"]] + ([node["fallback"]] if node.get("fallback") else [])
    errors = []
    for url in urls:
        log(f"git clone {url}")
        res = _run(["git", "clone", "--depth", "1", url, str(target)])
        if res.returncode == 0:
            return target
        errors.append((res.stderr or res.stdout)[-200:])
        shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError(f"Could not download {node['label']}: " + " | ".join(errors))


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
                    prog.detail("comfyui", "Downloading ComfyUI…")
                    res = _run(["git", "clone", "--depth", "1", COMFY_REPO,
                                str(comfy_dir)])
                    if res.returncode != 0:
                        raise RuntimeError("git clone failed: " +
                                           (res.stderr or res.stdout)[-600:])
                else:
                    prog.detail("comfyui", "Updating ComfyUI…")
                    _run(["git", "-C", str(comfy_dir), "pull", "--ff-only"])
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
            for node in wanted:
                prog.detail("nodes", f"Installing {node['label']}…")
                try:
                    node_paths.append(clone_node(node, comfy_dir, prog.log))
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
                prog.detail("deps", "Installing PyTorch — the long one…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"], prog.log)
                args = ["torch", "torchvision", "torchaudio"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                pip_install(str(target), args, prog.log)
                prog.detail("deps", "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(Path(cfg["comfy_dir"]) / "requirements.txt")],
                            prog.log)
            cfg["python"] = str(target)
            for path in node_paths:
                reqs = path / "requirements.txt"
                if reqs.exists():
                    prog.detail("deps", f"Requirements for {path.name}…")
                    try:
                        pip_install(str(target), ["-r", str(reqs)], prog.log)
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
            gb = sum(m["size"] for m in todo) / 1e9
            prog.log(f"{len(todo)} file(s) to download"
                     + (f", about {gb:.0f} GB" if gb else ""))
            for item in todo:
                def on_prog(got, total, speed, eta, _n=item["name"]):
                    prog.detail("models",
                                f"{_n} — {got/1e9:.1f} of {total/1e9:.1f} GB · "
                                f"{speed/1e6:.1f} MB/s · "
                                f"{int(eta//60)}m {int(eta%60)}s left")

                download_file(cfg, item["repo"], item["path"],
                              model_path(models_dir, item), on_prog)
                prog.log(f"Downloaded {item['name']}")
            prog.finish("models", "Weights ready")

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
            prog.detail("launch", "Waiting for ComfyUI…")
            if not wait_for_comfy(url, timeout=900):
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
