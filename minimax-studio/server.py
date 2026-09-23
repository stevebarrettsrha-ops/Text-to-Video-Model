"""
server.py - MiniMax Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7802)
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import requests

from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import manager
from bootstrap import (APP_DIR, ComfyProcess, Progress, comfy_online,
                       comfy_port, detect_comfy_dirs, load_config, normal_url,
                       save_config)
from comfy import ComfyClient, ComfyError

DATA_DIR = bootstrap.DATA_DIR          # honours MINIMAX_STUDIO_DATA
CLIPS_DIR = DATA_DIR / "clips"
GALLERY_PATH = DATA_DIR / "gallery.json"
BOARD_PATH = DATA_DIR / "board.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("MINIMAX_STUDIO_PORT", "7804"))

app = Flask(__name__, static_folder=None)
# jsonify alphabetises dict keys by default, which scrambled the setup steps
# on the one screen where order is the whole point.
app.json.sort_keys = False

cfg = load_config()
progress = Progress()
comfy_proc = ComfyProcess()
client = ComfyClient(cfg["comfy_url"])

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gallery_lock = threading.Lock()
board_lock = threading.Lock()
setup_lock = threading.Lock()
ws_progress: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def read_gallery() -> list[dict]:
    with gallery_lock:
        return _read_gallery_unlocked()


def _read_gallery_unlocked() -> list[dict]:
    """Read the gallery while the caller owns ``gallery_lock``.

    Treat a damaged or manually edited file as empty rather than allowing a
    dict/string to leak into endpoints that expect a list of clip records.
    """
    if not GALLERY_PATH.exists():
        return []
    try:
        value = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_gallery_unlocked(items: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # replace() prevents a crash in the middle of a write from leaving half a
    # JSON document behind.
    temporary = GALLERY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(items, indent=2), encoding="utf-8")
    temporary.replace(GALLERY_PATH)


def write_gallery(items: list[dict]) -> None:
    with gallery_lock:
        _write_gallery_unlocked(items)


def add_images(items: list[dict]) -> None:
    # A batch can launch four render threads. Keep the read/modify/write under
    # one lock or two jobs finishing together can silently discard a clip.
    with gallery_lock:
        _write_gallery_unlocked(items + _read_gallery_unlocked())


def title_from(p: dict) -> str:
    text = (p.get("prompt") or "").strip()
    if text:
        return " ".join(text.split()[:8]).strip(" ,.!?-")
    return "Untitled clip"


# --------------------------------------------------------------------------- #
# step progress over the ComfyUI websocket (optional dependency)
# --------------------------------------------------------------------------- #
def ws_listener() -> None:
    try:
        import websocket  # websocket-client
    except ImportError:
        return
    while True:
        try:
            url = cfg["comfy_url"].replace("http://", "ws://").replace(
                "https://", "wss://")
            ws = websocket.WebSocket()
            ws.connect(f"{url}/ws?clientId={client.client_id}", timeout=10)
            current = None
            while True:
                raw = ws.recv()
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data") or {}
                pid = data.get("prompt_id") or current
                if mtype == "execution_start":
                    current = data.get("prompt_id")
                elif mtype == "progress" and pid:
                    ws_progress.setdefault(pid, {}).update(
                        value=data.get("value", 0), max=data.get("max", 0))
                elif mtype in ("execution_success", "execution_error") and pid:
                    ws_progress.pop(pid, None)
        except Exception:
            time.sleep(4)


# --------------------------------------------------------------------------- #
# generation job
# --------------------------------------------------------------------------- #
def run_job(job_id: str, params: dict) -> None:
    def set_state(**kw):
        if kw.get("status", "running") != "running":
            kw["finished"] = time.time()
        with jobs_lock:
            jobs[job_id].update(kw)

    try:
        set_state(stage="Building the graph", pct=2)
        built = client.build(params) if params.get("kind") != "rtx" \
            else client.build_rtx_upscale(params.get("video", ""),
                                          int(params.get("scale") or 2),
                                          params.get("quality") or "ULTRA")
        prompt_id = client.queue(built["prompt"])
        set_state(prompt_id=prompt_id, seed=built.get("seed"), pct=5,
                  stage="Queued in ComfyUI")

        started = time.time()
        while True:
            time.sleep(1.0)
            with jobs_lock:
                cancelled = jobs[job_id].get("cancelled")
            if cancelled:
                client.cancel(prompt_id)
                set_state(status="cancelled", stage="Cancelled")
                return
            err = client.failed(prompt_id)
            if err:
                set_state(status="error", error=err, stage="Failed")
                return
            outs = client.outputs(prompt_id)
            if outs:
                break
            wp = ws_progress.get(prompt_id) or {}
            value, maximum = wp.get("value", 0), wp.get("max", 0)
            if maximum:
                set_state(pct=round(6 + min(value / maximum, 1) * 88, 1),
                          stage=f"Step {value} of {maximum}")
            else:
                set_state(pct=min(5 + (time.time() - started) / 4, 12),
                          stage="Loading the model")
            if time.time() - started > 6 * 3600:
                set_state(status="error", stage="Timed out",
                          error="Nothing after six hours. On a small card that "
                                "is usually paging rather than rendering — see "
                                "the preflight on the Engine page.")
                return

        set_state(stage="Saving", pct=96)
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for index, item in enumerate(outs):
            image_id = uuid.uuid4().hex[:12]
            ext = Path(item["filename"]).suffix or ".mp4"
            dest = CLIPS_DIR / f"{image_id}{ext}"
            with client.view(item) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1024 * 256):
                        fh.write(chunk)
            saved.append({
                "id": image_id, "file": dest.name,
                "kind": params.get("kind") or "clip",
                "title": params.get("title") or title_from(params),
                "prompt": params.get("prompt", ""),
                "refs": params.get("refs") or [],
                "ref_video": params.get("ref_video") or "",
                "voice": params.get("voice") or "",
                "width": built.get("width") or params.get("width"), "height": built.get("height") or params.get("height"),
                "length": built.get("length") or params.get("length"), "fps": built.get("fps") or params.get("fps"),
                "seconds": built.get("seconds") or params.get("seconds"),
                "megapixels": params.get("megapixels"),
                "aspect": params.get("aspect"),
                "steps": params.get("steps"), "shift": params.get("shift"),
                "sampler": params.get("sampler"),
                "scheduler": params.get("scheduler"),
                "upscaled": built.get("upscaled"),
                "note": built.get("note", ""),
                "seed": built.get("seed"), "batch_index": index,
                "model": (built.get("files") or {}).get("dit", ""),
                "lora": (built.get("files") or {}).get("lora", ""),
                "created": time.time(),
            })
        add_images(saved)
        set_state(status="done", pct=100, stage="Ready", images=saved)
    except ComfyError as exc:
        set_state(status="error", error=str(exc), stage="Failed")
    except Exception as exc:  # noqa: BLE001
        set_state(status="error", error=f"{type(exc).__name__}: {exc}",
                  stage="Failed")


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


@app.before_request
def local_only():
    """Only this machine's own pages may drive the app.

    Binding to 127.0.0.1 is not enough: any site the person visits can post
    to it, and DNS rebinding lets one read the answers — and the app runs
    pip, git and process kills. The Host must name this machine, and a
    request that changes something must come from this app's own page.
    """
    host = (request.host or "").rsplit(":", 1)[0].lower() \
        if not (request.host or "").startswith("[") \
        else (request.host or "").split("]")[0].lower() + "]"
    if host not in LOCAL_HOSTS:
        return jsonify({"error": "MiniMax Studio only answers to localhost."}), 403
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/") \
                and origin.rstrip("/") not in (
                    f"http://{h}:{PORT}" for h in LOCAL_HOSTS):
            return jsonify({"error": "Cross-site request refused."}), 403

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


# --------------------------------------------------------------------------- #
# status / setup
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    online = comfy_online(cfg["comfy_url"])
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    missing = []
    if models_dir and models_dir.is_dir():
        missing = [m["name"] for m in bootstrap.missing_models(models_dir, cfg)]
    payload = {
        "comfy_online": online,
        "setup_complete": bool(cfg.get("setup_complete")),
        "missing_models": missing,
        "detected": detect_comfy_dirs(),
        "precisions": {k: {"label": v["label"], "note": v["note"]}
                       for k, v in bootstrap.PRECISIONS.items()},
        "turbos": {k: {"label": v["label"], "steps": v["steps"]}
                   for k, v in bootstrap.TURBO_LORAS.items()},
        "config": {k: cfg.get(k) for k in
                   ("comfy_url", "comfy_dir", "models_dir", "managed",
                    "auto_start_comfy", "torch_index", "precision", "turbo",
                    "lowvram", "want_kjnodes", "want_rtx", "want_manager")},
        "nodes_ready": False, "ready": False,
    }
    if online:
        try:
            payload["capabilities"] = client.capabilities()
            payload["nodes_ready"] = client.has("MiniMaxH3ReferenceToVideo")
            payload["samplers"] = client.samplers()
            payload["schedulers"] = client.schedulers()
            payload["unets"] = client.unets()
            payload["attentions"] = client.attention_backends()
        except Exception as exc:  # noqa: BLE001
            payload["schema_error"] = str(exc)
        # The two silent "nothing works" states, named. ComfyUI scans its
        # model folders once, at startup: weights that landed later are on
        # disk yet absent from its lists until a restart. And an address can
        # be answered by a different install than the one set up here.
        payload["stale_models"] = bool(
            not missing and models_dir and models_dir.is_dir()
            and not any("minimax" in u.lower()
                        for u in payload.get("unets") or []))
        stats = bootstrap.comfy_stats(cfg["comfy_url"]) or {}
        argv = (stats.get("argv") or [""])[0]
        payload["engine_argv"] = argv
        want = (str(Path(cfg["comfy_dir"])).replace("\\", "/").lower()
                if cfg.get("comfy_dir") else "")
        payload["engine_mismatch"] = bool(
            argv and want and want not in argv.replace("\\", "/").lower())
        # low-VRAM mode is configured, but the engine answering lacks it
        payload["engine_lowvram_off"] = bool(
            cfg.get("lowvram", True)
            and bootstrap.engine_lowvram(stats) is False)
        payload["engine_managed"] = comfy_proc.alive()
    payload["ready"] = bool(online and payload["nodes_ready"] and not missing)
    return jsonify(payload)


@app.post("/api/setup/start")
def api_setup_start():
    with setup_lock:
        if progress.running:
            return jsonify({"error": "Setup is already running."}), 409
        progress.__init__()
        progress.running = True       # claimed here, so a double click is a 409
    b = request.get_json(silent=True) or {}
    for key in ("comfy_url", "models_dir", "precision", "turbo",
                "lowvram", "want_kjnodes", "want_rtx", "want_manager"):
        if key in b:
            cfg[key] = b[key]
    cfg["comfy_url"] = normal_url(cfg["comfy_url"])
    client.url = cfg["comfy_url"]
    save_config(cfg)
    threading.Thread(target=bootstrap.run_setup,
                     args=(cfg, progress, comfy_proc, b.get("comfy_dir", ""),
                           b.get("mode", "auto")), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/setup/state")
def api_setup_state():
    snap = progress.snapshot(int(request.args.get("since", 0)))
    snap["comfy_tail"] = comfy_proc.tail(12)
    return jsonify(snap)


def _note(msg: str) -> None:
    """Engine actions belong in the engine console, next to its own output."""
    comfy_proc.note(msg)
    progress.log(msg)


def take_over_port(url: str, port: int):
    """Close whatever ComfyUI answers on the port.

    Returns ("manager-reboot", None) when ComfyUI-Manager rebooted it in
    place, ("freed", None) when the port is now empty, or (None, advice)
    when it cannot be done — with advice that names the actual obstacle,
    because "close it yourself" against a windowless process is a treasure
    hunt through Task Manager.
    """
    _note("This ComfyUI was not started here — taking it over.")
    try:
        r = requests.post(f"{url}/manager/reboot", json={}, timeout=5)
        accepted = r.status_code in (200, 201, 204)
    except requests.exceptions.RequestException:
        accepted = True          # the connection dropping is the reboot
    if accepted:
        deadline = time.time() + 10
        while time.time() < deadline:
            if not comfy_online(url):
                _note("ComfyUI-Manager took the reboot; waiting for the "
                      "engine to come back.")
                return "manager-reboot", None
            time.sleep(0.5)
        _note("ComfyUI-Manager did not take the reboot; stopping the "
              "process instead.")

    def settled_free() -> bool:
        # a supervisor (ComfyUI Desktop, a launcher .bat) respawns in under
        # a second — quiet is only free once it stays quiet
        time.sleep(2.0)
        return not comfy_online(url) and not bootstrap.port_pids(port)

    first_pids: list[int] = []
    denied = False
    for attempt in range(3):
        pids = bootstrap.port_pids(port)
        if attempt == 0:
            first_pids = pids
        if not pids:
            if not comfy_online(url) and settled_free():
                return "freed", None
            if not comfy_online(url):
                _note("It came straight back — something restarted it.")
                continue
            return None, (f"Something answers on port {port} but its process "
                          "could not be found — it may belong to another "
                          "user account. Close it in Task Manager, then "
                          "press Start ComfyUI.")
        for pid in pids:
            cmd = bootstrap.pid_cmdline(pid)
            _note(f"Port {port} is held by pid {pid}"
                  + (f": {cmd[:120]}" if cmd else " (command line unreadable)"))
            if not cmd:
                # never kill what cannot be identified
                return None, (f"Port {port} is held by pid {pid}, whose "
                              "command line could not be read, so it was "
                              "left alone. Close it yourself, or point "
                              "Settings at a different address.")
            if not any(k in cmd.lower()
                       for k in ("python", "main.py", "comfy")):
                return None, (f"Port {port} is held by something that does "
                              f"not look like ComfyUI ({cmd[:90]}). Close it "
                              "yourself, or point Settings at a different "
                              "address.")
        for pid in pids:
            said = bootstrap.kill_pid(pid)
            _note(f"Stopping pid {pid} — {said or 'no reply'}")
            if "denied" in (said or "").lower() \
                    or "access" in (said or "").lower():
                denied = True
        deadline = time.time() + 8
        while comfy_online(url) and time.time() < deadline:
            time.sleep(0.5)
        if not comfy_online(url):
            if settled_free():
                return "freed", None
            _note("It came straight back — something restarted it.")
            continue
        _note("Still answering — trying again.")

    now = bootstrap.port_pids(port)
    if denied:
        return None, ("Windows refused to stop it (access denied) — it was "
                      "started as administrator. Run MiniMax Studio as "
                      "administrator once, or close it in Task Manager, "
                      "then press Start ComfyUI.")
    if now and set(now) != set(first_pids):
        return None, ("It keeps coming back under a new process id — "
                      "something is supervising it (ComfyUI Desktop, or a "
                      "launcher script). Close that application, then press "
                      "Start ComfyUI.")
    return None, ("It would not close. The Engine console shows what was "
                  "tried; close it in Task Manager, then press Start "
                  "ComfyUI.")


def _refresh_schema_when_up() -> None:
    """After a (re)start, drop the cached schema the moment the engine
    answers — otherwise the fresh model scan hides behind the old cache
    for up to two minutes."""
    def wait():
        if bootstrap.wait_for_comfy(cfg["comfy_url"], timeout=900):
            try:
                client.schema(force=True)
            except Exception:
                pass
    threading.Thread(target=wait, daemon=True).start()


@app.post("/api/comfy/start")
def api_comfy_start():
    if comfy_online(cfg["comfy_url"]):
        return jsonify({"ok": True, "already": True})
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return jsonify({"error": "Run setup first."}), 400
    comfy_proc.start(py, Path(cfg["comfy_dir"]),
                     comfy_port(cfg["comfy_url"]), progress,
                     cfg.get("lowvram", True))
    _refresh_schema_when_up()
    return jsonify({"ok": True})


@app.post("/api/comfy/restart")
def api_comfy_restart():
    """Stop and start ComfyUI, so it rescans its model folders and loads
    newly installed nodes — the two things only a restart does.

    An engine this app did not start (an orphan from an earlier run, or one
    launched by hand) is taken over rather than declared unreachable: first
    ComfyUI-Manager's own reboot, and failing that the process holding the
    configured port is verified to look like ComfyUI and stopped, then a
    managed one starts in its place. The old advice — "close it yourself" —
    asked people to hunt a windowless python in Task Manager.
    """
    url = cfg["comfy_url"]
    port = comfy_port(url)
    py = bootstrap.comfy_python(cfg)
    can_start = bool(cfg.get("comfy_dir") and py)

    if comfy_proc.alive():
        if not can_start:
            return jsonify({"error": "Run setup first."}), 400
        _note("Restarting the managed engine…")
        comfy_proc.stop()
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress,
                         cfg.get("lowvram", True))
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "managed"})

    if not comfy_online(url):
        if not can_start:
            return jsonify({"error": "Run setup first."}), 400
        _note("Starting ComfyUI…")
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress,
                         cfg.get("lowvram", True))
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "started"})

    # online, but not ours — take it over
    how, advice = take_over_port(url, port)
    if advice:
        return jsonify({"error": advice}), 409
    if how == "manager-reboot":
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "manager-reboot"})
    if not can_start:
        return jsonify({"ok": True, "how": "stopped",
                        "note": "Stopped it. This app has no ComfyUI of its "
                                "own to start — run setup, or start yours "
                                "again yourself."})
    _note("Starting a managed engine in its place…")
    comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress,
                     cfg.get("lowvram", True))
    _refresh_schema_when_up()
    return jsonify({"ok": True, "how": "takeover"})


@app.get("/api/comfy/log")
def api_comfy_log():
    """The engine's own console — the visible cue that it is starting,
    started, or telling you exactly what failed to import."""
    n = min(max(int(request.args.get("n", 80)), 1), 400)
    return jsonify({"lines": comfy_proc.tail(n),
                    "running": comfy_proc.alive(),
                    "online": comfy_online(cfg["comfy_url"])})


@app.post("/api/config")
def api_config():
    b = request.get_json(silent=True) or {}
    for key in ("comfy_url", "comfy_dir", "models_dir", "auto_start_comfy",
                "torch_index", "precision", "turbo", "lowvram",
                "want_kjnodes", "want_rtx", "want_manager"):
        if key in b:
            cfg[key] = b[key]
    cfg["comfy_url"] = normal_url(cfg["comfy_url"])
    client.url = cfg["comfy_url"]
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# dependencies / tasks
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    live = client if comfy_online(cfg["comfy_url"]) else None
    return jsonify({"items": manager.dependencies(cfg, live),
                    "torch_index": cfg.get("torch_index", "")})


@app.post("/api/deps/<path:dep_id>/install")
def api_dep_install(dep_id: str):
    b = request.get_json(silent=True) or {}
    if b.get("torch_index") is not None:
        cfg["torch_index"] = b["torch_index"]
        save_config(cfg)
    try:
        return jsonify({"ok": True,
                        "task": manager.install_dependency(dep_id, cfg, b).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    task_id = request.args.get("id", "")
    since = int(request.args.get("since", 0))
    if task_id:
        task = manager.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"]) for t in manager.TASKS.list()[:25]])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = manager.TASKS.get(task_id)
    if task:
        task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
@app.get("/api/hf/settings")
def api_hf_settings():
    token = cfg.get("hf_token") or ""
    return jsonify({"endpoint": cfg.get("hf_endpoint") or manager.DEFAULT_ENDPOINT,
                    "token_set": bool(token),
                    "token_hint": ("…" + token[-4:]) if len(token) > 4 else "",
                    "repo": cfg.get("hf_repo") or bootstrap.MODEL_REPO,
                    "curated": manager.curated(cfg),
                    "folders": manager.MODEL_FOLDERS,
                    "models_dir": cfg.get("models_dir", "")})


@app.post("/api/hf/settings")
def api_hf_settings_save():
    b = request.get_json(silent=True) or {}
    if "token" in b:
        cfg["hf_token"] = (b["token"] or "").strip()
    if b.get("endpoint") is not None:
        cfg["hf_endpoint"] = b["endpoint"].strip() or manager.DEFAULT_ENDPOINT
    if b.get("repo"):
        cfg["hf_repo"] = b["repo"].strip()
    if b.get("models_dir"):
        cfg["models_dir"] = b["models_dir"].strip()
    if b.get("precision"):
        cfg["precision"] = b["precision"]
    save_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/hf/browse")
def api_hf_browse():
    repo = (request.args.get("repo") or cfg.get("hf_repo") or "").strip()
    try:
        data = manager.hf_browse(cfg, repo)
        cfg["hf_repo"] = repo
        save_config(cfg)
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/hf/download")
def api_hf_download():
    b = request.get_json(silent=True) or {}
    try:
        if b.get("set"):
            if b.get("precision"):
                cfg["precision"] = b["precision"]
                save_config(cfg)
            tasks = manager.download_set(cfg)
            if not tasks:
                return jsonify({"ok": True, "tasks": [],
                                "note": "Everything in that set is already here."})
            return jsonify({"ok": True, "tasks": [t.view() for t in tasks]})
        path = (b.get("path") or "").strip()
        if not path:
            return jsonify({"error": "Pick a file to download."}), 400
        task = manager.hf_download(cfg, b.get("repo") or cfg.get("hf_repo")
                                   or bootstrap.MODEL_REPO, path,
                                   b.get("folder") or "")
        return jsonify({"ok": True, "task": task.view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/hf/local")
def api_hf_local():
    return jsonify({"models": manager.local_models(cfg),
                    "models_dir": cfg.get("models_dir", "")})


@app.delete("/api/hf/local")
def api_hf_delete():
    b = request.get_json(silent=True) or {}
    try:
        manager.delete_model(cfg, b.get("folder", ""), b.get("name", ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
def api_generate():
    params = request.get_json(silent=True) or {}
    if not isinstance(params, dict):
        return jsonify({"error": "Send generation settings as an object."}), 400
    prompt, refs = params.get("prompt", ""), params.get("refs", [])
    ref_video = params.get("ref_video") or ""
    if (not isinstance(prompt, str) or not isinstance(refs, list)
            or not isinstance(ref_video, str)):
        return jsonify({"error": "Prompt must be text and references must be "
                                 "a list."}), 400
    if not prompt.strip() and not refs and not ref_video:
        return jsonify({"error": "Describe the shot, or add an image or video "
                                 "reference."}), 400
    try:
        runs = int(params.get("runs") or 1)
        seconds = float(params.get("seconds") or 6)
        megapixels = float(params.get("megapixels") or 0.2)
        fps = int(params.get("fps") or 24)
    except (TypeError, ValueError):
        return jsonify({"error": "Runs, length, FPS and megapixels must be "
                                 "numbers."}), 400
    if not 1 <= runs <= 4:
        return jsonify({"error": "Runs must be between 1 and 4."}), 400
    if not math.isfinite(seconds) or not 0.5 <= seconds <= 30:
        return jsonify({"error": "Length must be between 0.5 and 30 seconds."}), 400
    if not math.isfinite(megapixels) or not 0.05 <= megapixels <= 2:
        return jsonify({"error": "Megapixels must be between 0.05 and 2."}), 400
    if not 1 <= fps <= 120:
        return jsonify({"error": "FPS must be between 1 and 120."}), 400
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running. Start it from the "
                                 "Engine page."}), 503
    created = []
    for run in range(runs):
        job_id = uuid.uuid4().hex[:12]
        job_params = dict(params)
        # a fixed seed gives each run its own neighbour, not the same clip 4x
        if str(params.get("seed", "")).strip().lstrip("-").isdigit():
            job_params["seed"] = int(params["seed"]) + run
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
                            "stage": "Starting", "created": time.time(),
                            "title": params.get("title") or title_from(params)}
        threading.Thread(target=run_job, args=(job_id, job_params),
                         daemon=True).start()
        created.append(job_id)
        time.sleep(0.2)
    return jsonify({"jobs": created})


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        active = [j for j in jobs.values()
                  if j["status"] == "running"
                  or time.time() - j.get("finished", j["created"]) < 180]
        return jsonify(sorted(active, key=lambda j: j["created"], reverse=True))


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "No such job."}), 404
        if job["status"] != "running":
            return jsonify({"error": "That job has already finished."}), 409
        job["cancelled"] = True
    # run_job sees the flag within a second and stops this prompt alone
    return jsonify({"ok": True})


@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    try:
        return jsonify({"ok": True,
                        "name": client.upload(request.files["file"])})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/upscale/<clip_id>")
def api_upscale(clip_id: str):
    """Push a finished clip back through RTX Video Super Resolution."""
    b = request.get_json(silent=True) or {}
    clip = next((c for c in read_gallery() if c["id"] == clip_id), None)
    if not clip:
        return jsonify({"error": "Clip not found."}), 404
    path = CLIPS_DIR / clip["file"]
    if not path.exists():
        return jsonify({"error": "That file is missing."}), 404
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running."}), 503
    try:
        scale = int(b.get("scale", 2))
    except (TypeError, ValueError):
        return jsonify({"error": "Scale must be a whole number."}), 400
    if scale not in (2, 4):
        return jsonify({"error": "Scale must be 2 or 4."}), 400
    try:
        with open(path, "rb") as fh:
            class Up:
                filename = clip["file"]
                stream = fh
                mimetype = "video/mp4"
            name = client.upload(Up())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not hand the clip to ComfyUI: {exc}"}), 500
    job_id = uuid.uuid4().hex[:12]
    params = {"kind": "rtx", "video": name, "scale": scale,
              "quality": b.get("quality", "ULTRA"),
              "title": clip["title"] + f" ×{scale}",
              "prompt": clip.get("prompt", ""),
              # the RTX graph carries no size of its own, so the source clip's
              # shape times the multiplier is what the gallery should show
              "width": (clip.get("width") or 0) * scale or None,
              "height": (clip.get("height") or 0) * scale or None,
              "seconds": clip.get("seconds"), "length": clip.get("length"),
              "fps": clip.get("fps"), "aspect": clip.get("aspect"),
              "megapixels": clip.get("megapixels"),
              "steps": clip.get("steps"), "shift": clip.get("shift")}
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
                        "stage": "Starting", "created": time.time(),
                        "title": params["title"]}
    threading.Thread(target=run_job, args=(job_id, params), daemon=True).start()
    return jsonify({"jobs": [job_id]})


@app.get("/api/preflight")
def api_preflight():
    return jsonify(bootstrap.preflight(cfg))


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
@app.get("/api/clips")
def api_clips():
    return jsonify(read_gallery())


@app.get("/api/clip/<image_id>")
def api_clip(image_id: str):
    for item in read_gallery():
        if item["id"] == image_id:
            path = CLIPS_DIR / item["file"]
            if not path.exists():
                return jsonify({"error": "That file is missing."}), 404
            mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
            return send_file(path, mimetype=mime, conditional=True,
                             download_name=f"{item['title']}{path.suffix}")
    return jsonify({"error": "Clip not found."}), 404


@app.delete("/api/clip/<image_id>")
def api_clip_delete(image_id: str):
    # Serialize deletion with render completion so a stale snapshot cannot
    # erase a clip that was added while the file was being removed.
    with gallery_lock:
        items = _read_gallery_unlocked()
        for item in items:
            if item.get("id") == image_id:
                try:
                    (CLIPS_DIR / item["file"]).unlink(missing_ok=True)
                except (KeyError, OSError):
                    pass
        _write_gallery_unlocked([i for i in items if i.get("id") != image_id])
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# board — the storyboard: shot cards, chained into a cut
# --------------------------------------------------------------------------- #
def _clean_shot(shot) -> dict | None:
    if not isinstance(shot, dict):
        return None
    try:
        seconds = float(shot.get("seconds") or 6)
    except (TypeError, ValueError):
        seconds = 6.0
    if not math.isfinite(seconds):
        seconds = 6.0
    return {"id": str(shot.get("id") or uuid.uuid4().hex[:12])[:32],
            "prompt": str(shot.get("prompt") or "")[:4000],
            "seconds": max(0.5, min(seconds, 30)),
            "chain": bool(shot.get("chain", True)),
            "clip": str(shot.get("clip") or "")[:32]}


@app.get("/api/board")
def api_board():
    with board_lock:
        if not BOARD_PATH.exists():
            return jsonify([])
        try:
            raw = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
        except Exception:
            return jsonify([])
    return jsonify([s for s in (_clean_shot(x) for x in raw) if s])


@app.post("/api/board")
def api_board_save():
    body = request.get_json(silent=True)
    shots = body if isinstance(body, list) else \
        (body or {}).get("shots") if isinstance(body, dict) else None
    if not isinstance(shots, list):
        return jsonify({"error": "Send the board as a list of shots."}), 400
    cleaned = [s for s in (_clean_shot(x) for x in shots[:200]) if s]
    with board_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BOARD_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "shots": len(cleaned)})


def ensure_engine_at_boot() -> None:
    """A launch ends with a working engine, without a button pressed.

    Offline: start the managed one. Online and healthy: adopt it. Online but
    useless — a stale scan hiding the weights, installed nodes it never
    loaded, or a different install squatting the port — replace it, with the
    same looks-like-ComfyUI guard the Restart button uses. An external-mode
    setup (managed False) is never touched: that engine is the person's own.
    """
    if not (cfg.get("setup_complete") and cfg.get("auto_start_comfy", True)):
        return
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return
    url = cfg["comfy_url"]
    port = comfy_port(url)

    if not comfy_online(url):
        _note("Starting ComfyUI…")
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress,
                         cfg.get("lowvram", True))
        _refresh_schema_when_up()
        return

    # something already answers — decide between adopting and replacing
    reasons = []
    try:
        client.schema(force=True)
        has_nodes = client.has("MiniMaxH3ReferenceToVideo")
        unets = client.unets()
    except Exception as exc:  # noqa: BLE001
        _note(f"The engine already running would not describe itself "
              f"({exc}) — leaving it alone.")
        return
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    weights_here = bool(models_dir and models_dir.is_dir() and
                        not bootstrap.missing_models(models_dir, cfg))
    if weights_here and not any("minimax" in u.lower() for u in unets):
        reasons.append("it started before the weights landed")
    if not has_nodes and weights_here:
        reasons.append("the MiniMax H3 nodes are not loaded")
    for node in bootstrap.CUSTOM_NODES:
        marker = {"rtx": "RTXVideoSuperResolution",
                  "kjnodes": "ModelPreviewOverrideKJ"}.get(node["id"])
        if marker and bootstrap.node_installed(Path(cfg["comfy_dir"]), node) \
                and not client.has(marker):
            reasons.append(f"{node['label']} is installed but not loaded")
    stats = bootstrap.comfy_stats(url) or {}
    argv = (stats.get("argv") or [""])[0]
    want = str(Path(cfg["comfy_dir"])).replace("\\", "/").lower()
    if argv and want and want not in argv.replace("\\", "/").lower():
        reasons.append("a different install is answering the address")
    if cfg.get("lowvram", True) and bootstrap.engine_lowvram(stats) is False:
        reasons.append("it was started without low-VRAM mode (--lowvram)")

    if not reasons:
        _note(f"Adopting the ComfyUI already running at {url}.")
        return
    if not cfg.get("managed", True):
        _note("The engine already running has problems ("
              + "; ".join(reasons) + ") but it is yours, not this app's — "
              "restart it yourself, or press Restart ComfyUI.")
        return
    _note("The engine already running is no use as it stands — "
          + "; ".join(reasons) + ". Replacing it.")
    how, advice = take_over_port(url, port)
    if advice:
        _note(advice)
        return
    if how == "manager-reboot":
        _refresh_schema_when_up()
        return
    _note("Starting a managed engine in its place…")
    comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress,
                     cfg.get("lowvram", True))
    _refresh_schema_when_up()


# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=ws_listener, daemon=True).start()
    # the engine comes up on its own; the page can open meanwhile
    threading.Thread(target=ensure_engine_at_boot, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  MiniMax Studio  →  {url}\n")
    if os.environ.get("MINIMAX_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        comfy_proc.stop()


if __name__ == "__main__":
    main()
