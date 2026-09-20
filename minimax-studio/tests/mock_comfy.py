"""A stand-in ComfyUI: /object_info shaped like the real one, plus a queue.

The schema in object_info.json is derived from the two reference workflows in
assets/ — the node classes, input names and types are the ones the real graphs
carry (LTXVSeparateAVLatent really does call its input `av_latent`). A stand-in
that disagrees with the thing it stands in for is worth very little, so
/prompt validates the way ComfyUI does: unknown nodes, unknown inputs, values
outside a combo's options, missing required inputs and dangling links are all
rejected.

Knobs, all environment variables:
  MOCK_DELAY              seconds a clip takes to "render" (default 1)
  MOCK_FAIL_AFTER         if set, every render ends in a CUDA out-of-memory
  MOCK_OMIT               comma-separated node classes to pretend not to have
  MOCK_NO_UPSCALER_MODEL  serve an empty model list on the latent upscaler
"""
import base64
import hashlib
import json
import os
import pathlib
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OBJECT_INFO = json.loads(
    (pathlib.Path(__file__).with_name("object_info.json")).read_text())
for cls in [c for c in os.environ.get("MOCK_OMIT", "").split(",") if c]:
    OBJECT_INFO.pop(cls, None)
if os.environ.get("MOCK_NO_UPSCALER_MODEL") and \
        "MinimaxH3LatentUpscaler3D" in OBJECT_INFO:
    OBJECT_INFO["MinimaxH3LatentUpscaler3D"]["input"]["required"][
        "model_name"][0] = []

DELAY = float(os.environ.get("MOCK_DELAY", "1"))

HISTORY: dict = {}
QUEUE_PENDING: list = []
QUEUE_RUNNING: list = []
INTERRUPTS: list = []
UPLOADS: list = []          # filenames handed to /upload/image
PROMPTS: dict = {}          # prompt_id -> the graph as queued
LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
WS_CLIENTS: list = []

# Enough of an MP4 header that mimetypes and players recognise the bytes.
FAKE_MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 256

# The browser tests need a clip a browser can actually decode. There is no
# encoder here, so a test records one itself (canvas.captureStream +
# MediaRecorder) and POSTs it to /testvideo; from then on every render is
# served as that webm. Without it, renders are FAKE_MP4 — header-only bytes,
# fine for the API tests, undecodable by design.
TEST_VIDEO: list = []       # [bytes] once a test has posted one


def _object_info():
    """The schema, with everything uploaded so far visible to LoadImage and
    LoadVideo — the real server rescans its input folder the same way."""
    out = json.loads(json.dumps(OBJECT_INFO))
    with LOCK:
        names = list(UPLOADS)
    for cls, key in (("LoadImage", "image"), ("LoadVideo", "file")):
        if cls in out:
            out[cls]["input"]["required"][key][0] += names
    return out


def ws_send(obj):
    """One unmasked text frame to every connected /ws client."""
    data = json.dumps(obj).encode()
    head = bytearray([0x81])
    if len(data) < 126:
        head.append(len(data))
    else:
        head += bytes([126]) + struct.pack(">H", len(data))
    frame = bytes(head) + data
    with LOCK:
        for sock in WS_CLIENTS[:]:
            try:
                sock.sendall(frame)
            except OSError:
                WS_CLIENTS.remove(sock)


def execute(pid, graph):
    with RUN_LOCK:                      # one prompt at a time, like the real queue
        with LOCK:
            if pid not in QUEUE_PENDING:
                return
            QUEUE_PENDING.remove(pid)
            QUEUE_RUNNING.append(pid)
        _execute(pid, graph)


def _execute(pid, graph):
    ws_send({"type": "execution_start", "data": {"prompt_id": pid}})
    steps = max(1, int(DELAY * 2))
    for i in range(steps):
        time.sleep(DELAY / steps)
        ws_send({"type": "progress",
                 "data": {"value": i + 1, "max": steps, "prompt_id": pid}})
        with LOCK:
            if pid in INTERRUPTS:
                break
    with LOCK:
        interrupted = pid in INTERRUPTS
    if os.environ.get("MOCK_FAIL_AFTER"):
        status = {"status_str": "error", "messages": [
            ["execution_error", {"node_type": "KSampler",
                                 "exception_message": "CUDA out of memory"}]]}
        outputs = {}
    elif interrupted:
        status = {"status_str": "error", "messages": [
            ["execution_interrupted", {"node_type": "KSampler",
                                       "exception_message": "interrupted"}]]}
        outputs = {}
    else:
        status = {"status_str": "success", "messages": []}
        outputs = {}
        with LOCK:
            ext = ".webm" if TEST_VIDEO else ".mp4"
        for nid, node in graph.items():
            if node["class_type"] == "SaveVideo":
                prefix = node["inputs"].get("filename_prefix", "video/ComfyUI")
                outputs[nid] = {"images": [{
                    "filename": f"{prefix.split('/')[-1]}_{pid}{ext}",
                    "subfolder": "video", "type": "output"}]}
    with LOCK:
        QUEUE_RUNNING.remove(pid)
        HISTORY[pid] = {"status": status, "outputs": outputs}
    ws_send({"type": "execution_success" if status["status_str"] == "success"
             else "execution_error", "data": {"prompt_id": pid}})


def validate(graph):
    """Reject anything ComfyUI itself would reject."""
    info_all = _object_info()
    errs = {}
    for nid, node in graph.items():
        cls = node["class_type"]
        info = info_all.get(cls)
        if not info:
            errs[nid] = {"class_type": cls, "errors": [
                {"message": "Node not found", "details": cls}]}
            continue
        spec = dict(info["input"].get("required", {}))
        spec.update(info["input"].get("optional", {}))
        node_errs = []
        for name, value in node["inputs"].items():
            if name == "control_after_generate":
                continue
            if name not in spec:
                node_errs.append({"message": "Unknown input", "details": name})
                continue
            kind = spec[name][0]
            if isinstance(value, list) and len(value) == 2 \
                    and isinstance(value[1], int):
                if str(value[0]) not in graph:
                    node_errs.append({"message": "Link to a node that is not "
                                      "in the prompt", "details": f"{name}={value}"})
                continue
            if isinstance(kind, list):
                if value not in kind:
                    node_errs.append({"message": "Value not in list",
                                      "details": f"{name}: {value!r}"})
            elif kind == "INT" and not isinstance(value, int):
                node_errs.append({"message": "Wrong type", "details": f"{name} INT"})
            elif kind == "FLOAT" and not isinstance(value, (int, float)):
                node_errs.append({"message": "Wrong type", "details": f"{name} FLOAT"})
            elif kind == "STRING" and not isinstance(value, str):
                node_errs.append({"message": "Wrong type", "details": f"{name} STRING"})
            elif kind == "BOOLEAN" and not isinstance(value, bool):
                node_errs.append({"message": "Wrong type", "details": f"{name} BOOLEAN"})
        for name in (info["input"].get("required") or {}):
            if name != "control_after_generate" and name not in node["inputs"]:
                node_errs.append({"message": "Required input is missing",
                                  "details": name})
        if node_errs:
            errs[nid] = {"class_type": cls, "errors": node_errs}
    return errs


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/ws":
            key = self.headers.get("Sec-WebSocket-Key", "")
            accept = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()).decode()
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            with LOCK:
                WS_CLIENTS.append(self.connection)
            try:
                while self.connection.recv(1024):
                    pass
            except OSError:
                pass
            self.close_connection = True
            return
        if p == "/object_info":
            self._send(200, _object_info())
        elif p == "/system_stats":
            self._send(200, {"system": {"comfyui_version": "0.3.75"}})
        elif p.startswith("/history/"):
            pid = p.rsplit("/", 1)[-1]
            with LOCK:
                self._send(200, {pid: HISTORY[pid]} if pid in HISTORY else {})
        elif p == "/view":
            with LOCK:
                real = TEST_VIDEO[0] if TEST_VIDEO else None
            if real:
                self._send(200, real, "video/webm")
            else:
                self._send(200, FAKE_MP4, "video/mp4")
        elif p == "/prompts":            # test-only: what was queued
            with LOCK:
                self._send(200, PROMPTS)
        else:
            self._send(404, {"error": "no route " + p})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if p == "/prompt":
            graph = json.loads(raw)["prompt"]
            bad = validate(graph)
            if bad:
                self._send(400, {"error": {
                    "type": "prompt_outputs_failed_validation",
                    "message": "Prompt outputs failed validation", "details": ""},
                    "node_errors": bad})
                return
            with LOCK:
                pid = f"pid{len(HISTORY) + len(QUEUE_PENDING) + len(QUEUE_RUNNING) + 1}"
                PROMPTS[pid] = graph
                QUEUE_PENDING.append(pid)
            threading.Thread(target=execute, args=(pid, graph),
                             daemon=True).start()
            self._send(200, {"prompt_id": pid, "number": 1, "node_errors": {}})
        elif p == "/interrupt":
            with LOCK:
                INTERRUPTS.extend(QUEUE_RUNNING)
            self._send(200, {})
        elif p == "/testvideo":
            with LOCK:
                TEST_VIDEO[:] = [raw]
            self._send(200, {"ok": True, "bytes": len(raw)})
        elif p == "/upload/image":
            # good enough multipart parsing for a stand-in: the filename field
            marker = b'filename="'
            i = raw.find(marker)
            name = raw[i + len(marker):raw.index(b'"', i + len(marker))].decode() \
                if i >= 0 else "upload.bin"
            with LOCK:
                if name not in UPLOADS:
                    UPLOADS.append(name)
            self._send(200, {"name": name, "subfolder": "", "type": "input"})
        else:
            self._send(404, {"error": "no route " + p})


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
