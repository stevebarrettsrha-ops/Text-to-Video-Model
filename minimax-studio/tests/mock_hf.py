"""A stand-in for huggingface.co: the tree API and resolve downloads, with
Range support, so the download/resume logic and the setup progress bar run
for real — as in the sibling apps' suites.

/mock/mode lets a test slow transfers down enough to watch the bar move.
"""
import os
import random

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
app.json.sort_keys = False

REPOS = {
    "Comfy-Org/MiniMax-H3": [
        ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
         2_500_000),
        ("diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
         2_000_000),
        ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
         3_000_000),
        ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
         2_800_000),
        ("vae/minimax_h3_video_vae_fp16.safetensors", 900_000),
        ("vae/minimax_h3_audio_vae_fp32.safetensors", 400_000),
        ("loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
         300_000),
        ("loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
         300_000),
        ("vae_approx/taeh3.safetensors", 120_000),
        ("README.md", 5_000),
        (".gitattributes", 1_500),
    ],
    "lightx2v/Minimax-h3-Turbo": [
        ("minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
         320_000),
        ("README.md", 3_000),
    ],
}

MODE = {"slow": 0.0, "gated": ""}
LOG = []


@app.post("/mock/mode")
def set_mode():
    MODE.update(request.get_json(silent=True) or {})
    return jsonify(MODE)


@app.get("/mock/log")
def get_log():
    return jsonify(LOG)


@app.get("/api/models/<path:repo>/tree/<rev>")
def tree(repo, rev):
    LOG.append(f"tree {repo}")
    if repo == MODE.get("gated") and not request.headers.get("Authorization"):
        return jsonify({"error": "gated"}), 401
    if repo not in REPOS:
        return jsonify({"error": "Repo not found"}), 404
    return jsonify([{"type": "file", "path": p, "size": s, "oid": "0" * 40}
                    for p, s in REPOS[repo]])


@app.get("/api/datasets/<path:repo>/tree/<rev>")
def tree_ds(repo, rev):
    return jsonify({"error": "Repo not found"}), 404


def body_for(path, size):
    return random.Random(sum(path.encode()) or 1).randbytes(size)


@app.get("/<path:repo>/resolve/<rev>/<path:fname>")
def resolve(repo, rev, fname):
    if repo not in REPOS:
        return "no repo", 404
    entry = next((e for e in REPOS[repo] if e[0] == fname), None)
    if not entry:
        return "no file", 404
    data = body_for(fname, entry[1])
    rng = request.headers.get("Range", "")
    start = 0
    status = 200
    headers = {"Accept-Ranges": "bytes",
               "Content-Type": "application/octet-stream"}
    if rng.startswith("bytes="):
        start = int(rng.split("=", 1)[1].split("-")[0])
        if start >= len(data):
            LOG.append(f"416 {fname}")
            return Response(status=416)
        status = 206
        headers["Content-Range"] = f"bytes {start}-{len(data)-1}/{len(data)}"
    chunk = data[start:]
    if MODE.get("slow"):
        import time as _t
        _t.sleep(float(MODE["slow"]))
    headers["Content-Length"] = str(len(chunk))
    LOG.append(f"get {fname} start={start} status={status} sent={len(chunk)}")
    return Response(chunk, status=status, headers=headers)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
    app.run(host="127.0.0.1", port=port, threaded=True)
