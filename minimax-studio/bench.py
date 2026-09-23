#!/usr/bin/env python3
"""Time MiniMax H3 on this machine — base sizes, the H3 upscale, RTX after.

    python bench.py                         the default matrix, ~6 renders
    python bench.py --bases 0.2,0.3 --upscale on --rtx
    python bench.py --ref face.png --prompt "..."    with a reference image

Runs against the ComfyUI MiniMax Studio is set up for (or --url), through
the same graph builder the app uses, so the numbers are the app's numbers.
For every render it records, from ComfyUI's own websocket events:

  - wall time from queue to saved clip, and the wait before it started
  - time per stage: text encode + references, base sampling, H3 latent
    upscale, refine sampling, video decode, audio decode, save
  - seconds per sampling step, base and refine
  - peak VRAM in use and the lowest free system RAM, polled from
    /system_stats every half second (these include anything else on the GPU)

Every render in one matrix shares a seed, so the clips differ by size and
upscale only. The report lands in data/bench/ as Markdown and CSV, and the
clips beside it, so before and after can be compared by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import requests

import bootstrap
from comfy import ComfyClient, ComfyError, dimensions, frame_length

GIB = 1024 ** 3

# node class -> the stage it is reported under
STAGES = {
    "MiniMaxH3ReferenceToVideo": "encode",
    "LoadImage": "encode", "LoadVideo": "encode", "GetVideoComponents": "encode",
    "LoadAudio": "encode", "CLIPLoader": "encode",
    "UNETLoader": "load", "VAELoader": "load", "LoraLoaderModelOnly": "load",
    "MiniMaxH3SigmaShift": "load", "ModelAttentionBackend": "load",
    "LTXVSeparateAVLatent": "upscale", "MinimaxH3LatentUpscaler3D": "upscale",
    "LTXVConcatAVLatent": "upscale",
    "VAEDecode": "decode", "VAEDecodeTiled": "decode",
    "VAEDecodeAudio": "audio",
    "CreateVideo": "save", "SaveVideo": "save",
    "RTXVideoSuperResolution": "rtx",
}
STAGE_ORDER = ["load", "encode", "sample", "upscale", "refine", "decode",
               "audio", "save", "rtx"]


class Monitor:
    """Peak VRAM in use and lowest free RAM while a render runs."""

    def __init__(self, url: str) -> None:
        self.url, self.stop = url, threading.Event()
        self.vram_total = self.vram_peak = 0
        self.ram_total = 0
        self.ram_low: int | None = None
        self.gpu = ""
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        try:
            st = requests.get(f"{self.url}/system_stats", timeout=3).json()
        except Exception:
            return
        dev = (st.get("devices") or [{}])[0]
        self.gpu = dev.get("name", self.gpu)
        total, free = dev.get("vram_total") or 0, dev.get("vram_free")
        if total and free is not None:
            self.vram_total = total
            self.vram_peak = max(self.vram_peak, total - free)
        system = st.get("system") or {}
        if system.get("ram_free") is not None:
            self.ram_total = system.get("ram_total") or self.ram_total
            free = system["ram_free"]
            self.ram_low = free if self.ram_low is None else min(self.ram_low, free)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def run_one(client: ComfyClient, graph: dict, timeout: float) -> dict:
    """Queue one prompt and time it node by node over the websocket."""
    import websocket  # websocket-client, already a MiniMax Studio dependency

    ws_url = client.url.replace("http", "ws", 1) + f"/ws?clientId={client.client_id}"
    ws = websocket.WebSocket()
    ws.connect(ws_url, timeout=15)
    ws.settimeout(2)
    classes = {nid: n["class_type"] for nid, n in graph.items()}
    samplers = [nid for nid, c in classes.items() if c == "KSampler"]
    # the refine pass is the sampler fed by the upscaler's re-joined latent
    refine = {nid for nid in samplers
              if isinstance(graph[nid]["inputs"].get("latent_image"), list)
              and classes.get(str(graph[nid]["inputs"]["latent_image"][0]))
              == "LTXVConcatAVLatent"}

    queued = time.time()
    prompt_id = client.queue(graph)
    started = None
    current, since = None, None
    per_node: dict[str, float] = {}
    steps: dict[str, int] = {}
    error = ""
    deadline = queued + timeout

    def close_node(now: float) -> None:
        if current is not None:
            per_node[current] = per_node.get(current, 0.0) + now - since

    try:
        while time.time() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                if client.failed(prompt_id):
                    error = client.failed(prompt_id) or "failed"
                    break
                if client.outputs(prompt_id):
                    break
                continue
            if not isinstance(raw, str):
                continue                          # preview frames
            msg = json.loads(raw)
            data = msg.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            now, kind = time.time(), msg.get("type")
            if kind == "execution_start":
                started = now
            elif kind == "executing":
                close_node(now)
                current, since = data.get("node"), now
                if current is None:
                    break                          # the prompt is finished
                current = str(current)
            elif kind == "progress" and current:
                steps[current] = max(steps.get(current, 0),
                                     int(data.get("max") or 0))
            elif kind == "execution_error":
                close_node(now)
                error = (f"{data.get('node_type')}: "
                         f"{data.get('exception_message', '')}".strip())
                break
            elif kind == "execution_success":
                close_node(now)
                current = None
                break
        else:
            error = f"no result after {timeout:.0f} s"
        done = time.time()           # before the close handshake, not after
    finally:
        try:
            ws.close(timeout=1)
        except Exception:
            pass

    stages = {k: 0.0 for k in STAGE_ORDER}
    for nid, secs in per_node.items():
        cls = classes.get(nid, "")
        if cls == "KSampler":
            stages["refine" if nid in refine else "sample"] += secs
        else:
            stages[STAGES.get(cls, "save")] += secs

    def per_step(ids) -> float | None:
        secs = sum(per_node.get(n, 0) for n in ids)
        count = sum(steps.get(n, 0) for n in ids)
        return round(secs / count, 2) if count else None

    return {"prompt_id": prompt_id, "error": error,
            "wait": round((started or queued) - queued, 1),
            "total": round(done - queued, 1),
            "stages": {k: round(v, 1) for k, v in stages.items()},
            "s_per_step": per_step([n for n in samplers if n not in refine]),
            "refine_s_per_step": per_step(refine) if refine else None}


def save_output(client: ComfyClient, prompt_id: str, dest: Path) -> Path | None:
    outs = client.outputs(prompt_id)
    if not outs:
        return None
    item = outs[0]
    dest = dest.with_suffix(Path(item["filename"]).suffix or ".mp4")
    with client.view(item) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(1024 * 256):
                fh.write(chunk)
    return dest


class _Upload:
    def __init__(self, path: Path, fh, mimetype: str) -> None:
        self.filename, self.stream, self.mimetype = path.name, fh, mimetype


def upload(client: ComfyClient, path: Path, mimetype: str) -> str:
    with open(path, "rb") as fh:
        return client.upload(_Upload(path, fh, mimetype))


def fmt_s(v) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"


def main(argv: list[str] | None = None) -> int:
    cfg = bootstrap.load_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=cfg.get("comfy_url"),
                    help="ComfyUI address (default: the app's)")
    ap.add_argument("--bases", default="0.2,0.3,0.4",
                    help="base megapixels to try, comma-separated")
    ap.add_argument("--upscale", default="off,on",
                    help="H3 latent upscale pass: off, on, or off,on")
    ap.add_argument("--upscale-mp", type=float, default=0.6,
                    help="target of the H3 upscale (the workflow's 0.6)")
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rtx", action="store_true",
                    help="also time RTX Video Super Resolution x2 on each clip")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeats per setting (the first includes disk reads)")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--prompt", default=(
        "A medium shot of a lighthouse keeper climbing a spiral staircase at "
        "dusk, lantern swinging, wind howling against the windows."))
    ap.add_argument("--ref", action="append", default=[],
                    help="reference image (repeat for up to three)")
    ap.add_argument("--timeout", type=float, default=3 * 3600,
                    help="seconds before one render is given up on")
    ap.add_argument("--out", default=str(bootstrap.DATA_DIR / "bench"))
    a = ap.parse_args(argv)

    url = bootstrap.normal_url(a.url)
    if not bootstrap.comfy_online(url):
        print(f"ComfyUI is not answering at {url}. Start it from MiniMax "
              "Studio's Engine page (so it runs with --lowvram), then rerun.")
        return 2
    client = ComfyClient(url)
    try:
        client.ensure_supported()
    except ComfyError as exc:
        print(exc)
        return 2
    stats = bootstrap.comfy_stats(url) or {}
    lowvram = bootstrap.engine_lowvram(stats)
    if lowvram is False:
        print("WARNING: this ComfyUI was started without --lowvram. On an 8 GB "
              "card the numbers below are not what the app gets — restart it "
              "from MiniMax Studio first.\n")

    bases = [float(x) for x in a.bases.split(",") if x.strip()]
    ups = [u.strip() == "on" for u in a.upscale.split(",") if u.strip()]
    caps = client.capabilities()
    if any(ups) and not caps.get("upscaler"):
        print("The H3 latent upscaler is not installed — timing base size only.")
        ups = [False]
    if a.rtx and not caps.get("rtx"):
        print("The NVIDIA RTX nodes are not loaded — skipping the RTX pass.")
        a.rtx = False
    refs = [upload(client, Path(r), "image/png") for r in a.ref[:3]]

    out = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    length = frame_length(a.seconds)
    plan = [(mp, up, n) for mp in bases for up in ups for n in range(a.runs)]
    print(f"{len(plan)} render(s) of {length} frames ({a.seconds:g} s) at "
          f"{a.aspect}, {a.steps} steps, seed {a.seed} → {out}\n")

    rows = []
    for index, (mp, up, n) in enumerate(plan, 1):
        w, h = dimensions(mp, a.aspect)
        uw, uh = dimensions(a.upscale_mp, a.aspect) if up else (w, h)
        label = f"{mp:g} MP {w}×{h}" + (f" → H3 {a.upscale_mp:g} MP" if up else "")
        print(f"[{index}/{len(plan)}] {label}" + (f" (run {n + 1})" if a.runs > 1 else ""),
              flush=True)
        params = {"prompt": a.prompt, "refs": refs, "seconds": a.seconds,
                  "megapixels": mp, "aspect": a.aspect, "steps": a.steps,
                  "seed": a.seed, "upscale": up, "upscale_mp": a.upscale_mp,
                  "tiled_decode": True}
        row = {"base_mp": mp, "base_size": f"{w}x{h}", "h3_upscale": up,
               "out_size": f"{uw}x{uh}", "frames": length, "run": n + 1}
        try:
            built = client.build(params)
            if up and not built.get("upscaled"):
                row["note"] = built.get("note", "")
                row["out_size"] = f"{w}x{h}"
            with Monitor(url) as mon:
                res = run_one(client, built["prompt"], a.timeout)
                mon.sample()
        except (ComfyError, requests.RequestException) as exc:
            res, mon = {"error": str(exc), "stages": {}}, None
        row.update({k: res.get(k) for k in
                    ("total", "wait", "s_per_step", "refine_s_per_step", "error")})
        row.update({f"t_{k}": v for k, v in res.get("stages", {}).items()})
        if mon:
            row["gpu"] = mon.gpu
            row["vram_peak_gb"] = round(mon.vram_peak / GIB, 2) if mon.vram_peak else None
            row["vram_total_gb"] = round(mon.vram_total / GIB, 2) if mon.vram_total else None
            row["ram_low_free_gb"] = (round(mon.ram_low / GIB, 2)
                                      if mon.ram_low is not None else None)
        clip = None
        if not res.get("error"):
            tag = f"{mp:g}mp{'-h3up' if up else ''}-r{n + 1}"
            clip = save_output(client, res["prompt_id"], out / tag)
            row["file"] = clip.name if clip else ""
        if a.rtx and clip:
            try:
                name = upload(client, clip, "video/mp4")
                rbuilt = client.build_rtx_upscale(name, 2, "ULTRA")
                rres = run_one(client, rbuilt["prompt"], a.timeout)
                row["rtx_total"] = rres.get("total")
                row["rtx_size"] = f"{int(uw) * 2}x{int(uh) * 2}"
                if not rres.get("error"):
                    rclip = save_output(client, rres["prompt_id"],
                                        out / (clip.stem + "-rtx2x"))
                    row["rtx_file"] = rclip.name if rclip else ""
                else:
                    row["rtx_error"] = rres["error"]
            except (ComfyError, requests.RequestException) as exc:
                row["rtx_error"] = str(exc)
        rows.append(row)
        if row.get("error"):
            print(f"    FAILED — {row['error']}")
        else:
            print(f"    {fmt_s(row['total'])} total · sampling "
                  f"{fmt_s(row.get('t_sample'))} ({fmt_s(row.get('s_per_step'))}/step)"
                  + (f" · H3 upscale+refine {fmt_s((row.get('t_upscale') or 0) + (row.get('t_refine') or 0))}"
                     if up else "")
                  + (f" · peak VRAM {row['vram_peak_gb']} GB" if row.get("vram_peak_gb") else "")
                  + (f" · RTX ×2 {fmt_s(row.get('rtx_total'))}" if row.get("rtx_total") else ""),
                  flush=True)

    write_reports(out, rows, a, stats, lowvram)
    print(f"\nReport: {out / 'report.md'}")
    return 0 if all(not r.get("error") for r in rows) else 1


def write_reports(out: Path, rows: list[dict], a, stats: dict, lowvram) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    gpu = next((r.get("gpu") for r in rows if r.get("gpu")), "") or "unknown GPU"
    vram = next((r.get("vram_total_gb") for r in rows if r.get("vram_total_gb")), None)
    lines = [
        "# MiniMax H3 benchmark",
        "",
        f"- {time.strftime('%Y-%m-%d %H:%M')} · {gpu}"
        + (f" ({vram:.0f} GB)" if vram else "")
        + f" · engine low-VRAM mode: "
        + {True: "on", False: "**off**", None: "unknown"}[lowvram],
        f"- {rows[0]['frames'] if rows else '?'} frames ({a.seconds:g} s) at "
        f"{a.aspect}, {a.steps} steps, seed {a.seed}, tiled decode",
        f"- prompt: {a.prompt[:160]}",
        "",
        "| base | H3 upscale | output | total | wait | sampling | s/step | "
        "upscale | refine | decode | audio | peak VRAM | min free RAM | RTX ×2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['base_mp']:g} MP {r['base_size']} | "
                         f"{'0.6 MP' if r['h3_upscale'] else 'off'} | "
                         f"failed: {r['error'][:80]} |" + " |" * 11)
            continue
        lines.append(
            f"| {r['base_mp']:g} MP {r['base_size']} "
            f"| {f'{a.upscale_mp:g} MP' if r['h3_upscale'] else 'off'} "
            f"| {r['out_size']} | {fmt_s(r.get('total'))} | {fmt_s(r.get('wait'))} "
            f"| {fmt_s(r.get('t_sample'))} | {fmt_s(r.get('s_per_step'))} "
            f"| {fmt_s(r.get('t_upscale')) if r['h3_upscale'] else '—'} "
            f"| {fmt_s(r.get('t_refine')) if r['h3_upscale'] else '—'} "
            f"| {fmt_s(r.get('t_decode'))} | {fmt_s(r.get('t_audio'))} "
            f"| {str(r.get('vram_peak_gb')) + ' GB' if r.get('vram_peak_gb') else '—'} "
            f"| {str(r.get('ram_low_free_gb')) + ' GB' if r.get('ram_low_free_gb') is not None else '—'} "
            f"| {fmt_s(r.get('rtx_total')) + ' → ' + r.get('rtx_size', '') if r.get('rtx_total') else '—'} |")
    notes = [r for r in rows if r.get("note")]
    if notes:
        lines += ["", "Notes:"] + [f"- {r['base_mp']:g} MP: {r['note']}" for r in notes]
    lines += ["", "Model loading happens inside the first node that needs the "
              "weights, so with `--cache-none` it is counted in *sampling* (and "
              "the text encoder's in *encode*) on every run. Peak VRAM is what "
              "the whole GPU had in use, other programs included.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
