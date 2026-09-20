"""The HTTP surface, end to end: a real server.py against a mock ComfyUI.

This is the run-through a person would do by hand — set up, check status,
make a clip, watch the job, open the gallery, upscale the clip, delete it —
with the failure paths a person would eventually hit too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (Suite, Workspace, comfy, fake_weights,  # noqa: E402
                     finish_jobs, free_port, hub, studio, wait_for)


def run(slow: bool = False) -> Suite:
    s = Suite("api")
    with comfy() as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models) as app:
            # -- the shell and status ---------------------------------------
            page = requests.get(app.url + "/", timeout=10)
            s.check("the page is served",
                    page.ok and "MiniMax Studio" in page.text)
            st = requests.get(app.url + "/api/status", timeout=10).json()
            s.check("status says ready — engine, nodes and weights",
                    st["ready"] and st["comfy_online"] and st["nodes_ready"])
            s.equal("no missing models with the set on disk",
                    st["missing_models"], [])
            s.check("samplers and schedulers come from the live schema",
                    "euler" in st.get("samplers", [])
                    and "simple" in st.get("schedulers", []))
            s.check("the setup sheet gets its turbo list",
                    set(st.get("turbos", {})) == {"8step", "4step", "ref2v4"})
            s.check("precisions carry labels for the setup sheet",
                    st["precisions"]["int8"]["label"].startswith("INT8"))

            # -- guard rails --------------------------------------------------
            r = requests.post(app.url + "/api/generate", json={}, timeout=10)
            s.check("an empty ask is a 400 with advice",
                    r.status_code == 400 and "reference" in r.json()["error"])

            # -- a clip, end to end -------------------------------------------
            r = requests.post(app.url + "/api/generate",
                              json={"prompt": "a man walks to the window",
                                    "seconds": 6, "megapixels": 0.2,
                                    "aspect": "16:9", "steps": 8, "seed": 11},
                              timeout=30)
            s.check("generate starts a job", r.ok and len(r.json()["jobs"]) == 1)
            jobs = finish_jobs(app.url)
            job = jobs[0]
            s.equal("the job finishes", job["status"], "done")
            s.check("progress reached 100", job["pct"] == 100)
            clips = requests.get(app.url + "/api/clips", timeout=10).json()
            s.equal("one clip in the gallery", len(clips), 1)
            clip = clips[0]
            s.check("the gallery entry keeps the whole recipe",
                    clip["seed"] == 11 and clip["width"] == 608
                    and clip["height"] == 352 and clip["length"] == 158
                    and clip["steps"] == 8 and clip["kind"] == "clip")
            s.check("seconds are the delivered ones, not the asked ones",
                    abs(clip["seconds"] - 158 / 24) < 0.01)
            body = requests.get(f"{app.url}/api/clip/{clip['id']}", timeout=10)
            s.check("the clip streams back as video",
                    body.ok and body.content[4:8] == b"ftyp"
                    and "video/mp4" in body.headers.get("Content-Type", ""))

            # -- the RTX pass on that clip --------------------------------------
            r = requests.post(f"{app.url}/api/upscale/{clip['id']}",
                              json={"scale": 2}, timeout=30)
            s.check("upscale starts from the gallery", r.ok)
            finish_jobs(app.url)
            clips = requests.get(app.url + "/api/clips", timeout=10).json()
            s.equal("the upscale lands as a second clip", len(clips), 2)
            up = [c for c in clips if c["kind"] == "rtx"][0]
            s.check("the gallery shows source size times the multiplier, "
                    "not None x None",
                    up["width"] == 1216 and up["height"] == 704)
            s.check("the RTX job carries no seed", up["seed"] is None)
            s.check("audio length and fps ride along",
                    up["fps"] == clip["fps"] and up["seconds"] == clip["seconds"])
            r = requests.post(app.url + "/api/upscale/nonsense", json={},
                              timeout=10)
            s.equal("upscaling a clip that does not exist is a 404",
                    r.status_code, 404)

            # -- delete ----------------------------------------------------------
            gone = requests.delete(f"{app.url}/api/clip/{clip['id']}",
                                   timeout=10)
            s.check("delete says ok", gone.ok)
            clips = requests.get(app.url + "/api/clips", timeout=10).json()
            s.check("the clip is out of the gallery",
                    clip["id"] not in [c["id"] for c in clips])
            s.check("its file is off the disk",
                    not list((ws / "data" / "clips").glob(clip["file"])))
            s.equal("fetching it now is a 404",
                    requests.get(f"{app.url}/api/clip/{clip['id']}",
                                 timeout=10).status_code, 404)

            # -- the rest of the surface -----------------------------------------
            pf = requests.get(app.url + "/api/preflight", timeout=30).json()
            s.check("preflight measures and gives a verdict",
                    pf["verdict"] in ("ok", "tight", "hard")
                    and pf["download"] > 0)
            r = requests.post(app.url + "/api/config",
                              json={"torch_index": "https://example/whl"},
                              timeout=10)
            s.check("config saves", r.ok)
            deps = requests.get(app.url + "/api/deps", timeout=60).json()
            s.equal("the saved torch index comes back",
                    deps["torch_index"], "https://example/whl")
            s.check("the dependency report covers the engine",
                    any(i["id"] == "engine" and i["state"] == "ok"
                        for i in deps["items"]))
            hf = requests.get(app.url + "/api/hf/settings", timeout=10).json()
            s.check("hf settings list the curated sets",
                    set(hf["curated"]["sets"]) == {"int8", "fp8", "nvfp4"})
            s.check("no token yet, and no token leaked",
                    hf["token_set"] is False and "hf_token" not in hf)
            local = requests.get(app.url + "/api/hf/local", timeout=10).json()
            s.equal("the five fake weights are listed",
                    len(local["models"]), 5)
            r = requests.post(app.url + "/api/hf/settings",
                              json={"token": "hf_secret1234"}, timeout=10)
            hf = requests.get(app.url + "/api/hf/settings", timeout=10).json()
            s.check("a saved token is acknowledged by hint only",
                    r.ok and hf["token_set"] and hf["token_hint"] == "…1234"
                    and "hf_secret1234" not in str(hf))
            s.equal("the task list starts empty",
                    requests.get(app.url + "/api/tasks", timeout=10).json(), [])

            # -- the board ---------------------------------------------------
            s.equal("the board starts empty",
                    requests.get(app.url + "/api/board", timeout=10).json(), [])
            shots = [{"id": "s1", "prompt": "the window", "seconds": 6,
                      "chain": True, "clip": "abc123"},
                     {"id": "s2", "prompt": "he turns", "seconds": 4,
                      "chain": False, "clip": "",
                      "junk": "dropped", "seed": 999}]
            r = requests.post(app.url + "/api/board", json=shots, timeout=10)
            s.check("the board saves", r.ok and r.json()["shots"] == 2)
            back = requests.get(app.url + "/api/board", timeout=10).json()
            s.check("the board round-trips in order",
                    [b["id"] for b in back] == ["s1", "s2"]
                    and back[0]["clip"] == "abc123"
                    and back[1]["chain"] is False)
            s.check("foreign keys are stripped on the way in",
                    all(set(b) == {"id", "prompt", "seconds", "chain", "clip"}
                        for b in back))
            r = requests.post(app.url + "/api/board",
                              json={"nonsense": 1}, timeout=10)
            s.equal("a board that is not a list is a 400", r.status_code, 400)
            s.check("a bad save does not clobber the stored board",
                    len(requests.get(app.url + "/api/board",
                                     timeout=10).json()) == 2)

            # -- setup, in the connect-to-my-own-ComfyUI mode -------------------
            r = requests.post(app.url + "/api/setup/start",
                              json={"mode": "external"}, timeout=10)
            s.check("setup starts", r.ok)
            def setup_state():
                return requests.get(app.url + "/api/setup/state?since=0",
                                    timeout=10).json()
            def step(st, key):
                return [x for x in st["steps"] if x["key"] == key][0]
            wait_for(lambda: setup_state()["done"] or setup_state()["error"], 30)
            st = setup_state()
            s.check("external setup walks every step to done",
                    st["done"] and not st["error"],
                    st.get("error") or "")
            s.check("the steps arrive as a list, in run order — not "
                    "alphabetised by jsonify",
                    isinstance(st["steps"], list)
                    and [x["key"] for x in st["steps"]]
                    == ["python", "comfyui", "nodes", "deps", "models",
                        "launch"])
            s.check("the steps say what external mode skipped",
                    "your own ComfyUI" in step(st, "nodes")["detail"]
                    and step(st, "models")["state"] == "done")

            # -- deleting a weight, and the status noticing ----------------------
            vae = [m for m in local["models"] if m["folder"] == "vae"][0]
            r = requests.delete(app.url + "/api/hf/local",
                                json={"folder": "vae", "name": vae["name"]},
                                timeout=10)
            s.check("a weight can be deleted from the Models page", r.ok)
            st = requests.get(app.url + "/api/status", timeout=10).json()
            s.check("status immediately reports it missing",
                    vae["name"] in st["missing_models"] and not st["ready"])
            r = requests.delete(app.url + "/api/hf/local",
                                json={"folder": "..", "name": "x"}, timeout=10)
            s.equal("a path-climbing delete is refused", r.status_code, 400)

    # -- setup that actually downloads: the bar must move ----------------------
    with comfy() as mock, hub() as hf_hub, Workspace() as ws:
        models = ws / "models"
        models.mkdir(parents=True)
        with studio(mock.url, ws / "data", models,
                    hf_endpoint=hf_hub.url) as app:
            requests.post(hf_hub.url + "/mock/mode", json={"slow": 0.25},
                          timeout=10)
            r = requests.post(app.url + "/api/setup/start",
                              json={"mode": "external"}, timeout=10)
            s.check("setup with missing weights starts", r.ok)
            seen_pct, seen_detail = [], ""
            def sample():
                nonlocal seen_detail
                st = requests.get(app.url + "/api/setup/state?since=0",
                                  timeout=10).json()
                m = [x for x in st["steps"] if x["key"] == "models"][0]
                if isinstance(m.get("pct"), (int, float)):
                    seen_pct.append(m["pct"])
                    seen_detail = m["detail"] or seen_detail
                return st["done"] or st["error"]
            wait_for(sample, timeout=60, step=0.1)
            st = requests.get(app.url + "/api/setup/state?since=0",
                              timeout=10).json()
            s.check("the download setup finishes",
                    st["done"] and not st["error"], st.get("error") or "")
            s.check("the models bar showed real percentages on the way",
                    any(0 < p < 100 for p in seen_pct),
                    f"saw {sorted(set(int(p) for p in seen_pct))[:12]}")
            s.check("the bar covers the whole set and only moves forward",
                    seen_pct == sorted(seen_pct) and seen_pct[-1:] != [0],
                    f"{len(seen_pct)} samples")
            s.check("the detail line names the file and the byte counts",
                    "of" in seen_detail and "(" in seen_detail, seen_detail[:80])
            got = {p.name for p in models.rglob("*.safetensors")}
            s.check("all five weights landed on disk", len(got) == 5,
                    str(sorted(got)))
            st_now = requests.get(app.url + "/api/status", timeout=10).json()
            s.check("the app is ready once they land",
                    st_now["ready"] and st_now["missing_models"] == [])

    # -- ComfyUI down: refuse work, stay standing -----------------------------
    with Workspace() as ws:
        dead = f"http://127.0.0.1:{free_port()}"
        with studio(dead, ws / "data") as app:
            st = requests.get(app.url + "/api/status", timeout=10).json()
            s.check("status admits the engine is offline",
                    not st["comfy_online"] and not st["ready"])
            r = requests.post(app.url + "/api/generate",
                              json={"prompt": "x"}, timeout=10)
            s.check("generate is a 503 that names the Engine page",
                    r.status_code == 503 and "Engine" in r.json()["error"])

    # -- a render that dies: the person is told why ----------------------------
    with comfy(MOCK_FAIL_AFTER="1") as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models) as app:
            requests.post(app.url + "/api/generate",
                          json={"prompt": "doomed"}, timeout=30)
            jobs = finish_jobs(app.url)
            s.equal("the job reports the error", jobs[0]["status"], "error")
            s.check("out-of-memory advice points at the fix",
                    "megapixels" in jobs[0]["error"]
                    and "low-VRAM" in jobs[0]["error"])
            s.equal("nothing lands in the gallery",
                    requests.get(app.url + "/api/clips", timeout=10).json(), [])

    # -- cancelling a run ---------------------------------------------------------
    with comfy(delay=8.0) as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models) as app:
            r = requests.post(app.url + "/api/generate",
                              json={"prompt": "slow"}, timeout=30)
            job_id = r.json()["jobs"][0]
            wait_for(lambda: requests.get(app.url + "/api/jobs",
                                          timeout=10).json(), 10)
            requests.post(f"{app.url}/api/jobs/{job_id}/cancel", timeout=10)
            jobs = finish_jobs(app.url, timeout=30)
            s.equal("a cancelled job says cancelled",
                    jobs[0]["status"], "cancelled")
    return s
