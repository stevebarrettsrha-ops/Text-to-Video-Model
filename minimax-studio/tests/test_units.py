"""The maths and the model set, checked against the workflow's own tables.

The frame rule and the resolution table are not invented here: the length
expression is copied from the workflow's ComfyMathExpression and the sizes
from the MarkdownNote it ships with. If these drift, clips get rejected by
H3 itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bootstrap                                   # noqa: E402
import manager                                     # noqa: E402
from comfy import dimensions, frame_length         # noqa: E402
from harness import Suite                          # noqa: E402

# The workflow's MarkdownNote: megapixels at 16:9 -> width x height.
WORKFLOW_TABLE = {0.2: (608, 352), 0.3: (736, 416), 0.4: (864, 480),
                  0.5: (960, 544), 0.6: (1056, 608), 0.7: (1152, 640),
                  0.8: (1216, 672), 0.9: (1280, 736), 0.98: (1344, 768)}


def run(slow: bool = False) -> Suite:
    s = Suite("units")

    # -- frame length: (length - 5) % 17 == 0, and never below 5 ----------
    ok = all((frame_length(sec / 4) - 5) % 17 == 0 and frame_length(sec / 4) >= 5
             for sec in range(0, 121))
    s.check("every length from 0 to 30 s satisfies (n - 5) % 17 == 0", ok)
    s.equal("6 s becomes 158 frames, as the README promises",
            frame_length(6), 158)
    s.equal("the workflow's own 124-frame clip round-trips",
            frame_length(124 / 24), 124)
    s.check("rounding always goes up, never down",
            all(frame_length(sec / 4) >= max(5, round(sec / 4 * 24))
                for sec in range(0, 121)))

    # -- the page's own frames() must agree with the server ----------------
    # JS % keeps the sign where Python's does not; this pair has drifted once.
    import json as _json
    import re
    import shutil as _shutil
    import subprocess
    if _shutil.which("node"):
        page = (Path(__file__).resolve().parent.parent
                / "web" / "index.html").read_text()
        fn = re.search(r"function frames\(sec, fps\) \{[\s\S]*?\n\}", page)
        probe = (fn.group(0) + "\nconsole.log(JSON.stringify("
                 "Array.from({length: 121}, (_, i) => frames(i / 4, 24))));")
        out = subprocess.run(["node", "-e", probe], capture_output=True,
                             text=True)
        got = _json.loads(out.stdout) if out.returncode == 0 else None
        want = [frame_length(i / 4) for i in range(121)]
        s.check("the page's frames() matches frame_length() for 0-30 s",
                got == want,
                "" if got == want else
                next(f"{i/4} s: page {g}, server {w}"
                     for i, (g, w) in enumerate(zip(got or [], want))
                     if g != w))
    else:
        print("  --   node is not installed, so the page's frames() "
              "was not compared")

    # -- resolution: the workflow's own table, verbatim -------------------
    for mp, want in WORKFLOW_TABLE.items():
        s.equal(f"{mp} MP at 16:9 -> {want[0]}x{want[1]}",
                dimensions(mp, "16:9"), want)
    w, h = dimensions(0.2, "9:16")
    s.equal("9:16 is 16:9 turned on its side", (w, h), (352, 608))
    s.check("every output is a multiple of 32",
            all(v % 32 == 0 for mp in WORKFLOW_TABLE
                for v in dimensions(mp, "16:9")))
    s.equal("a junk aspect falls back to 16:9",
            dimensions(0.2, "banana"), dimensions(0.2, "16:9"))

    # -- the model set ------------------------------------------------------
    cfg = dict(bootstrap.DEFAULT_CONFIG)
    items = bootstrap.model_set(cfg)
    s.equal("the int8 set is five files", len(items), 5)
    s.check("both VAEs are in the set — audio is generated, not added",
            {i["name"] for i in items} >= {bootstrap.VIDEO_VAE["name"],
                                           bootstrap.AUDIO_VAE["name"]})
    s.check("every file knows where it goes",
            all(i["folder"] in manager.MODEL_FOLDERS for i in items))
    cfg["turbo"] = "4step"
    lora = [i for i in bootstrap.model_set(cfg) if i["folder"] == "loras"][0]
    s.check("the 4-step turbo comes from its own repo at the repo root",
            lora["repo"] == bootstrap.TURBO_REPO and "/" not in lora["path"])

    # -- guessing a folder from a path ---------------------------------------
    s.equal("explicit folder prefix wins",
            manager.guess_folder("vae/minimax_h3_video_vae_fp16.safetensors"),
            "vae")
    s.equal("qwen goes to text_encoders",
            manager.guess_folder("qwen3vl_32b_minimax_h3_int8.safetensors"),
            "text_encoders")
    s.equal("a lora-named repo goes to loras",
            manager.guess_folder("some_style.safetensors",
                                 "lightx2v/Minimax-h3-Turbo-lora"), "loras")
    s.equal("everything else is a diffusion model",
            manager.guess_folder("minimax_h3_ref2va.safetensors"),
            "diffusion_models")

    # -- deleting is path-checked ------------------------------------------
    s.fails_with("delete refuses a folder outside the list",
                 lambda: manager.delete_model({"models_dir": "/tmp"},
                                              "..", "x.safetensors"),
                 RuntimeError, "not allowed")
    s.fails_with("delete refuses a name that climbs out",
                 lambda: manager.delete_model({"models_dir": "/tmp"},
                                              "vae", "../../etc/passwd"),
                 RuntimeError, "not allowed")

    # -- setup progress: the machinery behind the bars ----------------------
    prog = bootstrap.Progress()
    snap = prog.snapshot()
    s.check("steps travel as a list, in run order — jsonify sorts dict keys",
            isinstance(snap["steps"], list)
            and [x["key"] for x in snap["steps"]]
            == [k for k, _ in bootstrap.Progress.STEPS])
    prog.begin("models")
    prog.track("models", 250, "over")
    s.equal("track clamps a runaway percentage",
            prog.snapshot()["steps"][4]["pct"], 100)
    prog.track("models", None, "no number yet")
    s.check("track(None) means an indeterminate bar, not 0%",
            prog.snapshot()["steps"][4]["pct"] is None)
    prog.track("models", 42.25)
    prog.finish("models")
    s.check("finishing a step clears its bar",
            prog.snapshot()["steps"][4]["pct"] is None
            and prog.snapshot()["steps"][4]["state"] == "done")

    s.check("pip raw progress lines parse",
            bootstrap.PIP_RAW.match("Progress 512 of 2048").groups()
            == ("512", "2048"))
    s.check("pip download lines yield the wheel's name",
            bootstrap.PIP_GET.match(
                "  Downloading https://x/torch-2.4.0-cp312.whl (2.4 GB)")
            .group(1).endswith("torch-2.4.0-cp312.whl"))
    s.check("git progress lines parse whichever phase",
            bootstrap.GIT_PHASE.search(
                "Receiving objects:  67% (1024/1522), 88.1 MiB").groups()
            == ("Receiving objects", "67"))
    got = []
    fake = bootstrap.Progress()
    fake.begin("nodes")
    cb = bootstrap._git_pct(fake, "nodes", "KJNodes", base=50, span=50)
    cb("Receiving objects", 100)
    got = fake.snapshot()["steps"][2]["pct"]
    s.check("a git slice stays inside its base..base+span window",
            50 <= got <= 100, f"pct {got}")
    s.equal("transfer lines read like a person would say them",
            bootstrap.fmt_transfer(1.5e9, 3e9, 12e6, 125),
            "1.50 GB of 3.00 GB · 12.0 MB/s · 2m 5s left")

    # -- preflight says something, whatever the machine ---------------------
    pf = bootstrap.preflight(cfg)
    s.check("preflight reports the download and peak sizes",
            pf["download"] > 50e9 and pf["peak"] > 30e9,
            f"download {pf['download']/1e9:.0f} GB, peak {pf['peak']/1e9:.0f} GB")
    s.check("preflight gives a verdict", pf["verdict"] in ("ok", "tight", "hard"))

    # -- the verdict is calibrated against the real 4060 run -----------------
    GB = 1e9
    peak, dl = 33e9, 54e9
    v, notes = bootstrap.assess(8 * GB, 32 * GB, 500 * GB, dl, peak)
    s.check("8 GB VRAM + 32 GB RAM — the proven 4060 case — is tight, "
            "never hard", v == "tight", f"got {v!r}")
    s.check("its notes say workable and point at the RTX route",
            any("proven workable" in n for n in notes)
            and any("RTX upscale" in n for n in notes))
    v, _ = bootstrap.assess(4 * GB, 32 * GB, 500 * GB, dl, peak)
    s.equal("4 GB of VRAM is genuinely hard", v, "hard")
    v, _ = bootstrap.assess(8 * GB, 16 * GB, 500 * GB, dl, peak)
    s.equal("16 GB of RAM against a 33 GB peak is hard", v, "hard")
    v, _ = bootstrap.assess(24 * GB, 64 * GB, 30 * GB, dl, peak)
    s.equal("a big card cannot outrun a full disk", v, "hard")
    v, _ = bootstrap.assess(24 * GB, 64 * GB, 500 * GB, dl, peak)
    s.equal("24 GB VRAM and room everywhere is ok", v, "ok")
    v, notes = bootstrap.assess(0, 32 * GB, 500 * GB, dl, peak)
    s.check("no GPU reading is not a verdict, just a note",
            any("install pytorch" in n.lower() for n in notes))

    # -- kill_pid tells the truth about a process that did stop -------------
    # A zombie keeps its pid until its parent reaps it, while holding no
    # sockets and running no code. Polling kill(pid, 0) counts one as alive,
    # so a process that died politely is reported as needing a SIGKILL.
    import os
    import subprocess
    import time as _time
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    for _ in range(50):
        if child.poll() is not None:
            break
        _time.sleep(0.05)
    s.check("an unreaped child is a zombie, not a running process",
            bootstrap._pid_gone(child.pid),
            "it still answers kill(pid, 0), so only /proc/<pid>/stat knows")
    child.wait()
    s.check("a reaped pid is gone too", bootstrap._pid_gone(child.pid))
    s.check("our own pid is not gone", not bootstrap._pid_gone(os.getpid()))

    sleeper = subprocess.Popen([sys.executable, "-c",
                               "import time; time.sleep(30)"])
    said = bootstrap.kill_pid(sleeper.pid)
    sleeper.wait()
    s.equal("kill_pid says stopped for one that took the SIGTERM",
            said, "stopped")
    s.equal("and already gone for one that was never there",
            bootstrap.kill_pid(sleeper.pid), "already gone")

    # -- an 8 GB card, as it reports itself -----------------------------------
    rtx4060 = 8_585_216_000                    # torch total_memory, in bytes
    verdict, notes = bootstrap.assess(rtx4060, 32 * 1024 ** 3, 10**12,
                                      50e9, 33e9)
    s.equal("an RTX 4060 + 32 GB is tight, as calibrated", verdict, "tight")
    s.check("and it reads as 8 GB, not 9", notes[0].startswith("8 GB"))
    verdict, notes = bootstrap.assess(rtx4060, 32 * 1024 ** 3, 10**12,
                                      50e9, 33e9, lowvram=False)
    s.check("without low-VRAM mode an 8 GB card is hard, and says why",
            verdict == "hard" and any("Low-VRAM" in n for n in notes))
    s.equal("the flag is read from the engine's own argv",
            [bootstrap.engine_lowvram({"argv": ["main.py", "--lowvram"]}),
             bootstrap.engine_lowvram({"argv": ["main.py"]}),
             bootstrap.engine_lowvram({})], [True, False, None])

    # -- the ComfyUI address, however it was typed ---------------------------
    s.equal("a trailing slash does not break the port",
            bootstrap.comfy_port("http://127.0.0.1:8188/"), 8188)
    s.equal("no port means ComfyUI's own",
            bootstrap.comfy_port("http://localhost"), 8188)
    s.equal("an explicit port is read",
            bootstrap.comfy_port(" http://127.0.0.1:9000 "), 9000)
    s.equal("the stored address loses its trailing slash",
            bootstrap.normal_url(" http://127.0.0.1:8188/ "),
            "http://127.0.0.1:8188")

    # -- download set takes each file from its own repo ----------------------
    seen = []
    real = manager.hf_download
    manager.hf_download = lambda cfg, repo, path, folder: seen.append(
        (repo, path, folder))
    try:
        cfg = dict(bootstrap.DEFAULT_CONFIG, turbo="4step",
                   hf_repo="someone/else", models_dir=str(Path(
                       __file__).resolve().parent / "no-such-models"))
        manager.download_set(cfg)
    finally:
        manager.hf_download = real
    lora = [x for x in seen if x[2] == "loras"]
    s.check("the 4-step LoRA downloads from the turbo repo root",
            lora == [(bootstrap.TURBO_REPO, bootstrap.TURBO_LORAS["4step"]["name"],
                      "loras")])
    s.check("a browsed repo does not redirect the set",
            all(x[0] != "someone/else" for x in seen))
    return s
