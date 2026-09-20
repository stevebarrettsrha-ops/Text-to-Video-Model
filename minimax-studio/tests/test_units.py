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

    # -- preflight says something, whatever the machine ---------------------
    pf = bootstrap.preflight(cfg)
    s.check("preflight reports the download and peak sizes",
            pf["download"] > 50e9 and pf["peak"] > 30e9,
            f"download {pf['download']/1e9:.0f} GB, peak {pf['peak']/1e9:.0f} GB")
    s.check("preflight gives a verdict", pf["verdict"] in ("ok", "tight", "hard"))
    return s
