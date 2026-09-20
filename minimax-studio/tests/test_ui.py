"""The interface itself, in a browser.

Wants Playwright and a Chromium; says so and steps aside when either is
missing (MM_CHROMIUM points at a browser executable if Playwright's own
download is not installed). The mock cannot encode video, so the page records
a real one-second webm itself — canvas.captureStream into MediaRecorder — and
hands it to the mock to serve as every render. That makes the clips genuinely
decodable, which is what the continue-this-clip capture needs.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Suite, Workspace, comfy, fake_weights, studio  # noqa: E402

RECORD_WEBM = """
async () => {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 36;
  const ctx = c.getContext('2d');
  const stream = c.captureStream(12);
  const rec = new MediaRecorder(stream, { mimeType: 'video/webm' });
  const parts = [];
  rec.ondataavailable = e => parts.push(e.data);
  const done = new Promise(r => rec.onstop = r);
  rec.start();
  for (let i = 0; i < 14; i++) {
    ctx.fillStyle = 'rgb(' + i * 18 + ',60,200)';
    ctx.fillRect(0, 0, 64, 36);
    await new Promise(r => setTimeout(r, 80));
  }
  rec.stop();
  await done;
  const buf = await new Blob(parts).arrayBuffer();
  return Array.from(new Uint8Array(buf));
}
"""


def chromium_path() -> str:
    """Playwright's own browser when installed, else the machine's."""
    if os.environ.get("MM_CHROMIUM"):
        return os.environ["MM_CHROMIUM"]
    for cand in ("/opt/pw-browsers/chromium", shutil.which("chromium"),
                 shutil.which("chromium-browser"), shutil.which("google-chrome")):
        if cand and Path(cand).exists():
            return cand
    return ""       # let Playwright try its own download


def available() -> str:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return "Playwright is not installed (pip install playwright)"
    return ""


def run(slow: bool = False) -> Suite:
    from playwright.sync_api import sync_playwright

    s = Suite("ui")
    with comfy(delay=0.4) as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models) as app, sync_playwright() as p:
            try:
                exe = chromium_path()
                browser = p.chromium.launch(executable_path=exe or None)
            except Exception as exc:  # noqa: BLE001
                print(f"  --   skipped: no Chromium to drive ({str(exc)[:80]})")
                return s
            errors: list[str] = []
            pg = browser.new_page(viewport={"width": 1400, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(app.url)
            pg.wait_for_timeout(1200)

            # a real webm for the mock to serve as every render
            data = bytes(pg.evaluate(RECORD_WEBM))
            requests.post(mock.url + "/testvideo", data=data, timeout=10)
            s.check("the page recorded a webm for the mock to serve",
                    len(data) > 1000, f"{len(data)} bytes")

            # -- the prompt bar's own maths ---------------------------------
            pg.click('[data-view="create"]')
            s.check("the feed is not nested inside the sticky prompt bar",
                    not pg.evaluate("!!document.querySelector('.barwrap .feed')"))
            pg.click('#segSeconds button[data-v="8"]')
            s.equal("the 8 s chip drives the computed length",
                    pg.text_content("#lenVal"), "8.0s")
            pg.click("#btnSettings")
            pg.eval_on_selector(
                "#sec-sl", "el => { el.value = 4; "
                "el.dispatchEvent(new Event('input')) }")
            s.equal("the length slider agrees with the server's rounding",
                    pg.text_content("#lenVal"), "4.5s")
            s.check("the slider re-syncs the chips",
                    pg.eval_on_selector('#segSeconds button[data-v="4"]',
                                        "el => el.classList.contains('on')"))
            pg.click('#segUpscale button[data-v="on"]')
            s.check("the upscale toggle flips",
                    pg.eval_on_selector('#segUpscale button[data-v="on"]',
                                        "el => el.classList.contains('on')"))
            pg.click("#btnCloseSettings")
            with pg.expect_file_chooser(timeout=4000) as fc:
                pg.click("#btnRefs")
            s.check("the references button opens a file chooser",
                    fc.value is not None)
            pg.keyboard.press("Escape")

            # -- a clip, from the bar to the feed ------------------------------
            pg.fill("#description", "a man walks to the window")
            pg.click("#btnGenerate")
            pg.wait_for_selector("#feed .tile video", timeout=30000)
            s.check("the clip lands in the feed as a tile", True)

            # -- continue this clip ----------------------------------------------
            # scroll clear of the sticky prompt bar, as a person would see it
            pg.eval_on_selector("#feed .tile video",
                                "el => el.scrollIntoView({block: 'center'})")
            pg.click("#feed .tile video")
            pg.wait_for_selector("#lightbox:not([hidden])")
            pg.click("#lbContinue")
            pg.wait_for_selector("#refsRow:not([hidden])", timeout=20000)
            s.check("continuing puts the last frame in the references row",
                    pg.eval_on_selector_all("#refsRow img", "els => els.length")
                    == 1)
            s.check("the prompt is cleared for the next shot",
                    pg.eval_on_selector("#description", "el => el.value") == "")
            s.check("the lightbox closed and the generator opened",
                    pg.eval_on_selector("#lightbox", "el => el.hidden"))
            frame_name = pg.eval_on_selector(
                "#refsRow .cap span", "el => el.textContent")
            s.check("the frame was uploaded under a continue_ name",
                    str(frame_name).startswith("continue_"))
            pg.fill("#description", "he opens the window")
            pg.click("#btnGenerate")
            pg.wait_for_function(
                "document.querySelectorAll('#feed .tile').length >= 2",
                timeout=30000)
            s.check("the continued clip renders", True)
            prompts = requests.get(mock.url + "/prompts", timeout=10).json()
            last = list(prompts.values())[-1]
            r2v = [n for n in last.values()
                   if n["class_type"] == "MiniMaxH3ReferenceToVideo"][0]
            s.check("the continued clip's graph carries the frame as ref 0",
                    "ref_images.ref_image_0" in r2v["inputs"])
            loads = [n["inputs"]["image"] for n in last.values()
                     if n["class_type"] == "LoadImage"]
            s.check("LoadImage reads the uploaded frame",
                    any(str(v).startswith("continue_") for v in loads))

            # -- RTX upscale from the lightbox ------------------------------------
            pg.eval_on_selector("#feed .tile video",
                                "el => el.scrollIntoView({block: 'center'})")
            pg.click("#feed .tile video")
            pg.wait_for_selector("#lightbox:not([hidden])")
            s.check("the RTX button shows when the node is loaded",
                    pg.eval_on_selector("#lbUpscale", "el => !el.hidden"))
            pg.click("#lbUpscale")
            pg.wait_for_function(
                "document.querySelectorAll('#feed .tile').length >= 3",
                timeout=30000)
            s.check("the upscale lands as another tile", True)

            s.check("no page errors the whole way through", not errors,
                    "; ".join(errors)[:120])
            browser.close()
    return s
