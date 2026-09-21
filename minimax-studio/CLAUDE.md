# MiniMax Studio — invariants

## Hard rules

1. **`web/index.html` stays one file, no build step.** Same shell as the other
   three apps: rail, prompt bar over a masonry feed, popover settings.
2. **Never hard-code a ComfyUI workflow.** `comfy.py` builds from `/object_info`
   and matches inputs through candidate-name lists.
3. **Frame length must satisfy `(length - 5) % 17 == 0`.** H3 rejects anything
   else. `frame_length()` mirrors the workflow's ComfyMathExpression exactly:
   `n = max(5, round(sec*24)); n += (5 - n % 17) % 17`.
4. **Resolution snaps to a multiple of 32**, from megapixels and aspect. Verified
   against the workflow's own table (0.2 MP 16:9 → 608×352).
5. **Both VAEs are required.** The same AV latent is decoded twice — `VAEDecode`
   with the video VAE, `VAEDecodeAudio` with the audio VAE — then muxed by
   `CreateVideo`. Never drop the audio branch; H3's audio is generated, not added.
6. **The preflight tells the truth — and the truth is calibrated.** It
   measures VRAM, RAM and free disk against the real published file sizes,
   and its verdict comes from `assess()`, a pure function with unit tests.
   The calibration is a real run, not a guess: an RTX 4060 (8 GB) with 32 GB
   of RAM renders shots in minutes on this app's exact defaults, so
   8 GB VRAM / 32 GB RAM is "tight — proven workable", never "hard". Keep
   "hard" for what genuinely blocks: under ~7 GB of VRAM, RAM under ~70% of
   the peak, disk short of the download. Do not soften "hard", and do not
   re-harden "tight" — both directions misinform.
7. **Low-VRAM launch flags** (`--lowvram --cache-none`) are the default and are
   the only reason the DiT loads at all on a small card.
8. Python detection by execution, downloads resumable, model deletes path-checked
   — as in the sibling apps.

## Nodes the workflow uses that this app does not

`minimaxh3_r2v_with_upscale.json` also carries Pixaroma timer/monitor/free-VRAM
nodes, `ResolutionSelector`, `ComfyMathExpression`, `PrimitiveFloat/String` and
`ModelPreviewOverrideKJ`. All are canvas conveniences whose jobs the front end
does itself (length and size maths, prompt entry, VRAM hygiene between runs), so
their packs are deliberately not required. KJNodes and the RTX nodes are offered
because they add something the front end cannot do.

## Graceful degradation

The upscale branch checks for `MinimaxH3LatentUpscaler3D`, the LTXV A/V split
nodes and an upscaler model, and falls back to base size with a note on the clip
rather than failing. `ModelAttentionBackend` and `MiniMaxH3SigmaShift` are
skipped if absent. The RTX action is hidden unless the node is loaded.

## Continue this clip

The last frame is captured **in the browser** — the same-origin clip drawn
onto a canvas, exported as PNG, pushed through the ordinary `/api/upload`
path. No server-side ffmpeg, no new dependency; keep it that way. The source
clip's own reference images are carried by name (they are still in
ComfyUI/input), so `renderRefs` must keep tolerating refs with no local
thumbnail URL. Seek to `duration − 1/24` before drawing: the exact end of
some containers decodes to a blank frame.

## H3 input modes

The generator exposes every conditioning family on
`MiniMaxH3ReferenceToVideo`: prompt-only text-to-video, up to three subject
images, one reference video (decoded through `LoadVideo` and
`GetVideoComponents` into `ref_videos.ref_video_0`), and one reference audio.
Do not collapse the video reference to a still; its frame sequence is the
motion/composition signal. Inputs can be mixed.

## The board

`data/board.json` is server-side state like the gallery, sanitised on save
(`_clean_shot`: known keys only, capped list). Chaining is defined by array
order — shot *n* chains to shot *n−1* — which is why **Render remaining is
strictly sequential**: a chained shot cannot start until the clip before it
exists to capture a frame from. Board generation reuses `collectBase()` (the
settings popover) and `frameRef()` (the continue-clip capture); do not grow a
second copy of either. `updateBoardJobs` refreshes only the `.bstatus` zones
so a full re-render never eats a keystroke in a card's textarea.

## Two bugs worth not reintroducing

- `run_job` must use `built.get("seed")`: the RTX graph has no seed.
- An RTX upscale has no size of its own — the source clip's dimensions times the
  multiplier are passed through `params` so the gallery shows 1216×704 rather
  than `None × None`.

## Validation gate

```bash
python tests/run.py            # gate + units + graph + api, ~10 s
python tests/run.py gate       # just the compile/parse/id checks
```

The `gate` module runs what used to be done by hand: py_compile on every
module, `node --check` on the inline script, the diff of element ids the JS
uses against the ids in the markup, **and** a scan for interactive controls no
listener ever touches — removing a panel without removing its wiring is how
this app broke twice, and shipping controls without wiring them at all is how
it broke a third time (seconds chips, refs picker, the RTX upscale button).

`graph` and `api` run against `tests/mock_comfy.py`, whose schema is derived
from the reference workflows in assets/ and which validates prompts the way
ComfyUI does — so "accepted" there means the real server would take the graph
too. `units` pins the frame rule and the resolution table to the workflow's
own numbers, and checks the page's `frames()` against Python's
`frame_length()` (JS `%` keeps the sign; that pair has drifted once).
