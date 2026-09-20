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
6. **The preflight tells the truth.** It measures VRAM, RAM and free disk and
   compares them to the real published file sizes. Do not soften its wording:
   21 GB DiT + 27 GB text encoder against 8 GB VRAM / 32 GB RAM is a "hard"
   verdict, and the person needs to know that before the download, not after.
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

## Two bugs worth not reintroducing

- `run_job` must use `built.get("seed")`: the RTX graph has no seed.
- An RTX upscale has no size of its own — the source clip's dimensions times the
  multiplier are passed through `params` so the gallery shows 1216×704 rather
  than `None × None`.

## Validation gate

```bash
python -m py_compile server.py comfy.py bootstrap.py manager.py
python - <<'PY'
import re, pathlib
src = pathlib.Path('web/index.html').read_text()
pathlib.Path('/tmp/mm.js').write_text('\n'.join(re.findall(r'<script>(.*?)</script>', src, re.S)))
PY
node --check /tmp/mm.js
```

After any big edit, also diff the element ids the JS uses against the ids in the
markup — removing a panel without removing its wiring is how this app broke
twice.
