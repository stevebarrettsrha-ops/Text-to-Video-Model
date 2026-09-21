# MiniMax Studio

A local frontend for **MiniMax H3** — reference images and a prompt in, a video
with its own audio out. Built around `minimaxh3_r2v_with_upscale.json`, with the
RTX super-resolution pass from `rtx_video_upscale.json` bolted on as a one-click
action on any finished clip.

---

## Read this before you download 56 GB

H3 is a large model and the weights are the floor:

| File | Size |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot` (the DiT) | 21 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot` (text encoder, a 32B model) | 27.1 GB |
| video VAE + audio VAE | 5.8 GB |
| 8-step turbo LoRA | 2.0 GB |

ComfyUI holds one of the two big files at a time, so the peak is about **33 GB
resident** and the download is about **56 GB**.

**8 GB of VRAM is workable — proven, not promised.** An RTX 4060 (8 GB)
renders H3 shots in minutes with exactly this app's defaults: INT8 weights,
the 8-step turbo LoRA, low-VRAM mode, 0.2 MP, tiled decode, the latent
upscale pass, and the RTX video upscale at the end. The weights stream from
system RAM, so a fast SSD and 32 GB of RAM matter; heavier settings (higher
megapixels, more steps, the 40-step base schedule) are where 8 GB stops
being practical.

The **Preflight** panel on the Engine page measures your actual VRAM, RAM and
free disk and says which of those applies, before anything is downloaded. If
it says *not enough headroom* — under ~7 GB of VRAM, RAM far below the peak,
or no disk for the download — the realistic routes are:

- Point Settings at a ComfyUI running on a rented 24 GB box. Setup's third
  option ("connect to a ComfyUI I start myself") exists for this.
- Use the RTX upscaler on its own. It runs on the NVIDIA SDK over decoded
  frames, holds no diffusion weights, and is comfortable on 8 GB.

Nothing here hides the numbers from you or pretends otherwise.

---

## Running it

**Windows** — `run.bat`  ·  **macOS / Linux** — `./run.sh` → <http://127.0.0.1:7804>

First launch offers three routes (existing ComfyUI / fresh managed install /
connect to one you run yourself), then installs the optional nodes, then
downloads the weights with resume and live progress.

Custom nodes, all optional:

- **ComfyUI-KJNodes** — live preview while a clip renders
- **NVIDIA RTX nodes** — `RTXVideoSuperResolution` for the upscale action
- **ComfyUI-Manager**

Low-VRAM mode is on by default and launches ComfyUI with `--lowvram
--cache-none`, which is what makes a 21 GB DiT loadable on a small card at all.

---

## Making a clip

The prompt bar holds everything: the shot description, aspect, length, up to
three **reference images** (H3 keeps those faces and outfits), a **reference
voice** (an audio file — H3 generates the clip's speech in that voice; press
the pill again to remove it; it rides with every shot, board cards included),
and Settings for the rest. None of it is required: a prompt alone is plain
text-to-video.

Two numbers are computed for you, the same way the workflow's helper nodes did
it:

- **Length** — `max(5, round(seconds × 24))`, rounded up so that
  `(frames − 5)` divides by 17, which is what H3 requires. Ask for 6 s and you
  get 158 frames, 6.58 s.
- **Size** — from megapixels and aspect, snapped to a multiple of 32. The
  defaults match the workflow's own table: 0.2 MP 16:9 → 608×352, 0.3 → 736×416,
  0.4 → 864×480, 0.5 → 960×544. Start at 0.2.

In Settings: steps (8, matching the turbo LoRA), shift and shift 2 (6 and 3),
sampler and scheduler, attention backend, seed, tiled decode (on — it keeps the
VAE step off the VRAM cliff), and the optional **latent upscale pass**, which
adds the workflow's `MinimaxH3LatentUpscaler3D` branch and a second 4-step
sampler at denoise 0.5. If those nodes or the upscaler model are absent, the
clip still renders at base size and the app says so on the clip rather than
failing.

Audio comes out with the video — H3 generates both, decoded by their own VAEs
and muxed by `CreateVideo`.

---

## The Board — the whole cut on one strip

The **Board** page is the multi-shot workflow made visual: one card per shot,
each carrying its prompt, its length and its rendered clip. By default every
shot is **chained** — generated with the previous shot's last frame as
reference 1 — so faces and places carry through the whole sequence. The tag
on each card turns chaining off; drag cards to reorder; size, steps and the
rest come from the generator's Settings.

**Render remaining** walks the strip in order and renders every written shot
that has no clip yet (in order, because a chained shot needs the clip before
it). **Play the cut** plays the rendered shots back to back in the player.
The board lives in `data/board.json`, next to the gallery, so it is still
there tomorrow. Final assembly — music, trims, transitions — is editor work;
every clip downloads from its lightbox.

## Continuing a clip

Open any finished clip and press **Continue this clip**. The clip's last frame
is read right in the browser, uploaded like any reference image, and loaded as
reference 1 in the generator — describe what happens next and press Generate.
The clip's own reference images ride along by name, so the same faces carry
into the next shot. That is how a long sequence is made with H3: one shot per
prompt, each starting from the frame the last one ended on, audio generated
with every clip, then the clips laid back to back in an editor.

Two things to know:

- H3 takes three references, so a continued shot keeps the frame plus the
  first two of the source clip's own reference images.
- The frame is a starting *reference*, not a frozen first frame — H3 keeps the
  subjects and the scene, not the exact pixels.

## Upscaling a finished clip

Open any clip and press **RTX upscale ×2**. The clip goes back to ComfyUI
through `LoadVideo` → `GetVideoComponents` → `RTXVideoSuperResolution` →
`CreateVideo` → `SaveVideo`, exactly as in `rtx_video_upscale.json`: audio and
frame rate are carried straight through from the source, quality ULTRA. A
608×352 clip comes back 1216×704.

This is the part of the app that will run properly on your hardware today.

---

## Layout

```
server.py      Flask API — clip jobs, gallery, upscaling, preflight, setup
bootstrap.py   Discovery, installs, weight downloads, preflight, ComfyUI process
manager.py     Dependency checks and installers, HuggingFace browsing
comfy.py       Builds the H3 and RTX graphs from ComfyUI's live schema
web/index.html The interface — one file, no build step
assets/        The two workflows this was built from
tests/         python tests/run.py — the suite, against a mock ComfyUI
data/          config.json, gallery.json, clips/
```

Port: `MINIMAX_STUDIO_PORT`. `MINIMAX_STUDIO_NO_BROWSER=1` stops it opening a tab.
