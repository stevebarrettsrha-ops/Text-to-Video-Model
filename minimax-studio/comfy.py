"""
comfy.py - talks to ComfyUI and builds the MiniMax H3 graphs.

Built from /object_info rather than a stored workflow, so a renamed input shows
up as a clear message instead of a silently wrong value.

Generation mirrors minimaxh3_r2v_with_upscale.json:

  UNETLoader ─ LoraLoaderModelOnly(turbo) ─ ModelAttentionBackend ─
    MiniMaxH3SigmaShift ─ KSampler ─┬─ VAEDecode(video vae) ─┐
                                    └─ VAEDecodeAudio(audio) ─┤
  CLIPLoader ─┐                                               ├─ CreateVideo ─ SaveVideo
  VAELoader×2 ┴─ MiniMaxH3ReferenceToVideo ─ positive, LATENT ─┘
                          │
                          └─ ConditioningZeroOut ─ negative

Two things the front end computes rather than asking nodes to do, because the
workflow used helper packs for them:

  length  = max(5, round(seconds * fps)), rounded up so (length - 5) % 17 == 0
  width/height from megapixels and aspect, snapped to a multiple of 32

The upscale pass (separate A/V latents, MinimaxH3LatentUpscaler3D, a second
short KSampler at denoise 0.5) is added only when those nodes and the upscaler
model are actually present.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid

import requests

R2V = "MiniMaxH3ReferenceToVideo"
SIGMA_SHIFT = "MiniMaxH3SigmaShift"
ATTENTION = "ModelAttentionBackend"
UPSCALER = "MinimaxH3LatentUpscaler3D"
RTX_UPSCALE = "RTXVideoSuperResolution"
PREVIEW = "ModelPreviewOverrideKJ"


class ComfyError(RuntimeError):
    pass


def frame_length(seconds: float, fps: int = 24) -> int:
    """H3 wants (length - 5) divisible by 17. Same maths as the workflow's
    expression node: max(5, round(a*24)) rounded up to the next valid count."""
    n = max(5, int(round(seconds * fps)))
    return n + (5 - (n % 17)) % 17


def dimensions(megapixels: float, aspect: str = "16:9", multiple: int = 32):
    try:
        w_ratio, h_ratio = (float(x) for x in aspect.split(":"))
    except Exception:
        w_ratio, h_ratio = 16.0, 9.0
    unit = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    w = max(multiple, int(round(w_ratio * unit / multiple)) * multiple)
    h = max(multiple, int(round(h_ratio * unit / multiple)) * multiple)
    return w, h


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188") -> None:
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None
        self._schema_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def schema(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._schema is None or time.time() - self._schema_at > 120:
                r = requests.get(f"{self.url}/object_info", timeout=30)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def has(self, class_type: str) -> bool:
        return class_type in self.schema()

    def node_inputs(self, class_type: str) -> dict:
        info = self.schema().get(class_type)
        if not info:
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. MiniMax H3 needs a "
                "recent ComfyUI — update it from the Engine page.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    REQUIRED_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", R2V,
                      "ConditioningZeroOut", "KSampler", "VAEDecode",
                      "VAEDecodeAudio", "CreateVideo", "SaveVideo")

    def ensure_supported(self) -> None:
        missing = [n for n in self.REQUIRED_NODES if not self.has(n)]
        if missing:
            raise ComfyError("This ComfyUI cannot run MiniMax H3 — it is missing "
                             + ", ".join(missing) +
                             ". Update ComfyUI from the Engine page, then "
                             "restart it.")

    @staticmethod
    def _combo_options(spec) -> list:
        """The options of a combo input, whichever schema wrote it: classic
        [[options], {...}] or the newer ["COMBO", {"options": [...]}] that
        recent nodes (MiniMax H3 among them) are served with."""
        if not isinstance(spec, (list, tuple)) or not spec:
            return []
        kind = spec[0]
        if isinstance(kind, list):
            return kind
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(kind, str) and kind.upper().startswith(
                ("COMBO", "COMFY_DYNAMICCOMBO")):
            options = opts.get("options") or []
            if options and isinstance(options[0], dict):
                return [o.get("key") for o in options]
            return list(options)
        return []

    def _enum(self, class_type: str, name: str) -> list[str]:
        try:
            spec = self.node_inputs(class_type).get(name)
        except ComfyError:
            return []
        return [str(v) for v in self._combo_options(spec)]

    def unets(self) -> list[str]:
        return self._enum("UNETLoader", "unet_name")

    def clips(self) -> list[str]:
        return self._enum("CLIPLoader", "clip_name")

    def vaes(self) -> list[str]:
        return self._enum("VAELoader", "vae_name")

    def loras(self) -> list[str]:
        for cls in ("LoraLoaderModelOnly", "LoraLoader"):
            if self.has(cls):
                vals = self._enum(cls, "lora_name")
                if vals:
                    return vals
        return []

    def samplers(self) -> list[str]:
        return self._enum("KSamplerSelect", "sampler_name") or \
            self._enum("KSampler", "sampler_name")

    def schedulers(self) -> list[str]:
        return self._enum("KSampler", "scheduler")

    def attention_backends(self) -> list[str]:
        return self._enum(ATTENTION, "backend") or self._enum(ATTENTION, "attention")

    def upscaler_models(self) -> list[str]:
        for name in ("model_name", "upscaler", "ckpt_name", "model"):
            vals = self._enum(UPSCALER, name)
            if vals:
                return vals
        return []

    def images(self) -> list[str]:
        return self._enum("LoadImage", "image")

    def videos(self) -> list[str]:
        return self._enum("LoadVideo", "file") or self._enum("LoadVideo", "video")

    def capabilities(self) -> dict:
        return {"r2v": self.has(R2V), "sigma_shift": self.has(SIGMA_SHIFT),
                "attention": self.has(ATTENTION), "upscaler": self.has(UPSCALER),
                "rtx": self.has(RTX_UPSCALE), "lora": self.has("LoraLoaderModelOnly"),
                "preview": self.has(PREVIEW) and bool(self.preview_vaes())}

    def preview_vaes(self) -> list[str]:
        """The tiny VAEs the KJ preview node can decode with (taeh3 for H3)."""
        for name in ("tiny_vae", "vae_name"):
            vals = self._enum(PREVIEW, name)
            if vals:
                return vals
        return []

    # ------------------------------------------------------------------ #
    # picking files
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick(names: list[str], wanted: str, contains: list[str],
              avoid: list[str] | None = None) -> str:
        if wanted and wanted in names:
            return wanted
        for n in names:
            low = n.lower().replace("\\", "/")
            if all(c in low for c in contains) and \
                    not any(a in low for a in (avoid or [])):
                return n
        return ""

    def resolve_models(self, p: dict) -> dict:
        unets, clips, vaes = self.unets(), self.clips(), self.vaes()
        dit = self._pick(unets, p.get("dit", ""), ["minimax", "ref2va"]) or \
            self._pick(unets, "", ["minimax"])
        if not dit:
            raise ComfyError(
                "ComfyUI's model list has no MiniMax H3 DiT. If the file is "
                "already on disk, restart ComfyUI from the Engine page — it "
                "scans its model folders once, at startup, so weights that "
                "arrived later are invisible until then. Otherwise download "
                "it on the Models page.")
        clip = self._pick(clips, p.get("clip", ""), ["qwen3vl"]) or \
            (clips[0] if clips else "")
        if not clip:
            raise ComfyError("No text encoder found. Download "
                             "qwen3vl_32b_minimax_h3_*.safetensors.")
        video_vae = self._pick(vaes, p.get("video_vae", ""),
                               ["minimax", "video"]) or \
            self._pick(vaes, "", ["video_vae"])
        audio_vae = self._pick(vaes, p.get("audio_vae", ""), ["minimax", "audio"]) or \
            self._pick(vaes, "", ["audio_vae"])
        if not video_vae or not audio_vae:
            raise ComfyError("H3 needs both VAEs — the video one and the audio "
                             "one. Download whichever is missing on the Models "
                             "page.")
        lora = ""
        if p.get("lora") is not False:
            lora = self._pick(self.loras(), p.get("lora", ""),
                              ["minimax", "turbo"])
        return {"dit": dit, "clip": clip, "video_vae": video_vae,
                "audio_vae": audio_vae, "lora": lora}

    # ------------------------------------------------------------------ #
    # graph building
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match(available: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        low = {k.lower(): k for k in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    @staticmethod
    def _dynamic_options(definition) -> dict:
        """A V3 dynamic combo's options: {key: {sub_input: definition}}.

        Newer nodes (the RTX upscaler's resize_type, the H3 latent
        upscaler's mode) hang settings off a dropdown. The prompt carries
        them as "<combo>.<sub>" — resize_type.scale, mode.megapixels — and
        only for the option chosen; a missing one is a rejected prompt.
        """
        if not isinstance(definition, (list, tuple)) or len(definition) < 2:
            return {}
        kind, opts = definition[0], definition[1]
        if not (isinstance(kind, str) and kind.upper().startswith(
                "COMFY_DYNAMICCOMBO") and isinstance(opts, dict)):
            return {}
        out = {}
        for option in opts.get("options") or []:
            if not isinstance(option, dict) or "key" not in option:
                continue
            ins = option.get("inputs") or {}
            merged = {}
            if "required" in ins or "optional" in ins:
                merged.update(ins.get("required") or {})
                merged.update(ins.get("optional") or {})
            else:
                merged.update(ins)
            out[option["key"]] = merged
        return out

    def _default(self, definition):
        """(True, value) for an input ComfyUI would want filled, else (False, None)."""
        if not isinstance(definition, (list, tuple)) or not definition:
            return False, None
        kind = definition[0]
        opts = definition[1] if len(definition) > 1 else {}
        if not isinstance(opts, dict):
            opts = {}
        combo = self._combo_options(definition)
        if combo:
            # both combo schemas — a V3 combo left unfilled is how a
            # required ref_image_size went missing on a real engine
            return True, opts.get("default", combo[0])
        if kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            if "default" in opts:
                return True, opts["default"]
            if kind == "STRING":
                return True, ""
        return False, None

    def _node(self, class_type: str, wanted: dict) -> dict:
        spec = self.node_inputs(class_type)
        dynamic = {name: self._dynamic_options(d) for name, d in spec.items()}
        dynamic = {k: v for k, v in dynamic.items() if v}
        # sub-inputs are matchable by their prompt name, "<combo>.<sub>"
        available = dict(spec)
        for name, options in dynamic.items():
            for subs in options.values():
                for sub, d in subs.items():
                    available.setdefault(f"{name}.{sub}", d)
        inputs: dict = {}
        for key, want in wanted.items():
            name = self._match(available, want["names"])
            if name is None:
                if want.get("required"):
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This ComfyUI "
                        "does not match the MiniMax Studio graph — update it.")
                continue
            inputs[name] = want["value"]
        for name, definition in spec.items():
            if name in inputs or name == "control_after_generate":
                continue
            has, value = self._default(definition)
            if has:
                inputs[name] = value
        # the chosen option's sub-inputs, filled; other options' dropped
        for name, options in dynamic.items():
            chosen = options.get(inputs.get(name), {})
            for key in [k for k in inputs if k.startswith(name + ".")]:
                if key[len(name) + 1:] not in chosen:
                    del inputs[key]
            for sub, definition in chosen.items():
                full = f"{name}.{sub}"
                if full not in inputs:
                    has, value = self._default(definition)
                    if has:
                        inputs[full] = value
        return {"class_type": class_type, "inputs": inputs}

    def build(self, p: dict) -> dict:
        """p: prompt, refs[] (filenames already in ComfyUI/input), seconds,
        fps, megapixels, aspect, steps, cfg, sampler, scheduler, shift,
        shift_2, seed, attention, upscale, upscale_mp, tiled_decode."""
        self.ensure_supported()
        files = self.resolve_models(p)
        seed = int(p.get("seed") if p.get("seed") not in (None, "") else
                   random.randint(0, 2**40))
        fps = int(p.get("fps") or 24)
        seconds = float(p.get("seconds") or 6)
        length = frame_length(seconds, fps)
        width, height = dimensions(float(p.get("megapixels") or 0.2),
                                   p.get("aspect") or "16:9")
        g: dict = {}

        g["1"] = self._node("UNETLoader", {
            "unet": {"names": ["unet_name"], "value": files["dit"],
                     "required": True},
            "dtype": {"names": ["weight_dtype"],
                      "value": p.get("weight_dtype") or "default"}})
        clip_wanted = {"clip": {"names": ["clip_name"], "value": files["clip"],
                                "required": True}}
        types = self._enum("CLIPLoader", "type")
        if types:
            clip_wanted["type"] = {"names": ["type"],
                                   "value": "minimax" if "minimax" in types
                                   else types[0]}
        g["2"] = self._node("CLIPLoader", clip_wanted)
        g["3"] = self._node("VAELoader", {
            "vae": {"names": ["vae_name"], "value": files["video_vae"],
                    "required": True}})
        g["4"] = self._node("VAELoader", {
            "vae": {"names": ["vae_name"], "value": files["audio_vae"],
                    "required": True}})

        # model chain: turbo LoRA, attention backend, sigma shift
        model_ref: list = ["1", 0]
        if files["lora"] and self.has("LoraLoaderModelOnly"):
            g["5"] = self._node("LoraLoaderModelOnly", {
                "model": {"names": ["model"], "value": model_ref, "required": True},
                "lora": {"names": ["lora_name"], "value": files["lora"],
                         "required": True},
                "strength": {"names": ["strength_model", "strength"],
                             "value": float(p.get("lora_strength", 1.0))}})
            model_ref = ["5", 0]
        if self.has(ATTENTION) and p.get("attention"):
            g["6"] = self._node(ATTENTION, {
                "model": {"names": ["model"], "value": model_ref, "required": True},
                "backend": {"names": ["backend", "attention"],
                            "value": p["attention"]}})
            model_ref = ["6", 0]
        if self.has(SIGMA_SHIFT):
            g["7"] = self._node(SIGMA_SHIFT, {
                "model": {"names": ["model"], "value": model_ref, "required": True},
                # the node calls them shift_video and shift_audio
                "shift": {"names": ["shift_video", "shift", "sigma_shift"],
                          "value": float(p.get("shift", 6))},
                "shift_2": {"names": ["shift_audio", "shift_2", "shift2"],
                            "value": float(p.get("shift_2", 3))}})
            model_ref = ["7", 0]

        # live preview, as the workflow wires it: KJ's override on the base
        # sampler's model, decoding each step with taeh3. The refine pass
        # keeps the plain model, as in the workflow.
        sample_ref = model_ref
        tae = self._pick(self.preview_vaes(), "", ["taeh3"]) \
            if p.get("preview", True) and self.has(PREVIEW) else ""
        if tae:
            g["8"] = self._node(PREVIEW, {
                "model": {"names": ["model"], "value": model_ref,
                          "required": True},
                "tiny_vae": {"names": ["tiny_vae", "vae_name"], "value": tae,
                             "required": True},
                "suppress": {"names": ["suppress_default_preview"],
                             "value": False},
                "fps": {"names": ["preview_fps"], "value": fps}})
            sample_ref = ["8", 0]

        # Reference images preserve subjects and appearance. A reference video
        # contributes its decoded frame sequence through ref_video_0, giving H3
        # motion/composition to follow instead of reducing it to one still.
        refs = [r for r in (p.get("refs") or []) if r][:3]
        r2v_wanted = {
            "clip": {"names": ["clip"], "value": ["2", 0], "required": True},
            "vae": {"names": ["vae"], "value": ["3", 0], "required": True},
            "audio_vae": {"names": ["audio_vae"], "value": ["4", 0],
                          "required": True},
            "prompt": {"names": ["prompt", "text"], "value": p.get("prompt", ""),
                       "required": True},
            "width": {"names": ["width"], "value": width},
            "height": {"names": ["height"], "value": height},
            "length": {"names": ["length", "frames"], "value": length},
            # the workflow's fifth widget: how reference images are resized.
            # 'match' is what minimaxh3_r2v_with_upscale.json ships with.
            "ref_size": {"names": ["ref_image_size"],
                         "value": p.get("ref_image_size") or "match"},
        }
        # a reference voice: H3 speaks with it (ref_audios.ref_audio_0)
        if p.get("voice") and self.has("LoadAudio"):
            g["14"] = self._node("LoadAudio", {
                "audio": {"names": ["audio", "file"], "value": p["voice"],
                          "required": True}})
            r2v_wanted["ref_voice"] = {
                "names": ["ref_audios.ref_audio_0", "ref_audio_0",
                          "ref_audio"],
                "value": ["14", 0]}
        if p.get("ref_video"):
            if not self.has("LoadVideo") or not self.has("GetVideoComponents"):
                raise ComfyError("Reference video needs LoadVideo and "
                                 "GetVideoComponents. Update ComfyUI from the "
                                 "Engine page, then restart it.")
            g["15"] = self._node("LoadVideo", {
                "video": {"names": ["file", "video"],
                          "value": p["ref_video"], "required": True}})
            g["16"] = self._node("GetVideoComponents", {
                "video": {"names": ["video"], "value": ["15", 0],
                          "required": True}})
            r2v_wanted["ref_video"] = {
                "names": ["ref_videos.ref_video_0", "ref_video_0",
                          "ref_video"],
                "value": ["16", 0]}
        for index, name in enumerate(refs):
            node_id = str(10 + index)
            g[node_id] = self._node("LoadImage", {
                "image": {"names": ["image"], "value": name, "required": True}})
            r2v_wanted[f"ref_{index}"] = {
                "names": [f"ref_images.ref_image_{index}", f"ref_image_{index}",
                          f"ref_images_{index}"],
                "value": [node_id, 0]}
        g["20"] = self._node(R2V, r2v_wanted)
        g["21"] = self._node("ConditioningZeroOut", {
            "conditioning": {"names": ["conditioning"], "value": ["20", 0],
                             "required": True}})

        g["22"] = self._node("KSampler", {
            "model": {"names": ["model"], "value": sample_ref, "required": True},
            "positive": {"names": ["positive"], "value": ["20", 0],
                         "required": True},
            "negative": {"names": ["negative"], "value": ["21", 0],
                         "required": True},
            "latent": {"names": ["latent_image"], "value": ["20", 1],
                       "required": True},
            "seed": {"names": ["seed", "noise_seed"], "value": seed},
            "steps": {"names": ["steps"], "value": int(p.get("steps") or 8)},
            "cfg": {"names": ["cfg"], "value": float(p.get("cfg") or 1.0)},
            "sampler": {"names": ["sampler_name"],
                        "value": p.get("sampler") or "euler"},
            "scheduler": {"names": ["scheduler"],
                          "value": p.get("scheduler") or "simple"},
            "denoise": {"names": ["denoise"], "value": 1.0}})

        latent_ref: list = ["22", 0]
        note = ""
        if p.get("upscale"):
            latent_ref, note = self._add_upscale(g, latent_ref, model_ref, p, seed)

        decode_class = "VAEDecodeTiled" if (p.get("tiled_decode")
                                            and self.has("VAEDecodeTiled")) \
            else "VAEDecode"
        decode_wanted = {
            "samples": {"names": ["samples"], "value": latent_ref,
                        "required": True},
            "vae": {"names": ["vae"], "value": ["3", 0], "required": True}}
        if decode_class == "VAEDecodeTiled":
            decode_wanted["tile"] = {"names": ["tile_size"],
                                     "value": int(p.get("tile_size") or 512)}
            decode_wanted["overlap"] = {"names": ["overlap"], "value": 64}
        g["30"] = self._node(decode_class, decode_wanted)
        g["31"] = self._node("VAEDecodeAudio", {
            "samples": {"names": ["samples"], "value": latent_ref,
                        "required": True},
            "vae": {"names": ["vae"], "value": ["4", 0], "required": True}})
        g["32"] = self._node("CreateVideo", {
            "images": {"names": ["images"], "value": ["30", 0], "required": True},
            "audio": {"names": ["audio"], "value": ["31", 0]},
            "fps": {"names": ["fps"], "value": fps}})
        g["33"] = self._node("SaveVideo", {
            "video": {"names": ["video"], "value": ["32", 0], "required": True},
            "prefix": {"names": ["filename_prefix"], "value": "video/MiniMaxH3"}})

        return {"prompt": g, "seed": seed, "files": files, "length": length,
                "width": width, "height": height, "fps": fps,
                "seconds": round(length / fps, 2), "note": note,
                "upscaled": bool(note == "")and bool(p.get("upscale")),
                "preview": bool(tae)}

    def _add_upscale(self, g: dict, latent_ref: list, model_ref: list,
                     p: dict, seed: int):
        """The workflow's upscale branch, added only if it can actually run."""
        if not self.has(UPSCALER):
            return latent_ref, ("The latent upscaler node is not installed, so "
                                "the clip was rendered at base size.")
        models = self.upscaler_models()
        if not models:
            return latent_ref, ("No upscaler model in ComfyUI, so the clip was "
                                "rendered at base size.")
        sep = "LTXVSeparateAVLatent"
        cat = "LTXVConcatAVLatent"
        if not (self.has(sep) and self.has(cat)):
            return latent_ref, ("The A/V latent split nodes are missing, so the "
                                "clip was rendered at base size.")
        g["40"] = self._node(sep, {
            "latent": {"names": ["av_latent", "latent", "samples"],
                       "value": latent_ref, "required": True}})
        wanted = {
            "model": {"names": ["model_name", "upscaler", "ckpt_name", "model"],
                      "value": models[0], "required": True},
            "latent": {"names": ["latent", "samples"], "value": ["40", 0],
                       "required": True},
            "mode": {"names": ["mode", "resize_type"], "value": "megapixels"},
            # a dynamic combo: the target size lives at mode.megapixels
            "megapixels": {"names": ["mode.megapixels",
                                     "resize_type.megapixels", "megapixels",
                                     "value"],
                           "value": float(p.get("upscale_mp") or 0.6)},
            "align": {"names": ["align", "multiple_of"], "value": 32},
            # chunked over time: the whole clip's latent never sits in VRAM
            "chunking": {"names": ["enable_temporal_chunking", "use_tiling"],
                         "value": True},
        }
        # the workflow's low-VRAM choices, set only where this node offers them
        for key, value in (("force_unload", "cuda"), ("device", "cuda"),
                           ("precision", "fp16")):
            if value in self._enum(UPSCALER, key):
                wanted[key] = {"names": [key], "value": value}
        g["41"] = self._node(UPSCALER, wanted)
        g["42"] = self._node(cat, {
            "video": {"names": ["video_latent", "latent", "samples"],
                      "value": ["41", 0], "required": True},
            "audio": {"names": ["audio_latent", "audio"], "value": ["40", 1]}})
        g["43"] = self._node("KSampler", {
            "model": {"names": ["model"], "value": model_ref, "required": True},
            "positive": {"names": ["positive"], "value": ["20", 0],
                         "required": True},
            "negative": {"names": ["negative"], "value": ["21", 0],
                         "required": True},
            "latent": {"names": ["latent_image"], "value": ["42", 0],
                       "required": True},
            "seed": {"names": ["seed", "noise_seed"], "value": seed},
            "steps": {"names": ["steps"], "value": int(p.get("upscale_steps") or 4)},
            "cfg": {"names": ["cfg"], "value": 1.0},
            "sampler": {"names": ["sampler_name"], "value": p.get("sampler") or "euler"},
            "scheduler": {"names": ["scheduler"], "value": p.get("scheduler") or "simple"},
            "denoise": {"names": ["denoise"], "value": float(p.get("upscale_denoise") or 0.5)}})
        return ["43", 0], ""

    # ------------------------------------------------------------------ #
    # RTX super resolution on a finished clip
    # ------------------------------------------------------------------ #
    def build_rtx_upscale(self, video_name: str, scale: int = 2,
                          quality: str = "ULTRA") -> dict:
        if not self.has(RTX_UPSCALE):
            raise ComfyError("The NVIDIA RTX nodes are not installed. Add them "
                             "from the Engine page, then restart ComfyUI.")
        for node in ("LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"):
            if not self.has(node):
                raise ComfyError(f"This ComfyUI has no '{node}' node — update it.")
        g = {}
        g["1"] = self._node("LoadVideo", {
            "file": {"names": ["file", "video"], "value": video_name,
                     "required": True}})
        g["2"] = self._node("GetVideoComponents", {
            "video": {"names": ["video"], "value": ["1", 0], "required": True}})
        g["3"] = self._node(RTX_UPSCALE, {
            "images": {"names": ["images"], "value": ["2", 0], "required": True},
            "mode": {"names": ["resize_type"], "value": "scale by multiplier"},
            "scale": {"names": ["resize_type.scale", "scale", "multiplier"],
                      "value": int(scale)},
            "quality": {"names": ["quality"], "value": quality}})
        g["4"] = self._node("CreateVideo", {
            "images": {"names": ["images"], "value": ["3", 0], "required": True},
            "audio": {"names": ["audio"], "value": ["2", 1]},
            "fps": {"names": ["fps"], "value": ["2", 2]}})
        g["5"] = self._node("SaveVideo", {
            "video": {"names": ["video"], "value": ["4", 0], "required": True},
            "prefix": {"names": ["filename_prefix"], "value": "video/MiniMaxH3_rtx"}})
        return {"prompt": g}

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(f"{self.url}/prompt", json=body, timeout=60)
        if r.status_code >= 400:
            try:
                raise ComfyError(_readable(r.json()))
            except ValueError:
                raise ComfyError(r.text[:400])
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    def cancel(self, prompt_id: str) -> None:
        """Stop this prompt and only this one.

        A bare /interrupt stops whatever is running, which may be another
        job; a prompt still waiting is taken off the queue instead.
        """
        try:
            r = requests.get(f"{self.url}/queue", timeout=10)
            r.raise_for_status()
            q = r.json()
        except Exception:
            return
        running = {e[1] for e in q.get("queue_running") or []
                   if isinstance(e, list) and len(e) > 1}
        if prompt_id in running:
            self.interrupt()
        else:
            try:
                requests.post(f"{self.url}/queue",
                              json={"delete": [prompt_id]}, timeout=10)
            except Exception:
                pass

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=20)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    VIDEO_SUFFIX = (".mp4", ".webm", ".mkv", ".mov", ".gif", ".avi")

    def outputs(self, prompt_id: str) -> list[dict]:
        hist = self.history(prompt_id)
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for key in ("videos", "video", "gifs", "images", "files"):
                for item in node_out.get(key, []) or []:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    if item.get("type") == "temp":
                        continue
                    if key in ("videos", "video", "gifs", "files") or \
                            item["filename"].lower().endswith(self.VIDEO_SUFFIX):
                        found.append(item)
        return found

    def failed(self, prompt_id: str) -> str | None:
        status = (self.history(prompt_id).get("status") or {})
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    msg = str(data.get("exception_message", ""))
                    if "out of memory" in msg.lower():
                        return (f"{data.get('node_type')}: out of memory. Drop "
                                "the megapixels or the clip length, and make "
                                "sure low-VRAM mode is on in Settings.")
                    return f"{data.get('node_type')}: {msg}"
            return "ComfyUI reported an error while generating."
        return None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return requests.get(f"{self.url}/view", params=params, stream=True,
                            timeout=600)

    def upload(self, file_storage, kind: str = "image") -> str:
        files = {"image": (file_storage.filename, file_storage.stream,
                           file_storage.mimetype or "application/octet-stream")}
        r = requests.post(f"{self.url}/upload/image", files=files,
                          data={"type": "input", "overwrite": "false"}, timeout=600)
        r.raise_for_status()
        data = r.json()
        name = data.get("name") or file_storage.filename
        sub = data.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name


def _readable(err: dict) -> str:
    for node_id, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            return (f"{info.get('class_type', 'node ' + str(node_id))}: "
                    f"{e.get('message')} {e.get('details', '')}".strip())
    top = err.get("error") or {}
    if top:
        return f"{top.get('message', 'Rejected by ComfyUI')} " \
               f"{top.get('details', '')}".strip()
    return json.dumps(err)[:300]
