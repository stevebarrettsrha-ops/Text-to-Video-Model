"""comfy.py — the graphs handed to ComfyUI.

The mock validates every prompt the way ComfyUI does — unknown inputs, values
outside a combo's options, missing required inputs and dangling links are all
rejected — so "accepted" here means the real server would have taken it too.
The schema comes from the reference workflows in assets/, including the
awkward names (LTXVSeparateAVLatent's input really is `av_latent`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfy import ComfyClient, ComfyError          # noqa: E402
from harness import Suite, comfy                   # noqa: E402


def nodes_of(graph: dict, cls: str) -> list[tuple[str, dict]]:
    return [(nid, n) for nid, n in graph.items() if n["class_type"] == cls]


def run(slow: bool = False) -> Suite:
    s = Suite("graph")
    with comfy() as mock:
        client = ComfyClient(mock.url)

        # -- schema reads the whole design rests on -------------------------
        s.check("the H3 nodes are seen", client.has("MiniMaxH3ReferenceToVideo"))
        s.check("unets read from the combo",
                client.unets() == ["h3/minimax_h3_ref2va_pruned_int8_convrot"
                                   ".safetensors"])
        s.check("samplers come from KSampler", "euler" in client.samplers())
        s.check("attention backends listed",
                "pytorch attention" in client.attention_backends())
        s.check("capabilities report everything on",
                all(client.capabilities().values()))

        # -- the plain clip --------------------------------------------------
        built = client.build({"prompt": "a man walks to the window",
                              "seconds": 6, "megapixels": 0.2,
                              "aspect": "16:9", "steps": 8, "seed": 7})
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts the plain clip", True)
        s.equal("158 frames for six seconds", built["length"], 158)
        s.equal("608x352 for 0.2 MP 16:9",
                (built["width"], built["height"]), (608, 352))
        s.equal("the seed asked for is the seed used", built["seed"], 7)

        r2v = nodes_of(g, "MiniMaxH3ReferenceToVideo")[0][1]["inputs"]
        s.equal("length lands on the node", r2v["length"], 158)
        s.equal("ref_image_size is filled — a required V3 combo a real "
                "engine rejected as missing", r2v.get("ref_image_size"),
                "match")
        dec_v = nodes_of(g, "VAEDecode")
        dec_a = nodes_of(g, "VAEDecodeAudio")
        s.check("both decodes are present — video and audio",
                len(dec_v) == 1 and len(dec_a) == 1)
        s.check("the two decodes read the same latent",
                dec_v[0][1]["inputs"]["samples"]
                == dec_a[0][1]["inputs"]["samples"])
        s.check("they decode through different VAEs",
                dec_v[0][1]["inputs"]["vae"] != dec_a[0][1]["inputs"]["vae"])
        create = nodes_of(g, "CreateVideo")[0][1]["inputs"]
        s.check("CreateVideo muxes the audio branch in",
                create.get("audio") == [dec_a[0][0], 0])
        sampler = nodes_of(g, "KSampler")[0][1]["inputs"]
        s.check("the model chain runs through LoRA and sigma shift",
                sampler["model"][0]
                == nodes_of(g, "MiniMaxH3SigmaShift")[0][0])

        # -- reference images -------------------------------------------------
        class FakeUpload:
            filename = "face.png"
            stream = b"png"
            mimetype = "image/png"

        name = client.upload(FakeUpload())
        s.equal("upload comes back under its own name", name, "face.png")
        built = client.build({"prompt": "same face", "refs": [name, name],
                              "seconds": 2})
        client.queue(built["prompt"])
        s.check("ComfyUI accepts a clip with reference images", True)
        r2v = nodes_of(built["prompt"], "MiniMaxH3ReferenceToVideo")[0][1]["inputs"]
        s.check("refs land on ref_images.ref_image_0 and _1",
                "ref_images.ref_image_0" in r2v and "ref_images.ref_image_1" in r2v)

        # -- a reference voice --------------------------------------------------
        class FakeVoice:
            filename = "my_voice.wav"
            stream = b"wav"
            mimetype = "audio/wav"

        voice = client.upload(FakeVoice())
        built = client.build({"prompt": "he speaks", "voice": voice})
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts a clip with a reference voice", True)
        load = nodes_of(g, "LoadAudio")
        s.check("the voice loads through LoadAudio",
                bool(load) and load[0][1]["inputs"]["audio"] == "my_voice.wav")
        r2v = nodes_of(g, "MiniMaxH3ReferenceToVideo")[0][1]["inputs"]
        s.check("and lands on ref_audios.ref_audio_0",
                r2v.get("ref_audios.ref_audio_0") == [load[0][0], 0])
        built = client.build({"prompt": "no voice"})
        s.check("without a voice, no audio loader is added",
                not nodes_of(built["prompt"], "LoadAudio"))

        # -- tiled decode -----------------------------------------------------
        built = client.build({"prompt": "x", "tiled_decode": True})
        s.check("tiled decode swaps the video decode node",
                bool(nodes_of(built["prompt"], "VAEDecodeTiled")))
        client.queue(built["prompt"])
        s.check("ComfyUI accepts the tiled clip", True)

        # -- the latent upscale pass -----------------------------------------
        built = client.build({"prompt": "x", "upscale": True, "seed": 3})
        g = built["prompt"]
        s.equal("no fallback note — the branch was added", built["note"], "")
        s.check("the app reports the clip as upscaled", bool(built["upscaled"]))
        client.queue(g)
        s.check("ComfyUI accepts the upscale graph", True)
        sep = nodes_of(g, "LTXVSeparateAVLatent")
        s.check("the A/V split is wired through av_latent",
                bool(sep) and "av_latent" in sep[0][1]["inputs"])
        cat = nodes_of(g, "LTXVConcatAVLatent")[0]
        s.check("audio latent is carried around the upscaler",
                cat[1]["inputs"]["audio_latent"] == [sep[0][0], 1])
        second = [n for _, n in nodes_of(g, "KSampler")
                  if n["inputs"]["denoise"] < 1.0]
        s.check("the second sampler runs at denoise 0.5",
                len(second) == 1 and second[0]["inputs"]["denoise"] == 0.5)
        s.equal("both samplers share the seed",
                {n["inputs"]["seed"] for _, n in nodes_of(g, "KSampler")}, {3})

        # -- the RTX pass on a finished clip ----------------------------------
        class FakeClip:
            filename = "clip.mp4"
            stream = b"mp4"
            mimetype = "video/mp4"

        vid = client.upload(FakeClip())
        built = client.build_rtx_upscale(vid, 2, "ULTRA")
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts the RTX graph", True)
        s.check("the RTX graph carries no seed — run_job must cope",
                "seed" not in built)
        rtx = nodes_of(g, "RTXVideoSuperResolution")[0][1]["inputs"]
        s.equal("quality ULTRA as in the workflow", rtx["quality"], "ULTRA")
        comp = nodes_of(g, "GetVideoComponents")[0]
        create = nodes_of(g, "CreateVideo")[0][1]["inputs"]
        s.check("audio and fps pass straight through from the source",
                create["audio"] == [comp[0], 1] and create["fps"] == [comp[0], 2])

    # -- graceful degradation, in a ComfyUI missing the extras ---------------
    with comfy(MOCK_OMIT="MinimaxH3LatentUpscaler3D,RTXVideoSuperResolution,"
                         "ModelAttentionBackend,MiniMaxH3SigmaShift,"
                         "LoraLoaderModelOnly,ModelPreviewOverrideKJ") as bare:
        client = ComfyClient(bare.url)
        caps = client.capabilities()
        s.check("capabilities admit what is missing",
                not caps["rtx"] and not caps["upscaler"] and caps["r2v"])
        built = client.build({"prompt": "x", "upscale": True})
        s.check("no upscaler node -> base size with a note",
                "not installed" in built["note"] and not built["upscaled"])
        client.queue(built["prompt"])
        s.check("the degraded graph is still a valid prompt", True)
        s.check("sigma shift and attention are simply skipped",
                not nodes_of(built["prompt"], "MiniMaxH3SigmaShift"))
        s.fails_with("RTX upscale without the node says how to fix it",
                     lambda: client.build_rtx_upscale("v.mp4"),
                     ComfyError, "not installed")

    with comfy(MOCK_NO_UPSCALER_MODEL="1") as nomodel:
        client = ComfyClient(nomodel.url)
        built = client.build({"prompt": "x", "upscale": True})
        s.check("upscaler node without its model -> base size with a note",
                "No upscaler model" in built["note"])
        client.queue(built["prompt"])
        s.check("that graph is accepted too", True)
    return s
