# Copyright 2026 Friends for Humanity. Licensed under the Apache License, Version 2.0.
"""CPU self-test, no weights, a few seconds. Run it after installing or upgrading diffusers (or ComfyUI):

    python -m pytest tests/test_cpu.py -q        # or: python tests/test_cpu.py

1. With no keyframes, the FRef2VA layout is Pika's own ref2va layout, row for row: so the copied rotary helpers still
   match whatever diffusers you have installed.
2. With a first, a last, or both keyframes, the layout matches ComfyUI's MiniMax H3 packed layout (Add Guide next to
   Reference to Video), which is how the tested ComfyUI workflows run. Skipped when ComfyUI is not importable; set
   COMFYUI_PATH to a ComfyUI checkout to run it.
"""

import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from block import build_fref2va_packed_sequence  # noqa: E402
from diffusers.modular_pipelines.pika_h3_flex.before_denoise import PikaH3FlexRef2VAPrepareLayoutStep  # noqa: E402

PATCH = (1, 2, 2)
AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG = 2, 2, 0
TEXT = 37
LATENT_T, LATENT_H, LATENT_W = 72, 52, 30  # 10 s at 480x832
AUDIO_T = 405


def _references():
    """Two pictures (one portrait, one landscape) and a 4 s voice."""
    refs = [SimpleNamespace(kind="image", has_audio=False), SimpleNamespace(kind="image", has_audio=False),
            SimpleNamespace(kind="audio", has_audio=True)]
    latents = [torch.zeros(1, 24, 1, 52, 30), torch.zeros(1, 24, 1, 30, 52)]
    audio = [torch.zeros(160 * AUDIO_CHANNELS, 32)]
    return refs, latents, audio


def _ours(anchors):
    refs, latents, audio = _references()
    return build_fref2va_packed_sequence(torch.ones(TEXT, dtype=torch.long), tuple(anchors), refs, latents, audio,
                                         LATENT_T, LATENT_H, LATENT_W, AUDIO_T, PATCH, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG)


def test_matches_pikas_layout_without_keyframes():
    refs, latents, audio = _references()
    theirs = PikaH3FlexRef2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(TEXT, dtype=torch.long), refs, latents, audio, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, PATCH,
        AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG)
    ours = _ours(())
    for name, a, b in zip(("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"), ours, theirs):
        assert torch.equal(a, b), f"{name} differs from Pika's ref2va layout: diffusers changed; see README, Compatibility"
    assert ours[5:] == theirs[5:]


def _comfy_layout():
    path = os.environ.get("COMFYUI_PATH")
    if path:
        sys.path.insert(0, path)
    try:
        from comfy.ldm.minimax.model import PackedLayout
    except Exception:
        return None
    return PackedLayout


def test_matches_comfyui_with_keyframes():
    PackedLayout = _comfy_layout()
    if PackedLayout is None:
        import pytest
        pytest.skip("ComfyUI not importable; set COMFYUI_PATH")
    frames = sum((1, 4, 4, 4, 4)[k % 5] for k in range(LATENT_T))
    refs = [{"kind": "image", "latent_h": 52, "latent_w": 30}, {"kind": "image", "latent_h": 30, "latent_w": 52},
            {"kind": "audio", "ref_audio_t": 160}]
    for anchors in (("first",), ("last",), ("first", "last")):
        keyframes = [{"resolved_frame_index": 0 if a == "first" else frames - 1,
                      "latent": torch.zeros(1, 24, 1, LATENT_H, LATENT_W)} for a in anchors]
        comfy = PackedLayout(TEXT, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, keyframes=keyframes, refs=refs)
        ours = _ours(anchors)[0]
        assert ours.shape == comfy.position_ids.shape, f"{anchors}: {tuple(ours.shape)} vs {tuple(comfy.position_ids.shape)}"
        assert torch.allclose(ours, comfy.position_ids, atol=1e-9), f"{anchors}: rotary positions differ from ComfyUI"


if __name__ == "__main__":
    test_matches_pikas_layout_without_keyframes()
    print("ok   layout matches Pika's ref2va layout without keyframes")
    if _comfy_layout() is None:
        print("skip ComfyUI comparison (set COMFYUI_PATH to a ComfyUI checkout)")
    else:
        test_matches_comfyui_with_keyframes()
        print("ok   layout matches ComfyUI's packed layout for first, last and first + last")
