# Copyright 2026 Friends for Humanity. Licensed under the Apache License, Version 2.0.
#
# Built on the Pika H3 Flex modular pipeline in diffusers (Apache-2.0). Loads no weights of its own: the model is
# MiniMax H3 with Pika's Flex Ref2VA adapter, each under its own licence.

r"""
FRef2VA for Pika H3 Flex: a first and/or last frame together with references and voice, in one request.

Pika H3 Flex ships two task adapters. FL2VA takes endpoint frames but no references; Ref2VA takes images, video and
audio references but no frames, and the diffusers pipeline picks one workflow or the other from its inputs. MiniMax H3
itself has no such split: a keyframe is a block of clean conditioning rows anchored on the generated video's timeline,
and the Ref2VA trunk honours keyframes placed next to its reference blocks (ComfyUI's `MiniMaxH3AddGuide` beside
`MiniMaxH3ReferenceToVideo` does exactly this, and SGLang documents it as a hybrid `ref2va` request).

These blocks run that hybrid on the Ref2VA trunk with the Flex Ref2VA adapter and Flex's six-step sampler:

    [ text | keyframe rows | reference blocks | target audio | target video ]

Only three things differ from Pika's `ref2va` workflow:
  - the keyframes are put on the canvas and VAE-encoded as well as the references;
  - the layout reserves one block of rows per keyframe, right after the text, anchored where the generated video
    starts (`image`) or ends (`last_image`) — the same anchors `fl2va` uses, measured from the origin the reference
    blocks leave behind;
  - the keyframe latents join the reference latents at the head of the conditioning list, so they are noised to
    the anchor level, packed and held fixed exactly like the references.

Keyframes stay in fp32, the Ref2VA adapter's `fp32_conditioning_v1` policy, not the FL2VA adapter's bf16 rounding.
"""

import numpy as np
import torch

from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.modular_pipelines.pika_h3_flex.before_denoise import (
    PikaH3FlexPrepareConditionLatentsStep,
    PikaH3FlexPrepareLatentsStep,
    PikaH3FlexRef2VAPrepareLatentsStep,
    PikaH3FlexRef2VAPrepareLayoutStep,
    PikaH3FlexSetTimestepsStep,
)
from diffusers.modular_pipelines.pika_h3_flex.before_encoder import PikaH3FlexRef2VASetupStep, PikaH3FlexResizeStep
from diffusers.modular_pipelines.pika_h3_flex.decoders import PikaH3FlexAfterDenoiseStep
from diffusers.modular_pipelines.pika_h3_flex.denoise import PikaH3FlexRef2VADenoiseStep
from diffusers.modular_pipelines.pika_h3_flex.modular_blocks_pika_h3_flex import PikaH3FlexDecodeStep
from diffusers.modular_pipelines.pika_h3_flex.encoders import (
    PikaH3FlexRef2VAReferenceEncoderStep,
    PikaH3FlexRef2VATextEncoderStep,
    encode_vae_condition,
)
from diffusers.models import AutoencoderKLMiniMaxH3

# Tested with diffusers at commit d0b9b1b5 (Pika's pin). These blocks extend Pika's ref2va layout step by overriding
# its `build_ref2va_packed_sequence`; fail with a clear message if a later diffusers moves it, rather than build a
# wrong layout.
if not callable(getattr(PikaH3FlexRef2VAPrepareLayoutStep, "build_ref2va_packed_sequence", None)):
    raise ImportError(
        "FRef2VA needs Pika's H3 Flex modular pipeline with PikaH3FlexRef2VAPrepareLayoutStep.build_ref2va_packed_sequence. "
        "Install the tested diffusers: pip install 'diffusers @ git+https://github.com/huggingface/diffusers.git@d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f'"
    )

# MiniMax H3's rotary geometry, copied from diffusers' pika_h3_flex/before_denoise.py (Apache-2.0) at d0b9b1b5 so these
# blocks do not depend on private helpers that diffusers may rename. tests/test_cpu.py checks the layout still matches
# Pika's own for the same references.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # numpy's linspace(endpoint=False), not torch.linspace: the float64 grid has to match H3's exactly
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows, num_audio_latents, rotary_time, width_grid, audio_channels) -> None:
    # channel-major audio rows: no height coordinate, pinned to the two extremes of their own block's width grid
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def keyframe_anchor_time(anchor: str, origin: float, num_latent_frames: int) -> float:
    """Rotary time of a keyframe block, `origin` being where the generated video's clock starts."""
    if anchor == "first":
        return origin
    if anchor == "last":
        # The span of the generated frames, summed pairwise as `fl2va` sums it, less one frame's slot.
        spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
        for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
            spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
        return origin + float(spans.sum()) - _ROPE_FRAME_RESCALE
    raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")


def build_fref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    keyframe_anchors: tuple[str, ...],
    references: list,
    condition_latents: list[torch.Tensor],
    audio_condition_latents: list[torch.Tensor],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int,
    video_tag: int,
):
    r"""
    Build `[text | keyframe rows | reference blocks | target audio | target video]`.

    `condition_latents` holds the reference latents only; the keyframes are counted from `keyframe_anchors`, one block
    of `rows_per_frame` rows each, sharing the target's spatial grid. Returns the same tuple as Pika's `ref2va` layout:
    `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the leading video and audio
    rows that are conditioning (keyframes and references) rather than generated.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_keyframe_rows = len(keyframe_anchors) * rows_per_frame
    num_target_video_rows = num_latent_frames * rows_per_frame
    num_target_audio_rows = num_audio_latents * audio_channels
    visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
    audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
    num_reference_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
    )
    num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
    sequence_length = (
        num_text_tokens
        + num_keyframe_rows
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Keyframe rows come straight after the text, but take no time on the shared clock: the reference blocks start
    # where the text ends, and the keyframes are placed on the generated video's own timeline once that is known.
    keyframe_start = num_text_tokens
    cursor = keyframe_start + num_keyframe_rows
    rotary_time = float(num_text_tokens)
    video_indices = [torch.arange(keyframe_start, cursor)]
    audio_indices = []
    for reference in references:
        if reference.kind == "image":
            frames, reference_height, reference_width = next(visual_geometry)
            num_video_rows = frames * (reference_height // patch_h) * (reference_width // patch_w)
            rows = slice(cursor, cursor + num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            num_audio_rows = next(audio_row_counts)
            reference_audio_latents = num_audio_rows // audio_channels
            rows = slice(cursor, cursor + num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids, rows, reference_audio_latents, rotary_time, target_width_grid, audio_channels
            )
            rotary_time += float(reference_audio_latents)
        elif reference.kind == "video":
            num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
            reference_audio_latents = num_audio_rows // audio_channels
            frames, reference_height, reference_width = next(visual_geometry)
            num_video_rows = frames * (reference_height // patch_h) * (reference_width // patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            _fill_audio_positions(
                position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid, audio_channels
            )
            frame_time = _temporal_position_grid(frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(frames, 1)
            # Summed sequentially, as Pika's `ref2va` layout sums a reference's span (see its comment on ulps).
            video_span = sum(
                _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                for index in range(frames)
            )
            rotary_time += max(float(reference_audio_latents), video_span)
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

    # The generated video's clock starts where the references leave it; the keyframes are anchored on that clock.
    for index, anchor in enumerate(keyframe_anchors):
        rows = slice(keyframe_start + index * rows_per_frame, keyframe_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = keyframe_anchor_time(anchor, rotary_time, num_latent_frames)
        position_ids[rows, 1:] = target_frame_grid

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid, audio_channels
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_keyframe_rows + num_reference_video_rows,
        num_reference_audio_rows,
    )


class FlexFRef2VAKeyframeEncodeStep(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    @property
    def description(self) -> str:
        return "Encodes the keyframes into fp32 conditioning latents, kept apart from the reference latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3)]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam(name="keyframes", type_hint=list, required=True)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "keyframe_latents",
                type_hint=list[torch.Tensor],
                description="One `(1, latent_channels, 1, latent_height, latent_width)` tensor per keyframe.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = components._execution_device
        block_state.keyframe_latents = [
            encode_vae_condition(
                components.vae,
                torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None],
                components.pixel_mean,
                components.pixel_std,
                components.keyframe_encode_seed,
            ).to(torch.float32)
            for image in block_state.keyframes
        ]
        self.set_block_state(state, block_state)
        return components, state


class FlexFRef2VAPrepareLayoutStep(PikaH3FlexRef2VAPrepareLayoutStep):
    model_name = "pika-h3-flex"

    @property
    def description(self) -> str:
        return "Builds `[text | keyframe rows | reference blocks | target audio | target video]` and its rotary grid."

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [InputParam(name="keyframe_anchors", type_hint=tuple, required=True)]

    # Pika's step resolves the latent geometry and calls this builder through `self`; ours adds the keyframe rows.
    def build_ref2va_packed_sequence(self, text_token_tags, *rest):
        return build_fref2va_packed_sequence(text_token_tags, self._keyframe_anchors, *rest)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        self._keyframe_anchors = self.get_block_state(state).keyframe_anchors
        return super().__call__(components, state)


class FlexFRef2VAMergeConditionsStep(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    @property
    def description(self) -> str:
        return "Puts the keyframe latents ahead of the reference latents, the order the layout reserved their rows in."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="keyframe_latents", type_hint=list, required=True),
            InputParam(name="condition_latents", type_hint=list, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("condition_latents", type_hint=list[torch.Tensor])]

    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        block_state.condition_latents = list(block_state.keyframe_latents) + list(block_state.condition_latents)
        self.set_block_state(state, block_state)
        return components, state


class FlexFRef2VABeforeEncodeStep(SequentialPipelineBlocks):
    model_name = "pika-h3-flex"
    # Keyframes first, so the canvas follows the first keyframe's aspect ratio unless `height`/`width` are given;
    # the reference setup then normalizes the references and resolves `num_frames` on that canvas.
    block_classes = [PikaH3FlexResizeStep, PikaH3FlexRef2VASetupStep]
    block_names = ["keyframes", "ref2va"]


class FlexFRef2VAVaeEncodeStep(SequentialPipelineBlocks):
    model_name = "pika-h3-flex"
    block_classes = [FlexFRef2VAKeyframeEncodeStep, PikaH3FlexRef2VAReferenceEncoderStep]
    block_names = ["keyframes", "references"]


class FlexFRef2VACoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "pika-h3-flex"
    block_classes = [
        FlexFRef2VAPrepareLayoutStep,
        FlexFRef2VAMergeConditionsStep,
        PikaH3FlexPrepareConditionLatentsStep,
        PikaH3FlexPrepareLatentsStep,
        PikaH3FlexRef2VAPrepareLatentsStep,
        PikaH3FlexSetTimestepsStep,
        PikaH3FlexRef2VADenoiseStep,
        PikaH3FlexAfterDenoiseStep,
    ]
    block_names = [
        "prepare_layout",
        "merge_conditions",
        "prepare_condition_latents",
        "prepare_latents",
        "prepare_ref2va_latents",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]


class FlexFRef2VABlocks(SequentialPipelineBlocks):
    """
    Pika H3 Flex with a first and/or last frame plus references and voice: pass `image` and/or `last_image` together
    with `references` (images, videos, audio) and `num_frames`. Load the Ref2VA adapter
    (`ref2va/adapter.safetensors`) onto `transformer_ref`.
    """

    model_name = "pika-h3-flex"
    block_classes = [
        FlexFRef2VABeforeEncodeStep,
        PikaH3FlexRef2VATextEncoderStep,
        FlexFRef2VAVaeEncodeStep,
        FlexFRef2VACoreDenoiseStep,
        PikaH3FlexDecodeStep,
    ]
    block_names = ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"]

    @property
    def description(self) -> str:
        return (
            "Pika H3 Flex FRef2VA: a first and/or last keyframe together with ordered image, video and audio "
            "references, on the Ref2VA trunk with the Flex Ref2VA adapter and Flex's six-step sampler."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos", description="The generated video."),
            OutputParam("audio", type_hint=torch.Tensor, description="The soundtrack, `(1, 2, num_samples)`."),
            OutputParam("sampling_rate", type_hint=int, description="Sample rate of the soundtrack in Hz."),
        ]
