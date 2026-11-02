# Copyright 2026 Friends for Humanity. Licensed under the Apache License, Version 2.0.
"""Generate with a first and/or last frame plus references and voice on Pika H3 Flex, and chain clips.

    python example.py --prompt prompt.txt --first start.png [--last end.png] \
        --reference character.png --reference place.png --voice voice.wav --out clip.mp4
    python example.py ... --chain 3 --out chain.mp4      # each clip starts on the previous clip's last frame

`--blocks` is this repository (a Hub id or a local folder); the model is Pika-Labs/Pika-H3-Flex at a pinned revision.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

# The Pika-Labs/Pika-H3-Flex snapshot these blocks were measured on; a later upload by Pika cannot change results.
PIKA_REVISION = "3ff166447d5edb1406107ab8370c42000ce173e4"

PIKA_REPO = "Pika-Labs/Pika-H3-Flex"
BUCKETS = {"480p": (480, 832), "768p": (768, 1344)}  # portrait (width, height); swap for landscape
AUDIO_REFERENCE_SECONDS = 14.9  # Pika's ref2va setup refuses more than 15 s of voice references in total


def load_pipeline(blocks_repo, pika_revision, resident=True, snapshot=None):
    """`snapshot` = a complete local copy of Pika-Labs/Pika-H3-Flex at `pika_revision` (offline use)."""
    from diffusers import ComponentsManager, ModularPipelineBlocks

    source = str(snapshot) if snapshot else PIKA_REPO
    revision = None if snapshot else pika_revision
    blocks = ModularPipelineBlocks.from_pretrained(blocks_repo, trust_remote_code=True)
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda:0")
    pipe = blocks.init_pipeline(source, components_manager=manager)
    pipe.load_components(
        pretrained_model_name_or_path=source,
        revision=revision,
        dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
    )
    pipe.load_lora_weights(source, revision=revision, weight_name="ref2va/adapter.safetensors")
    if not resident:
        pipe.transformer_ref.enable_group_offload(
            onload_device=torch.device("cuda:0"), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=False,
        )
    return pipe


def is_image(path):
    """By content, not extension: references are often stored without one."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def generate(pipe, prompt, references, first=None, last=None, size="480p", seconds=15, seed=0):
    from diffusers.modular_pipelines.pika_h3_flex import PikaH3FlexAudioReference, PikaH3FlexImageReference

    refs = [PikaH3FlexImageReference.from_file(str(p)) if is_image(p) else PikaH3FlexAudioReference.from_file(str(p))
            for p in references]
    # Pika's setup caps voice references at 15 s in total; give each its share rather than refuse the request.
    voices = [r for r in refs if isinstance(r, PikaH3FlexAudioReference)]
    for voice in voices:
        rate = voice.sample_rate or 48000
        voice.audio = voice.audio[..., : int(rate * AUDIO_REFERENCE_SECONDS / len(voices))]
    width, height = BUCKETS[size]
    return pipe(
        prompt=prompt, references=refs, image=first, last_image=last, width=width, height=height,
        num_frames=seconds * 24, generator=torch.Generator("cuda:0").manual_seed(seed),
        output=["videos", "audio", "sampling_rate"],
    )


def chain(pipe, prompts, references, first, size="480p", seconds=15, seed=0):
    """One clip per prompt; each starts exactly on the last frame the clip before it ended on."""
    clips = []
    for index, prompt in enumerate(prompts):
        result = generate(pipe, prompt, references, first=first, size=size, seconds=seconds, seed=seed + index)
        clips.append(result)
        first = result["videos"][0][-1]
    return clips


def main():
    from diffusers.utils.export_utils import encode_video

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default=str(Path(__file__).parent))
    parser.add_argument("--pika-revision", default=PIKA_REVISION, help="Pika-Labs/Pika-H3-Flex commit (default: the tested one)")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--voice", type=Path, action="append", default=[])
    parser.add_argument("--size", choices=BUCKETS, default="480p")
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.first or args.last):
        parser.error("give --first, --last or both (without either, use Pika's own ref2va workflow)")

    pipe = load_pipeline(args.blocks, args.pika_revision)
    prompt = args.prompt.read_text().strip()
    first = Image.open(args.first).convert("RGB") if args.first else None
    last = Image.open(args.last).convert("RGB") if args.last else None
    references = [*args.reference, *args.voice]
    if args.chain > 1:
        results = chain(pipe, [prompt] * args.chain, references, first, args.size, args.seconds, args.seed)
    else:
        results = [generate(pipe, prompt, references, first, last, args.size, args.seconds, args.seed)]
    for index, result in enumerate(results):
        out = args.out if len(results) == 1 else args.out.with_name(f"{args.out.stem}_{index}{args.out.suffix}")
        encode_video(result["videos"][0], fps=24, output_path=str(out), audio=result["audio"][0].float().cpu(),
                     audio_sample_rate=result["sampling_rate"])
        print(out)


if __name__ == "__main__":
    main()
