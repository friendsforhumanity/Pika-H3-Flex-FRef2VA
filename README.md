# Pika H3 Flex — FRef2VA (first/last frame + Ref2VA references)

**Powered by MiniMax H3.** Pin a clip's first frame, last frame, or both, while keeping [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) Ref2VA's image, video and voice references, in one request. Code only: no new weights.

Also on Hugging Face: [FriendsforHumanity/Pika-H3-Flex-FRef2VA](https://huggingface.co/FriendsforHumanity/Pika-H3-Flex-FRef2VA).

## Why

Pika ships first/last frame (FL2VA) and references (Ref2VA) as two separate adapters, and its diffusers pipeline runs one or the other. Chaining clips needs both: each clip should start exactly where the last one ended, and keep the same faces and voices. With FL2VA alone the cast drifts; with Ref2VA alone the clips don't join.

## How it works

No merging or retraining. H3 reads text, references, pinned frames and the clip being made as one sequence. A pinned frame is a clean block placed at frame 0 or the last frame and held fixed while sampling. The Ref2VA model already follows pinned frames placed beside its references (ComfyUI does this with `MiniMaxH3AddGuide` next to `MiniMaxH3ReferenceToVideo`). These blocks build the same sequence in diffusers:

| Part of the sequence | From |
| --- | --- |
| Text, references, target audio and video | Pika's own Ref2VA layout, unchanged |
| Pinned first and/or last frame | added beside the references, at the same positions ComfyUI uses |

`tests/test_cpu.py` checks both rows on CPU in a few seconds, with no weights: without pinned frames the sequence equals Pika's, and with them it equals ComfyUI's.

## Usage in ComfyUI

Stock nodes only; three API workflows are in `comfyui/` (first, last, first + last, each with two image references and a voice):

- `UNETLoader` → the H3 Ref2VA checkpoint
- `MiniMaxH3ReferenceToVideo` with your references, plus `MiniMaxH3AddGuide` for each pinned frame (`frame_idx` 0 for the first, -1 for the last)
- `LoraLoaderModelOnly` → `pika_h3_flex_ref2va_comfy_bf16`, strength **1.0**
- Sampler **`lcm`**, `BasicGuider` (CFG 1.0)
- `ManualSigmas`: `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0` (6 steps)

Each workflow saves the clip and its last frame, ready to pin as the next clip's first frame.

## Usage in diffusers

```bash
pip install -r requirements.txt
python example.py --prompt prompt.txt --first start.png --last end.png \
    --reference character.png --reference place.png --voice voice.wav --out clip.mp4
```

Give `--first`, `--last` or both; `--chain 3` makes three clips, each starting on the previous clip's last frame. In Python, load the blocks with `ModularPipelineBlocks.from_pretrained("FriendsforHumanity/Pika-H3-Flex-FRef2VA", trust_remote_code=True)` and pass `image=` and/or `last_image=` beside `references=` (see `example.py`).

Requires diffusers at Pika's commit `d0b9b1b5`, the one these blocks were tested on, and `Pika-Labs/Pika-H3-Flex` revision `3ff16644` (both pinned). Before upgrading either, run `python tests/test_cpu.py`; if it fails, stay on the pins. Voice references may total at most 15 s (Pika's limit).

## Testing so far

On one B200, 15 s clips at 480p, four stories with two image references and a voice each. One, Maddox, comes from a 54-clip story made in the [Daydream World](https://daydreamapp.ai) app.

- Every clip landed on its given frames, first, last or both, within 2–4/255 (mean absolute difference).
- The ComfyUI version ran the whole 54-clip, 13.5-minute Maddox story with no burn, softening or darkening; every clip stayed within 2.4/255 of its pinned frame.
- About 140 s per clip in diffusers (CPU offload), 81 s in ComfyUI (everything on the GPU).

Known limits: faces from small or crowded references can drift in close-ups; motion and sound restart at each join; large changes between the first and last frame are untested; only 480p and 768p were tried.

## License and notices

This code: Apache-2.0 (see `LICENSE`). It downloads, and needs, MiniMax H3 and the Pika H3 Flex adapters, which are under the **MiniMax H3 Community License Agreement**, its Acceptable Use Policy and territorial scope. Check that licence before running them.

- MiniMax H3 © 2026 MiniMax. All Rights Reserved.
- Pika H3 Flex adapters © 2026 Mellis, Inc.
- FRef2VA blocks and workflows © 2026 Friends for Humanity.
