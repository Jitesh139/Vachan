# Speech-to-text benchmarks

Measured with `scripts/benchmark_stt.py` (each setup in its own process) on 11 test phrases:
5 English, 3 Hindi, 3 Hinglish, Indian-accent edge-tts voices, each about 4–6 s long.
The language is given to Whisper (no auto-detection), the app's vocabulary prompt is used,
decoding is greedy. *Refs* = phrases where the parser found the right Bible reference
(parser as of 2026-10-13; it has learned more Whisper spellings since).

## Machine

Intel Core Ultra (Family 6 Model 170, 16 physical / 22 logical cores), Arc integrated GPU,
Intel NPU, 16.5 GB RAM, Windows 11, Balanced power mode.

| Run | Power | Free RAM | Setups |
|---|---|---|---|
| 2026-10-13 | plugged in | 4.6 GB | faster-whisper (`fw-*`) |
| 2026-10-16 | **battery 66 %** | 1.4 GB | OpenVINO (`ov-*`) |

The two runs are not under identical conditions. The OpenVINO run was on battery, so if
anything it understates OpenVINO. Repeat both, plugged in, before quoting exact ratios.

## Results (6 CPU threads)

| Setup | Load (s) | s per 5 s clip | RTF | Peak RAM (MB) | WER | Refs |
|---|---|---|---|---|---|---|
| **ov-large-v3-turbo-int8-GPU** (Arc iGPU) | 28.0 | **1.21** | 0.24 | 2366 | **10 %** | **9/11** |
| ov-large-v3-turbo-int8-CPU | 16.9 | 5.81 | 1.16 | 2028 | 10 % | 9/11 |
| fw-large-v3-turbo-int8 (CPU) | 13.5 | 6.99 | 1.40 | 1949 | 10 % | 9/11 |
| fw-distil-large-v3-int8 (CPU) | 13.5 | 6.36 | 1.27 | 1838 | 53 % | 5/11 |
| fw-medium-int8 (CPU) | 10.6 | 4.70 | 0.94 | 1805 | 13 % | 10/11 |
| ov-medium-int8-GPU | 41.7 | 2.07 | 0.41 | 2723 | 17 % | 8/11 |
| ov-medium-int8-CPU | 22.0 | 5.70 | 1.14 | 2191 | 15 % | 8/11 |
| **ov-small-int8-GPU** | 7.0 | 0.83 | 0.17 | 999 | 19 % | 8/11 |
| ov-small-int8-CPU | 11.0 | 1.67 | 0.34 | 870 | 21 % | 6/11 |
| **fw-small-int8 (CPU)** | 4.9 | 1.89 | 0.38 | 706 | 18 % | 7/11 |
| ov-base-int8-GPU | 18.4 | 0.83 | 0.17 | 1028 | 30 % | 6/11 |
| ov-base-int8-CPU | 4.1 | 0.68 | 0.14 | 394 | 26 % | 6/11 |
| fw-base-int8 (CPU) | 3.3 | 1.17 | 0.23 | 312 | 113 % | 6/11 |
| ov-distil-large-v3-int8-CPU | – | – | – | – | – | model not on the Hub under that name |

GPU load times include compiling kernels on first use. Later loads use `models/openvino_cache`.

**CPU threads (fw-small, earlier run with Chrome open):** 4 → 4.58 s, **6 → 3.21 s**, 8 → 3.71 s,
16 → 4.51 s per 5 s clip. More threads than performance cores is slower on this hybrid CPU.
So `cpu_threads` = physical cores is only a default. On this laptop 6 is best.

## NPU: not used

`ov-small-int8-NPU` with the vocabulary prompt fails with `roi_end <= max_dim`. With the language
set and no prompt it outputs `!!!!`. With auto language it errors. All three cases fail,
so profiles never pick the NPU.

## Chosen profiles (`vachan/stt/profiles.py`)

Each profile is an ordered list; the first engine that loads on the machine wins.

| Profile | NVIDIA GPU | Intel Arc iGPU | CPU |
|---|---|---|---|
| accurate | large-v3-turbo | large-v3-turbo | medium (slower than real time) |
| balanced | large-v3-turbo | small | small |
| fast | small | small | base (OpenVINO, then faster-whisper) |

`auto` = **accurate** when there is an NVIDIA GPU or an OpenVINO `GPU` device, otherwise **balanced**.
Medium is not used on the iGPU: turbo there is faster *and* more accurate.

## End-to-end delay (the number that matters)

`scripts/test_speech.py --realtime --profile auto` plays each phrase in real time through the live
pipeline. It measures the time from the end of the spoken phrase to the suggestion on the card.
Result on this laptop, **on battery**, accurate profile → large-v3-turbo on the Arc iGPU:
**1.47–2.70 s**, 14/14 phrases correct (second run, after the parser fixes; the first run gave 1.50–2.20 s).
Target ≤ 3 s. The delay includes the 0.5 s pause that ends a phrase.

`scripts/repeat_ui_test.py --runs 10` plays Psalm 91:2 in English, Hindi and Hinglish 10 times each,
in real time, through the real operator window, pressing Show on every card. Result:
**30/30 suggestions, card == output window 30/30, all Psalms 91:2.**

Not measured yet: the RTX 5060 desktop (expected to be faster: large-v3-turbo float16 on CUDA).
