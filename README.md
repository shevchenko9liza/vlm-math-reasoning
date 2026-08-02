# VLM Math Reasoning

A vision-language pipeline for visual math QA — image + question → answer — built from scratch (custom vision-to-text adapter, prompt/label masking, robust benchmark parsing) and evaluated on **MathVista**: Qwen2.5-VL-3B score-mode reaches **0.717 on 540 multiple-choice problems**, with a blank-image ablation proving **+20.7 pp comes from actually seeing the image**.

[Русская версия](README.ru.md)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![MathVista](https://img.shields.io/badge/MathVista-0.717%20(n%3D540)-success)
![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-green)

## What's inside

**A minimal VLM implemented end-to-end** (`hw/`):
- `dataset.py` — manifest-driven loading, split filtering, RGB image handling.
- `processor.py` — image resizing/tiling, prompt construction with `<image_start>/<image>/<image_end>` tokens, tokenization with **prompt-token masking in labels**, batch collation.
- `model.py` — **`VisionToTextAdapter`**: trainable queries + attention pooling + MLP projection; `merge_visual_embeddings` inserts visual embeddings exactly at `<image>` token positions; `MathVLM` with frozen backbones.
- `train.py` — training loop with loss-finiteness checks and checkpointing.
- `benchmark.py` — accuracy overall and per subject, with a **robust multiple-choice parser** (`"A"`, `"(B)"`, `"Answer: C"`, `"The correct answer is D."`).

**Adapter-only training on Apple Silicon** (`scripts/`): CLIP ViT-B/32 + Qwen2.5-1.5B-Instruct, both frozen — only a **3.56M-parameter adapter (0.218% of 1.63B)** trains, in 74 seconds on MPS. Honest result: loss fell 1.025 → 0.614, but dev accuracy did not beat the random-adapter baseline on the small train set — analysed rather than hidden (small data + strong LLM text priors + insufficient alignment).

**Real-VLM MathVista evaluation** (`scripts/eval_real_vlm_mathvista.py`): options are scored by **next-token log-probability** instead of fragile text parsing — `invalid_prediction_count = 0` across all runs. MPS-friendly (`--max-image-side 1024` guards against OOM).

## Results

| Run | Dataset | n | Accuracy |
|---|---|---|---|
| **Qwen2.5-VL-3B, score-mode** | MathVista testmini MC | **540** | **0.7167** |
| Same rows, blank-image ablation | MathVista testmini MC | 540 | 0.5093 |
| **Visual contribution** | — | — | **+20.7 pp** |
| Trained CLIP+Qwen adapter | strict A–D subset | 22 | 0.500 |
| Pipeline-format baseline | MathVista MC | 50 | 0.260 |

The blank-image ablation is the key control: it shows the score is not explained by text priors or option distribution — the model genuinely uses the image.

## Getting started
```bash
pip install -e ".[ml]"
python -m pytest              # 14 public tests
python -m hw.train --config configs/track_a_cpu.yaml --fast-train
python scripts/eval_real_vlm_mathvista.py --help
```
See `MATHVISTA_EVAL.md` and `DATA_SOURCES.md` for dataset preparation; `report.md` for the full experiment log and error analysis.

---
**Keywords:** vision-language model, VLM, multimodal, visual question answering, math reasoning, MathVista, adapter training, CLIP, Qwen, log-probability scoring

**Ключевые слова:** мультимодальные модели, VLM, визуальный вопрос-ответ, математическое рассуждение, MathVista, адаптеры, CLIP, Qwen
