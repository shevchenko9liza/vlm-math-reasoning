# VLM Math Reasoning

Vision-language пайплайн для визуального математического QA — изображение + вопрос → ответ — построен с нуля (собственный vision-to-text адаптер, маскирование промпта в labels, устойчивый парсинг ответов) и провалидирован на **MathVista**: Qwen2.5-VL-3B в score-режиме даёт **0.717 на 540 multiple-choice задачах**, а абляция с пустым изображением доказывает: **+20.7 п.п. даёт именно картинка**.

[English version](README.md)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![MathVista](https://img.shields.io/badge/MathVista-0.717%20(n%3D540)-success)

## Что внутри

**Минимальная VLM, реализованная целиком** (`hw/`): датасет по манифесту; процессор с тайлингом изображений, `<image>`-токенами и маскированием промпта в labels; **`VisionToTextAdapter`** (обучаемые queries + attention-пулинг + MLP) со вставкой визуальных эмбеддингов ровно в позиции `<image>`-токенов; цикл обучения с проверками; бенчмарк с устойчивым парсером вариантов ответа.

**Adapter-only обучение на Apple Silicon** (`scripts/`): CLIP ViT-B/32 + Qwen2.5-1.5B-Instruct заморожены — обучается только **адаптер на 3.56 млн параметров (0.218% от 1.63 млрд)**, 74 секунды на MPS. Честный результат: loss упал 1.025 → 0.614, но dev-точность не превзошла random-adapter бейзлайн на маленькой выборке — это проанализировано, а не скрыто.

**Оценка реальной VLM на MathVista**: варианты скорятся **лог-вероятностью следующего токена** вместо хрупкого текстового парсинга — `invalid_prediction_count = 0` во всех прогонах.

## Результаты

| Прогон | Датасет | n | Точность |
|---|---|---|---|
| **Qwen2.5-VL-3B, score-mode** | MathVista testmini MC | **540** | **0.7167** |
| Те же строки, пустое изображение | MathVista testmini MC | 540 | 0.5093 |
| **Вклад изображения** | — | — | **+20.7 п.п.** |
| Обученный CLIP+Qwen адаптер | строгий A–D сабсет | 22 | 0.500 |

Абляция с пустым изображением — ключевой контроль: скор не объясняется текстовыми priors, модель действительно использует картинку.

## Запуск
```bash
pip install -e ".[ml]"
python -m pytest              # 14 публичных тестов
```
Подготовка данных — в `MATHVISTA_EVAL.md` и `DATA_SOURCES.md`; полный журнал экспериментов и анализ ошибок — в `report.md`.

---
**Ключевые слова:** мультимодальные модели, VLM, визуальный вопрос-ответ, математическое рассуждение, MathVista, адаптеры, CLIP, Qwen

**Keywords:** vision-language model, multimodal, visual question answering, MathVista, adapter training
