# Gemini Technical Drawing Localizer

A compact, quality-first pipeline for removing Russian text from mechanical
engineering drawings and placing an English translation in the same region.

The solution intentionally avoids a large OCR stack. Gemini handles visual text
recognition, translation, transliteration, and normalized bounding boxes;
Pydantic and Pillow make the output deterministic and reproducible.

## Approach

```text
full-page Gemini Flash pass
        ↓
full-width lower-table audit
        ↓
sequence-gap detection in repeated document codes
        ↓
Gemini Pro fallback for one unresolved micro-crop
        ↓
row-aware deduplication and fragment cleanup
        ↓
line-aware text removal and fitted English rendering
```

Three extraction strategies are included for comparison:

1. **Baseline** — one full-page request.
2. **Tiled** — the baseline plus six overlapping crops.
3. **Adaptive** — the baseline, one dense lower-section audit, and a Pro
   micro-crop only when a suspicious sequence gap remains.

## Measured result

Drawing 01 was manually reviewed with 46 visible Cyrillic text occurrences.

| Strategy | Gemini calls | Matched | Recall |
|---|---:|---:|---:|
| Baseline | 1 | 44 / 46 | 95.65% |
| Fixed 2 × 3 tiles | 7 | 45 / 46 | 97.83% |
| Adaptive Flash + Pro fallback | 4 | 46 / 46 | **100%** |

The adaptive strategy reached higher recall with fewer requests than fixed
tiling. The final missing row was not inferred from numeric order: a local rule
only selected the suspicious crop, and Gemini Pro transcribed the actual pixels.

Gold lists for drawings 02 and 03 are included as seed annotations and should be
manually reviewed before their scores are presented as final benchmark results.

## Why this design

- **High recall:** full-page context plus a targeted audit of dense title blocks
  and repetitive specification rows.
- **Efficient API use:** expensive fallback is applied only to unresolved small
  regions, not to the full page.
- **Stable output:** structured JSON, Pydantic validation, bounded retries,
  request caching, deterministic coordinate conversion, and row-aware dedupe.
- **Measurable quality:** one-to-one phrase matching against manually reviewed
  gold occurrences rather than visual inspection alone.
- **Minimal architecture:** no web server, database, queue, agent framework, or
  model-training pipeline.

## Project structure

```text
.
├── data/
│   ├── input/                 # Three supplied drawings
│   └── gold/                  # Phrase-level benchmark annotations
├── src/drawing_localizer/
│   ├── pipeline.py            # Gemini calls, caching, crops, cleanup
│   ├── prompts.py             # English system/task prompts
│   ├── schemas.py             # Pydantic response contracts
│   ├── evaluate.py            # One-to-one phrase recall
│   └── render.py              # Removal and fitted English rendering
├── tests/
├── main.py
└── requirements.txt
```

## Setup

Create a Gemini API key in Google AI Studio, then keep it only in `.env`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

```env
GEMINI_API_KEY=replace_me
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODEL=gemini-3.1-pro-preview
```

## Run the final pipeline

```powershell
python main.py extract `
  --image data/input/drawing_01_support.jpg `
  --strategy adaptive

python main.py evaluate `
  --result outputs/adaptive/drawing_01_support.json `
  --gold data/gold/drawing_01_support.json

python main.py render `
  --image data/input/drawing_01_support.jpg `
  --result outputs/adaptive/drawing_01_support.json `
  --output outputs/localized/drawing_01_support.png
```

The renderer uses a local system font and does not bundle font files. Pass a
specific font when needed:

```powershell
python main.py render `
  --image data/input/drawing_01_support.jpg `
  --result outputs/adaptive/drawing_01_support.json `
  --output outputs/localized/drawing_01_support.png `
  --font C:\Windows\Fonts\arial.ttf
```

## Other benchmark commands

```powershell
python main.py extract --image data/input/drawing_01_support.jpg --strategy baseline
python main.py extract --image data/input/drawing_01_support.jpg --strategy tiled
python main.py batch --strategy adaptive
python -m unittest discover -v
```

Successful Gemini responses are cached under `outputs/cache/`. Retries count as
API calls, while cached responses are reported separately.

## Optional domain-specific OCR extension

For repeated production use, a practical extension is to fine-tune
[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) on cropped text
from engineering drawings and use it as a local recall/verification model. The
[PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga)
project demonstrates the same pattern: domain-specific supervised fine-tuning of
a general OCR-VLM with real and synthetic text crops. This is especially relevant
for on-premise processing, recurring GOST title blocks, and Kazakh/Russian text.
