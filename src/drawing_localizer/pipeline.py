"""High-recall Gemini extraction pipeline for technical drawings."""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageOps
from rapidfuzz.fuzz import ratio

from .prompts import (
    BOTTOM_AUDIT_PROMPT,
    EXACT_ROW_OCR_PROMPT,
    FULL_PAGE_PROMPT,
    SEQUENCE_GAP_PROMPT_TEMPLATE,
    SYSTEM_INSTRUCTION,
    TILE_PROMPT_TEMPLATE,
)
from .schemas import ExtractionResult, TextRegion


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in full-image pixel coordinates."""

    y_min: int
    x_min: int
    y_max: int
    x_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class Tile:
    """A crop and its position inside the original image."""

    index: int
    row: int
    column: int
    box: PixelBox
    image: Image.Image


@dataclass(frozen=True)
class DetectedRegion:
    """A model region converted to full-image pixel coordinates."""

    source_text: str
    target_text: str
    operation: str
    box: PixelBox
    rotation_degrees: int
    text_kind: str
    confidence: float
    is_partial: bool
    source_pass: str


@dataclass(frozen=True)
class PipelineResult:
    """Output of one extraction strategy.

    ``api_calls`` is the total number of Gemini requests represented by the
    result, including cached requests and retries. ``new_api_calls`` is the
    number made by the current CLI invocation.
    """

    image_path: Path
    strategy: str
    model: str
    api_calls: int
    new_api_calls: int
    cache_hits: int
    raw_region_count: int
    regions: list[DetectedRegion]


class GeminiDrawingExtractor:
    """Small Gemini client with validation, diagnostics, and bounded retries."""

    def __init__(self, model: str | None = None, max_attempts: int = 2) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and set the key."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL", "gemini-3.1-pro-preview"
        )
        self._max_attempts = max_attempts
        self._api_calls = 0
        self._diagnostic_dir = Path(
            os.getenv("GEMINI_DIAGNOSTIC_DIR", "outputs/raw_failures")
        )

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    @property
    def fallback_model(self) -> str:
        """Return the model used only for unresolved micro-crops."""
        return self._fallback_model

    @property
    def api_calls(self) -> int:
        """Return the exact number of requests made by this client instance."""
        return self._api_calls

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_output_tokens: int = 16_384,
        model: str | None = None,
        media_resolution: types.MediaResolution | None = None,
    ) -> ExtractionResult:
        """Extract and validate Cyrillic text regions from one image.

        A truncated structured response is retried once with a larger token
        budget. Every request, including retries, is counted for benchmarking.
        Invalid raw output is saved locally to make failures reproducible.
        """
        request_model = model or self._model
        image_part = types.Part.from_bytes(
            data=_encode_jpeg(image),
            mime_type="image/jpeg",
        )
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            # Gemini 3.6 Flash supports up to 65,536 output tokens. The second
            # attempt doubles the budget only when the first response fails.
            token_budget = min(max_output_tokens * (2 ** (attempt - 1)), 65_536)
            response = None
            try:
                self._api_calls += 1
                response = self._client.models.generate_content(
                    model=request_model,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                        max_output_tokens=token_budget,
                        media_resolution=(
                            media_resolution
                            or types.MediaResolution.MEDIA_RESOLUTION_HIGH
                        ),
                    ),
                )

                response_text = response.text or ""
                finish_reason = _get_finish_reason(response)
                if finish_reason == "MAX_TOKENS":
                    raise ValueError(
                        f"Gemini response was truncated at {token_budget} output tokens"
                    )
                if not response_text:
                    raise ValueError(
                        f"Gemini returned an empty response (finish_reason={finish_reason})"
                    )

                try:
                    return ExtractionResult.model_validate_json(response_text)
                except Exception as parse_error:
                    # EOF and an absent closing brace are strong truncation signals,
                    # even when an SDK version does not expose finish_reason cleanly.
                    if not response_text.rstrip().endswith("}"):
                        raise ValueError(
                            f"Gemini returned truncated JSON at {token_budget} tokens"
                        ) from parse_error
                    raise

            except Exception as exc:  # SDK error classes vary across releases.
                last_error = exc
                self._save_failure(
                    response, prompt, token_budget, attempt, exc, request_model
                )
                if attempt < self._max_attempts:
                    time.sleep(1.5 * attempt)

        raise RuntimeError(f"Gemini extraction failed: {last_error}") from last_error

    def _save_failure(
        self,
        response: object | None,
        prompt: str,
        token_budget: int,
        attempt: int,
        error: Exception,
        model: str,
    ) -> None:
        """Write a compact local diagnostic without exposing the API key."""
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        payload = {
            "model": model,
            "attempt": attempt,
            "max_output_tokens": token_budget,
            "finish_reason": _get_finish_reason(response),
            "error_type": type(error).__name__,
            "error": str(error),
            "prompt": prompt,
            "response_text": getattr(response, "text", None) if response else None,
        }
        path = self._diagnostic_dir / f"gemini_failure_{timestamp}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def run_baseline(
    image_path: Path,
    output_dir: Path,
    extractor: GeminiDrawingExtractor,
) -> PipelineResult:
    """Run a single full-page Gemini request."""
    image = Image.open(image_path).convert("RGB")
    calls_before = extractor.api_calls
    response = extractor.extract(
        image,
        FULL_PAGE_PROMPT,
        max_output_tokens=32_768,
    )
    new_calls = extractor.api_calls - calls_before
    regions = [
        _to_global_region(
            region,
            image.size,
            PixelBox(0, 0, image.height, image.width),
            "full",
        )
        for region in response.regions
    ]
    regions = clean_redundant_fragments(deduplicate_regions(regions))

    result = PipelineResult(
        image_path=image_path,
        strategy="baseline",
        model=extractor.model,
        api_calls=new_calls,
        new_api_calls=new_calls,
        cache_hits=0,
        raw_region_count=len(response.regions),
        regions=regions,
    )
    save_result(result, image, output_dir)
    return result


def run_tiled(
    image_path: Path,
    output_dir: Path,
    extractor: GeminiDrawingExtractor,
    rows: int = 3,
    columns: int = 2,
    overlap: float = 0.15,
) -> PipelineResult:
    """Run a cached full-page pass plus overlapping high-resolution tiles.

    Successful components are checkpointed. A rerun after a transient failure
    resumes from cache instead of repeating already-paid Gemini requests.
    """
    image = Image.open(image_path).convert("RGB")
    all_regions: list[DetectedRegion] = []
    raw_region_count = 0
    total_calls = 0
    cache_hits = 0
    command_calls_before = extractor.api_calls
    cache_dir = output_dir / "cache" / image_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the previously measured baseline when available. This avoids paying
    # for an identical full-page request merely to start the tiled experiment.
    baseline_path = output_dir / "baseline" / f"{image_path.stem}.json"
    if baseline_path.exists():
        baseline_regions, baseline_raw_count, baseline_calls = _load_saved_result(
            baseline_path
        )
        all_regions.extend(baseline_regions)
        raw_region_count += baseline_raw_count
        total_calls += baseline_calls
        cache_hits += 1
    else:
        full_response, calls_used, was_cached = _extract_with_cache(
            extractor=extractor,
            image=image,
            prompt=FULL_PAGE_PROMPT,
            cache_path=cache_dir / "full.json",
            max_output_tokens=32_768,
        )
        raw_region_count += len(full_response.regions)
        total_calls += calls_used
        cache_hits += int(was_cached)
        all_regions.extend(
            _to_global_region(
                region,
                image.size,
                PixelBox(0, 0, image.height, image.width),
                "full",
            )
            for region in full_response.regions
        )

    tiles = make_tiles(image, rows=rows, columns=columns, overlap=overlap)
    for tile in tiles:
        prompt = TILE_PROMPT_TEMPLATE.format(
            tile_index=tile.index,
            tile_count=len(tiles),
            row_index=tile.row,
            column_index=tile.column,
        )
        response, calls_used, was_cached = _extract_with_cache(
            extractor=extractor,
            image=tile.image,
            prompt=prompt,
            cache_path=cache_dir / f"tile_{tile.index:02d}.json",
            max_output_tokens=16_384,
        )
        raw_region_count += len(response.regions)
        total_calls += calls_used
        cache_hits += int(was_cached)
        all_regions.extend(
            _to_global_region(region, tile.image.size, tile.box, f"tile_{tile.index}")
            for region in response.regions
        )

    regions = clean_redundant_fragments(deduplicate_regions(all_regions))
    result = PipelineResult(
        image_path=image_path,
        strategy="tiled",
        model=extractor.model,
        api_calls=total_calls,
        new_api_calls=extractor.api_calls - command_calls_before,
        cache_hits=cache_hits,
        raw_region_count=raw_region_count,
        regions=regions,
    )
    save_result(result, image, output_dir)
    return result



def run_adaptive(
    image_path: Path,
    output_dir: Path,
    extractor: GeminiDrawingExtractor,
    max_gap_audits: int = 2,
) -> PipelineResult:
    """Run a compact domain-aware audit after the full-page baseline.

    The strategy spends one request on the dense lower drawing area, where GOST
    title blocks and parts tables usually contain the smallest text. It then
    opens a tiny focused crop only when detected document-code rows reveal a
    suspicious numeric gap such as ``...001`` followed by ``...003``.

    This is deliberately simpler and cheaper than scanning six fixed tiles.
    """
    image = Image.open(image_path).convert("RGB")
    all_regions: list[DetectedRegion] = []
    raw_region_count = 0
    total_calls = 0
    cache_hits = 0
    command_calls_before = extractor.api_calls
    cache_dir = output_dir / "cache" / image_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "baseline" / f"{image_path.stem}.json"
    if baseline_path.exists():
        baseline_regions, baseline_raw_count, baseline_calls = _load_saved_result(
            baseline_path
        )
        all_regions.extend(baseline_regions)
        raw_region_count += baseline_raw_count
        total_calls += baseline_calls
        cache_hits += 1
    else:
        full_response, calls_used, was_cached = _extract_with_cache(
            extractor=extractor,
            image=image,
            prompt=FULL_PAGE_PROMPT,
            cache_path=cache_dir / "full.json",
            max_output_tokens=32_768,
        )
        raw_region_count += len(full_response.regions)
        total_calls += calls_used
        cache_hits += int(was_cached)
        all_regions.extend(
            _to_global_region(
                region,
                image.size,
                PixelBox(0, 0, image.height, image.width),
                "full",
            )
            for region in full_response.regions
        )

    # A single full-width lower crop preserves table context and avoids the
    # horizontal split that can hide a repetitive middle row in fixed tiling.
    lower_box = _lower_audit_box(image)
    lower_image = image.crop(
        (lower_box.x_min, lower_box.y_min, lower_box.x_max, lower_box.y_max)
    )
    lower_response, calls_used, was_cached = _extract_with_cache(
        extractor=extractor,
        image=lower_image,
        prompt=BOTTOM_AUDIT_PROMPT,
        cache_path=cache_dir / "adaptive_bottom_v1.json",
        max_output_tokens=24_576,
    )
    raw_region_count += len(lower_response.regions)
    total_calls += calls_used
    cache_hits += int(was_cached)
    all_regions.extend(
        _to_global_region(region, lower_image.size, lower_box, "adaptive_bottom")
        for region in lower_response.regions
    )

    merged = clean_redundant_fragments(deduplicate_regions(all_regions))
    gap_audits = _find_sequence_gap_audits(merged, image.size)[:max_gap_audits]

    fallback_used = False
    for audit_index, (audit_box, row_box, upper_text, lower_text) in enumerate(
        gap_audits, start=1
    ):
        audit_image = image.crop(
            (audit_box.x_min, audit_box.y_min, audit_box.x_max, audit_box.y_max)
        )
        prompt = SEQUENCE_GAP_PROMPT_TEMPLATE.format(
            upper_text=upper_text,
            lower_text=lower_text,
        )
        response, calls_used, was_cached = _extract_with_cache(
            extractor=extractor,
            image=audit_image,
            prompt=prompt,
            cache_path=cache_dir / f"adaptive_gap_v1_{audit_index:02d}.json",
            max_output_tokens=8_192,
        )
        raw_region_count += len(response.regions)
        total_calls += calls_used
        cache_hits += int(was_cached)
        all_regions.extend(
            _to_global_region(
                region,
                audit_image.size,
                audit_box,
                f"adaptive_gap_{audit_index}",
            )
            for region in response.regions
        )

        # Escalate only if the numeric gap remains after the Flash audit. The
        # fallback sees a single tightly cropped row, so it cannot collapse the
        # row with visually similar neighbors.
        merged_after_flash = clean_redundant_fragments(
            deduplicate_regions(all_regions)
        )
        unresolved = _find_sequence_gap_audits(merged_after_flash, image.size)
        if any(item[2] == upper_text and item[3] == lower_text for item in unresolved):
            row_image = image.crop(
                (row_box.x_min, row_box.y_min, row_box.x_max, row_box.y_max)
            )
            row_image = _prepare_micro_ocr_crop(row_image)
            pro_response, calls_used, was_cached = _extract_with_cache(
                extractor=extractor,
                image=row_image,
                prompt=EXACT_ROW_OCR_PROMPT,
                cache_path=cache_dir / f"adaptive_gap_pro_v1_{audit_index:02d}.json",
                max_output_tokens=2_048,
                model=extractor.fallback_model,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            )
            raw_region_count += len(pro_response.regions)
            total_calls += calls_used
            cache_hits += int(was_cached)
            fallback_used = True
            all_regions.extend(
                _to_global_region(
                    region,
                    row_image.size,
                    row_box,
                    f"adaptive_pro_gap_{audit_index}",
                )
                for region in pro_response.regions
            )

    regions = clean_redundant_fragments(deduplicate_regions(all_regions))
    result = PipelineResult(
        image_path=image_path,
        strategy="adaptive",
        model=(
            f"{extractor.model} + {extractor.fallback_model}"
            if fallback_used
            else extractor.model
        ),
        api_calls=total_calls,
        new_api_calls=extractor.api_calls - command_calls_before,
        cache_hits=cache_hits,
        raw_region_count=raw_region_count,
        regions=regions,
    )
    save_result(result, image, output_dir)
    return result

def make_tiles(
    image: Image.Image,
    rows: int,
    columns: int,
    overlap: float,
) -> list[Tile]:
    """Split an image into an overlapping grid with deterministic geometry."""
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if not 0.0 <= overlap < 0.5:
        raise ValueError("overlap must be in [0.0, 0.5)")

    width, height = image.size
    base_width = math.ceil(width / columns)
    base_height = math.ceil(height / rows)
    pad_x = round(base_width * overlap)
    pad_y = round(base_height * overlap)

    tiles: list[Tile] = []
    index = 1
    for row in range(rows):
        for column in range(columns):
            x_min = max(0, column * base_width - pad_x)
            y_min = max(0, row * base_height - pad_y)
            x_max = min(width, (column + 1) * base_width + pad_x)
            y_max = min(height, (row + 1) * base_height + pad_y)
            box = PixelBox(y_min, x_min, y_max, x_max)
            crop = image.crop((x_min, y_min, x_max, y_max))
            tiles.append(Tile(index=index, row=row, column=column, box=box, image=crop))
            index += 1

    return tiles


def deduplicate_regions(regions: Iterable[DetectedRegion]) -> list[DetectedRegion]:
    """Merge overlapping detections using geometry and text similarity.

    The model confidence is used only as a tie-breaker. A complete non-partial
    tile detection is preferred over a partial or shorter duplicate.
    """
    ordered = sorted(
        regions,
        key=lambda item: (
            item.is_partial,
            -item.confidence,
            -len(_normalize_text(item.source_text)),
        ),
    )
    kept: list[DetectedRegion] = []

    for candidate in ordered:
        duplicate_index = _find_duplicate(candidate, kept)
        if duplicate_index is None:
            kept.append(candidate)
            continue

        current = kept[duplicate_index]
        if _region_quality(candidate) > _region_quality(current):
            kept[duplicate_index] = candidate

    return sorted(kept, key=lambda item: (item.box.y_min, item.box.x_min))



def clean_redundant_fragments(
    regions: Iterable[DetectedRegion],
) -> list[DetectedRegion]:
    """Remove crop-edge fragments already covered by a complete detection.

    Fixed overlapping crops improve recall but also create short pieces such as
    ``ГОСТ 52`` or ``Цил``. Keeping them would inflate the region count and later
    erase the same text twice. A partial region is removed when most of its box
    is covered by any complete region. A non-partial substring is removed only
    when it overlaps a longer complete reading of the same word.
    """
    items = list(regions)
    complete = [item for item in items if not item.is_partial]
    kept: list[DetectedRegion] = []

    for candidate in items:
        redundant = False
        candidate_text = _normalize_text(candidate.source_text).replace("-", "")
        for reference in complete:
            if candidate is reference:
                continue
            coverage = _intersection_area(candidate.box, reference.box) / max(
                candidate.box.area, 1
            )
            if candidate.is_partial and coverage >= 0.65:
                redundant = True
                break

            reference_text = _normalize_text(reference.source_text).replace("-", "")
            if (
                not candidate.is_partial
                and len(candidate_text) < len(reference_text)
                and candidate_text
                and candidate_text in reference_text
                and coverage >= 0.55
            ):
                redundant = True
                break

        if not redundant:
            kept.append(candidate)

    return sorted(kept, key=lambda item: (item.box.y_min, item.box.x_min))


def _lower_audit_box(image: Image.Image) -> PixelBox:
    """Return the dense lower drawing area with a small border margin."""
    width, height = image.size
    return PixelBox(
        y_min=round(height * 0.52),
        x_min=round(width * 0.03),
        y_max=height,
        x_max=round(width * 0.98),
    )


def _find_sequence_gap_audits(
    regions: list[DetectedRegion],
    image_size: tuple[int, int],
) -> list[tuple[PixelBox, PixelBox, str, str]]:
    """Find suspicious gaps between vertically adjacent document-code rows.

    The numeric pattern is used only to decide where to look. The recovery
    prompt explicitly forbids inferring the missing value and asks Gemini to
    transcribe the crop from pixels.
    """
    width, height = image_size
    parsed: list[tuple[DetectedRegion, str, int, str]] = []
    pattern = re.compile(r"^(.*?)(\d{3})(\D*)$")

    for region in regions:
        if region.is_partial:
            continue
        if region.text_kind not in {"document_code", "table_value"}:
            continue
        text = _normalize_text(region.source_text)
        if text.count(".") < 2:
            continue
        match = pattern.match(text)
        if not match:
            continue
        parsed.append((region, match.group(1), int(match.group(2)), match.group(3)))

    audits: list[tuple[PixelBox, PixelBox, str, str]] = []
    for left_index, (upper, prefix, number, suffix) in enumerate(parsed):
        for lower, lower_prefix, lower_number, lower_suffix in parsed[left_index + 1 :]:
            if prefix != lower_prefix or suffix != lower_suffix:
                continue
            if lower_number - number != 2:
                continue
            if lower.box.y_min <= upper.box.y_min:
                continue
            horizontal_overlap = max(
                0,
                min(upper.box.x_max, lower.box.x_max)
                - max(upper.box.x_min, lower.box.x_min),
            )
            min_width = max(1, min(upper.box.width, lower.box.width))
            if horizontal_overlap / min_width < 0.7:
                continue

            audit_box = PixelBox(
                y_min=max(0, upper.box.y_min - 25),
                x_min=max(0, min(upper.box.x_min, lower.box.x_min) - 45),
                y_max=min(height, lower.box.y_max + 25),
                x_max=min(width, max(upper.box.x_max, lower.box.x_max) + 45),
            )
            row_box = PixelBox(
                y_min=max(0, upper.box.y_max - 2),
                x_min=max(0, min(upper.box.x_min, lower.box.x_min) - 30),
                y_max=min(height, lower.box.y_min + 2),
                x_max=min(width, max(upper.box.x_max, lower.box.x_max) + 30),
            )
            audits.append(
                (audit_box, row_box, upper.source_text, lower.source_text)
            )

    # Largest vertical gaps are inspected first; duplicates are removed by box.
    unique: list[tuple[PixelBox, PixelBox, str, str]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for audit in sorted(
        audits,
        key=lambda item: item[0].height,
        reverse=True,
    ):
        key = (
            audit[0].y_min,
            audit[0].x_min,
            audit[0].y_max,
            audit[0].x_max,
        )
        if key not in seen:
            unique.append(audit)
            seen.add(key)
    return unique


def _prepare_micro_ocr_crop(image: Image.Image, scale: int = 4) -> Image.Image:
    """Upscale and normalize a tiny row before the fallback OCR request."""
    normalized = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    return normalized.resize(
        (max(1, normalized.width * scale), max(1, normalized.height * scale)),
        Image.Resampling.LANCZOS,
    )


def _intersection_area(left: PixelBox, right: PixelBox) -> int:
    y_min = max(left.y_min, right.y_min)
    x_min = max(left.x_min, right.x_min)
    y_max = min(left.y_max, right.y_max)
    x_max = min(left.x_max, right.x_max)
    return max(0, y_max - y_min) * max(0, x_max - x_min)

def save_result(result: PipelineResult, image: Image.Image, output_dir: Path) -> None:
    """Persist JSON output and a numbered debug overlay."""
    strategy_dir = output_dir / result.strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    stem = result.image_path.stem

    payload = {
        "image": str(result.image_path),
        "strategy": result.strategy,
        "model": result.model,
        "api_calls": result.api_calls,
        "new_api_calls": result.new_api_calls,
        "cache_hits": result.cache_hits,
        "raw_region_count": result.raw_region_count,
        "unique_region_count": len(result.regions),
        "regions": [
            {
                "id": index,
                "source_text": region.source_text,
                "target_text": region.target_text,
                "operation": region.operation,
                "box_pixels": [
                    region.box.y_min,
                    region.box.x_min,
                    region.box.y_max,
                    region.box.x_max,
                ],
                "rotation_degrees": region.rotation_degrees,
                "text_kind": region.text_kind,
                "confidence": region.confidence,
                "is_partial": region.is_partial,
                "source_pass": region.source_pass,
            }
            for index, region in enumerate(result.regions, start=1)
        ],
    }
    (strategy_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(result.regions, start=1):
        draw.rectangle(
            (region.box.x_min, region.box.y_min, region.box.x_max, region.box.y_max),
            outline="black",
            width=2,
        )
        draw.rectangle(
            (region.box.x_min, region.box.y_min, region.box.x_min + 30, region.box.y_min + 18),
            fill="white",
            outline="black",
        )
        draw.text((region.box.x_min + 3, region.box.y_min + 2), str(index), fill="black")
    overlay.save(strategy_dir / f"{stem}_boxes.jpg", quality=95)


def _extract_with_cache(
    *,
    extractor: GeminiDrawingExtractor,
    image: Image.Image,
    prompt: str,
    cache_path: Path,
    max_output_tokens: int,
    model: str | None = None,
    media_resolution: types.MediaResolution | None = None,
) -> tuple[ExtractionResult, int, bool]:
    """Load a validated request checkpoint or create it atomically."""
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        result = ExtractionResult.model_validate(payload["result"])
        return result, int(payload.get("api_calls_used", 1)), True

    calls_before = extractor.api_calls
    result = extractor.extract(
        image,
        prompt,
        max_output_tokens=max_output_tokens,
        model=model,
        media_resolution=media_resolution,
    )
    calls_used = extractor.api_calls - calls_before
    payload = {
        "model": model or extractor.model,
        "api_calls_used": calls_used,
        "result": result.model_dump(mode="json"),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(cache_path)
    return result, calls_used, False


def _load_saved_result(path: Path) -> tuple[list[DetectedRegion], int, int]:
    """Load global pixel regions from a previously saved pipeline result."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for item in payload.get("regions", []):
        y_min, x_min, y_max, x_max = item["box_pixels"]
        regions.append(
            DetectedRegion(
                source_text=item["source_text"],
                target_text=item["target_text"],
                operation=item["operation"],
                box=PixelBox(y_min, x_min, y_max, x_max),
                rotation_degrees=item["rotation_degrees"],
                text_kind=item["text_kind"],
                confidence=item["confidence"],
                is_partial=item["is_partial"],
                source_pass=item.get("source_pass", "full"),
            )
        )
    return (
        regions,
        int(payload.get("raw_region_count", len(regions))),
        int(payload.get("api_calls", 1)),
    )


def _get_finish_reason(response: object | None) -> str | None:
    """Return a version-tolerant finish reason such as ``MAX_TOKENS``."""
    if response is None:
        return None
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    if name:
        return str(name)
    value = getattr(reason, "value", None)
    if value:
        return str(value).split(".")[-1]
    return str(reason).split(".")[-1]


def _to_global_region(
    region: TextRegion,
    local_size: tuple[int, int],
    global_crop: PixelBox,
    source_pass: str,
) -> DetectedRegion:
    """Convert a normalized crop-local box into full-image pixels.

    Gemini coordinates are normalized to the image that was sent to the API.
    A crop may be resized before inference (for example, a 4x micro-OCR crop),
    but resizing does not change its location in the original drawing. Therefore
    normalized coordinates must be projected onto ``global_crop`` dimensions,
    not onto the processed image dimensions.
    """
    del local_size  # Geometry is defined by global_crop, even after preprocessing.
    y_min, x_min, y_max, x_max = region.box_2d

    crop_height = global_crop.height
    crop_width = global_crop.width
    global_box = PixelBox(
        y_min=global_crop.y_min + round(y_min / 1000 * crop_height),
        x_min=global_crop.x_min + round(x_min / 1000 * crop_width),
        y_max=global_crop.y_min + round(y_max / 1000 * crop_height),
        x_max=global_crop.x_min + round(x_max / 1000 * crop_width),
    )
    global_box = PixelBox(
        y_min=max(global_crop.y_min, min(global_box.y_min, global_crop.y_max)),
        x_min=max(global_crop.x_min, min(global_box.x_min, global_crop.x_max)),
        y_max=max(global_crop.y_min, min(global_box.y_max, global_crop.y_max)),
        x_max=max(global_crop.x_min, min(global_box.x_max, global_crop.x_max)),
    )
    return DetectedRegion(
        source_text=region.source_text.strip(),
        target_text=region.target_text.strip(),
        operation=region.operation,
        box=global_box,
        rotation_degrees=region.rotation_degrees,
        text_kind=region.text_kind,
        confidence=region.confidence,
        is_partial=region.is_partial,
        source_pass=source_pass,
    )


def _find_duplicate(
    candidate: DetectedRegion,
    kept: list[DetectedRegion],
) -> int | None:
    """Return the matching region index without collapsing adjacent table rows.

    Similar document codes often differ by one digit and are placed in vertically
    adjacent rows. A single scalar center-distance check is unsafe for such long,
    thin boxes: their large width makes separate rows appear artificially close.
    Duplicate matching therefore requires substantial overlap on both axes when
    IoU alone is inconclusive.
    """
    candidate_text = _normalize_text(candidate.source_text)
    for index, current in enumerate(kept):
        text_score = ratio(candidate_text, _normalize_text(current.source_text))
        overlap = _intersection_over_union(candidate.box, current.box)

        if overlap >= 0.45 and text_score >= 65:
            return index

        x_overlap = _axis_overlap_ratio(
            candidate.box.x_min,
            candidate.box.x_max,
            current.box.x_min,
            current.box.x_max,
        )
        y_overlap = _axis_overlap_ratio(
            candidate.box.y_min,
            candidate.box.y_max,
            current.box.y_min,
            current.box.y_max,
        )
        if text_score >= 88 and x_overlap >= 0.65 and y_overlap >= 0.55:
            return index
    return None


def _axis_overlap_ratio(
    left_min: int,
    left_max: int,
    right_min: int,
    right_max: int,
) -> float:
    """Return overlap divided by the shorter interval length."""
    intersection = max(0, min(left_max, right_max) - max(left_min, right_min))
    shorter = max(1, min(left_max - left_min, right_max - right_min))
    return intersection / shorter


def _region_quality(region: DetectedRegion) -> tuple[int, float, int, int]:
    """Return a deterministic preference score for duplicate resolution."""
    return (
        int(not region.is_partial),
        region.confidence,
        len(_normalize_text(region.source_text)),
        int(region.source_pass.startswith("tile_")),
    )


def _intersection_over_union(left: PixelBox, right: PixelBox) -> float:
    y_min = max(left.y_min, right.y_min)
    x_min = max(left.x_min, right.x_min)
    y_max = min(left.y_max, right.y_max)
    x_max = min(left.x_max, right.x_max)
    intersection = max(0, y_max - y_min) * max(0, x_max - x_min)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def _normalized_center_distance(left: PixelBox, right: PixelBox) -> float:
    left_center = ((left.x_min + left.x_max) / 2, (left.y_min + left.y_max) / 2)
    right_center = ((right.x_min + right.x_max) / 2, (right.y_min + right.y_max) / 2)
    distance = math.dist(left_center, right_center)
    scale = max(left.width, left.height, right.width, right.height, 1)
    return distance / scale


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def _encode_jpeg(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    return buffer.getvalue()
