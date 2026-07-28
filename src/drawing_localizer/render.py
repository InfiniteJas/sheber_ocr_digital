"""Conservative in-place rendering for localized technical drawings."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


@dataclass(frozen=True)
class RenderStats:
    """Summary of one deterministic rendering pass."""

    total_regions: int
    rendered_regions: int
    skipped_partial: int
    skipped_invalid: int


def render_localized_image(
    image_path: Path,
    result_path: Path,
    output_path: Path,
    font_path: Path | None = None,
) -> RenderStats:
    """Erase detected Cyrillic text and render the English replacement.

    The renderer is intentionally conservative: it uses the validated Gemini
    boxes, restores straight table lines that cross a cleared box, and shrinks
    the replacement text until it fits. It never changes pure numeric regions,
    because those regions are excluded by the extraction prompt.
    """
    image = Image.open(image_path).convert("RGB")
    original = image.copy()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    regions = list(payload.get("regions", []))
    font_file = resolve_font(font_path)

    actionable: list[dict] = []
    skipped_partial = 0
    skipped_invalid = 0

    for region in regions:
        source = str(region.get("source_text", ""))
        target = str(region.get("target_text", ""))
        box = region.get("box_pixels")

        if region.get("is_partial", False):
            skipped_partial += 1
            continue
        if (
            not _CYRILLIC_RE.search(source)
            or _CYRILLIC_RE.search(target)
            or not target.strip()
            or not _valid_box(box, image.size)
        ):
            skipped_invalid += 1
            continue
        actionable.append(region)

    # Erase first and render second. This prevents a later overlapping box from
    # clearing text that has already been drawn by an earlier region.
    for region in actionable:
        _erase_region(image, original, region["box_pixels"])

    rendered = 0
    for region in actionable:
        if _draw_region(image, region, font_file):
            rendered += 1
        else:
            skipped_invalid += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    stats = RenderStats(
        total_regions=len(regions),
        rendered_regions=rendered,
        skipped_partial=skipped_partial,
        skipped_invalid=skipped_invalid,
    )
    stats_path = output_path.with_suffix(".render.json")
    stats_path.write_text(
        json.dumps(stats.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def resolve_font(font_path: Path | None = None) -> Path:
    """Resolve a readable system font without bundling font files."""
    candidates = [font_path] if font_path else []
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\calibri.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No TrueType font was found. Pass --font with a local .ttf file path."
    )


def _erase_region(image: Image.Image, original: Image.Image, box: list[int]) -> None:
    """Clear one tight text box and restore straight lines crossing it."""
    y_min, x_min, y_max, x_max = box
    padding = 1
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(image.width - 1, x_max + padding)
    y2 = min(image.height - 1, y_max + padding)

    horizontal, vertical = _detect_crossing_lines(original, (x1, y1, x2, y2))
    ImageDraw.Draw(image).rectangle((x1, y1, x2, y2), fill="white")
    draw = ImageDraw.Draw(image)
    for y in horizontal:
        draw.line((x1, y, x2, y), fill="black", width=1)
    for x in vertical:
        draw.line((x, y1, x, y2), fill="black", width=1)


def _detect_crossing_lines(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    """Detect long black horizontal/vertical runs likely belonging to tables."""
    x1, y1, x2, y2 = box
    gray = image.convert("L")
    horizontal: list[int] = []
    vertical: list[int] = []
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)

    for y in range(y1, y2 + 1):
        dark = sum(gray.getpixel((x, y)) < 96 for x in range(x1, x2 + 1))
        if dark / width >= 0.82:
            horizontal.append(y)

    for x in range(x1, x2 + 1):
        dark = sum(gray.getpixel((x, y)) < 96 for y in range(y1, y2 + 1))
        if dark / height >= 0.82:
            vertical.append(x)

    return _collapse_adjacent(horizontal), _collapse_adjacent(vertical)


def _collapse_adjacent(values: Iterable[int]) -> list[int]:
    """Represent a multi-pixel line by its center coordinate."""
    values = sorted(values)
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value <= groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def _draw_region(image: Image.Image, region: dict, font_path: Path) -> bool:
    """Render one translated label inside its original axis-aligned box."""
    y_min, x_min, y_max, x_max = region["box_pixels"]
    target = str(region["target_text"]).strip()
    rotation = int(region.get("rotation_degrees", 0))
    box_width = max(1, x_max - x_min)
    box_height = max(1, y_max - y_min)

    rendered = _fit_text_layer(
        target,
        font_path,
        box_width=box_width,
        box_height=box_height,
        rotation_degrees=rotation,
    )
    if rendered is None:
        return False

    paste_x = x_min + (box_width - rendered.width) // 2
    paste_y = y_min + (box_height - rendered.height) // 2
    image.paste(rendered, (paste_x, paste_y), rendered)
    return True


def _fit_text_layer(
    text: str,
    font_path: Path,
    *,
    box_width: int,
    box_height: int,
    rotation_degrees: int,
) -> Image.Image | None:
    """Choose the largest one- or two-line layout that fits the source box."""
    max_font_size = max(6, min(72, round(max(box_height, box_width) * 0.9)))
    layouts = _line_layouts(text)

    for font_size in range(max_font_size, 4, -1):
        font = ImageFont.truetype(str(font_path), size=font_size)
        for lines in layouts:
            layer = _render_text_layer(lines, font)
            rotated = layer.rotate(
                -rotation_degrees,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            if rotated.width <= box_width and rotated.height <= box_height:
                return rotated
    return None


def _line_layouts(text: str) -> list[list[str]]:
    """Try one line first, then balanced two-line word splits."""
    words = text.split()
    layouts = [[text]]
    if len(words) < 2:
        return layouts

    split_points = sorted(
        range(1, len(words)),
        key=lambda index: abs(
            len(" ".join(words[:index])) - len(" ".join(words[index:]))
        ),
    )
    layouts.extend(
        [" ".join(words[:index]), " ".join(words[index:])]
        for index in split_points
    )
    return layouts


def _render_text_layer(lines: list[str], font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render centered black text into a tight transparent layer."""
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(probe)
    spacing = max(0, round(font.size * 0.05))
    bbox = draw.multiline_textbbox(
        (0, 0),
        "\n".join(lines),
        font=font,
        spacing=spacing,
        align="center",
    )
    # Pillow 11 may return floating-point bounds for some fonts. Image.new
    # requires integer dimensions, so round outward to avoid clipping glyphs.
    width = max(1, math.ceil(bbox[2] - bbox[0]))
    height = max(1, math.ceil(bbox[3] - bbox[1]))
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.multiline_text(
        (-bbox[0], -bbox[1]),
        "\n".join(lines),
        font=font,
        fill="black",
        spacing=spacing,
        align="center",
    )
    return layer


def _valid_box(box: object, image_size: tuple[int, int]) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        y_min, x_min, y_max, x_max = (int(value) for value in box)
    except (TypeError, ValueError):
        return False
    width, height = image_size
    return (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    )
