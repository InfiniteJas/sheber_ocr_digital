"""Typed contracts used by Gemini and the local pipeline."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]


class TextRegion(BaseModel):
    """A single text region detected in an image.

    Coordinates follow Gemini's object-detection convention:
    ``[y_min, x_min, y_max, x_max]`` normalized to ``0..1000``.
    """

    source_text: str = Field(
        description="Exact source text visible in the image. Keep punctuation and symbols."
    )
    target_text: str = Field(
        description="English translation or Latin transliteration with no Cyrillic letters."
    )
    operation: Literal["translate", "transliterate"] = Field(
        description="Translate ordinary language; transliterate abbreviations and document codes."
    )
    box_2d: list[NormalizedCoordinate] = Field(
        min_length=4,
        max_length=4,
        description="Tight text box as [y_min, x_min, y_max, x_max], normalized to 0..1000.",
    )
    rotation_degrees: int = Field(
        ge=-180,
        le=180,
        description="Clockwise text rotation relative to normal reading orientation.",
    )
    text_kind: Literal[
        "title",
        "table_header",
        "table_value",
        "note",
        "callout",
        "border_label",
        "document_code",
        "abbreviation",
        "other",
    ]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Visual recognition confidence, not semantic translation confidence.",
    )
    is_partial: bool = Field(
        description="True only when the crop cuts off part of the visible text."
    )

    @field_validator("target_text")
    @classmethod
    def validate_target_text(cls, value: str) -> str:
        """Reject outputs that would reinsert Cyrillic into the localized image."""
        if any("\u0400" <= char <= "\u04FF" for char in value):
            raise ValueError("target_text must not contain Cyrillic characters")
        return value

    @field_validator("box_2d")
    @classmethod
    def validate_box_order(cls, value: list[int]) -> list[int]:
        """Reject inverted or zero-area boxes early."""
        y_min, x_min, y_max, x_max = value
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box_2d must have positive width and height")
        return value


class ExtractionResult(BaseModel):
    """Structured response returned by the extraction prompt."""

    regions: list[TextRegion] = Field(default_factory=list)
