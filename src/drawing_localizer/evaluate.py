"""Phrase-level recall evaluation for the three seed drawings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.fuzz import ratio


@dataclass(frozen=True)
class GoldOccurrence:
    """One expected visible occurrence from the manually reviewed gold set."""

    key: str
    text: str


@dataclass(frozen=True)
class DetectionCandidate:
    """One detected phrase, optionally reconstructed from wrapped fragments."""

    text: str
    region_ids: frozenset[int]


def evaluate_result(result_path: Path, gold_path: Path, threshold: int = 86) -> dict:
    """Compute one-to-one phrase recall against a manually reviewed gold list.

    Each detection can satisfy at most one gold occurrence. This prevents short
    labels such as ``Лит.`` from being counted multiple times through fuzzy
    partial matches against unrelated strings such as ``Лист``.
    """
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    candidates = _build_candidates(result.get("regions", []))
    occurrences = _expand_gold(gold["expected_texts"])

    edges: list[tuple[float, int, int]] = []
    for gold_index, occurrence in enumerate(occurrences):
        target = _normalize(occurrence.text)
        required_score = _required_score(target, threshold)
        for candidate_index, candidate in enumerate(candidates):
            score = _match_score(target, _normalize(candidate.text))
            if score >= required_score:
                edges.append((score, gold_index, candidate_index))

    # Greedy maximum-score one-to-one assignment is deterministic and adequate
    # for this small benchmark because exact matches dominate the edge list.
    edges.sort(reverse=True)
    matched_gold: set[int] = set()
    used_region_ids: set[int] = set()
    assignments: dict[int, tuple[int, float]] = {}

    for score, gold_index, candidate_index in edges:
        if gold_index in matched_gold:
            continue
        candidate = candidates[candidate_index]
        if candidate.region_ids & used_region_ids:
            continue
        matched_gold.add(gold_index)
        used_region_ids.update(candidate.region_ids)
        assignments[gold_index] = (candidate_index, score)

    missed = [
        occurrence.text
        for index, occurrence in enumerate(occurrences)
        if index not in matched_gold
    ]
    details = []
    for index, occurrence in enumerate(occurrences):
        assignment = assignments.get(index)
        details.append(
            {
                "text": occurrence.text,
                "matched": assignment is not None,
                "matched_detection": (
                    candidates[assignment[0]].text if assignment is not None else None
                ),
                "score": assignment[1] if assignment is not None else 0,
            }
        )

    total = len(occurrences)
    matched = len(matched_gold)
    return {
        "image": result.get("image"),
        "strategy": result.get("strategy"),
        "model": result.get("model"),
        "api_calls": result.get("api_calls"),
        "gold_occurrences": total,
        "matched_occurrences": matched,
        "recall": matched / total if total else 1.0,
        "missed": missed,
        "details": details,
    }


def _expand_gold(items: list[dict]) -> list[GoldOccurrence]:
    occurrences: list[GoldOccurrence] = []
    for item in items:
        count = item.get("occurrences", 1)
        for occurrence_index in range(count):
            occurrences.append(
                GoldOccurrence(
                    key=f"{item['text']}#{occurrence_index + 1}",
                    text=item["text"],
                )
            )
    return occurrences


def _build_candidates(regions: list[dict]) -> list[DetectionCandidate]:
    candidates = [
        DetectionCandidate(
            text=region["source_text"],
            region_ids=frozenset({int(region.get("id", index + 1))}),
        )
        for index, region in enumerate(regions)
    ]

    # Add virtual candidates for words wrapped with a visual line-break hyphen.
    # The original regions remain available, while the virtual candidate consumes
    # both region IDs during one-to-one assignment.
    for left in regions:
        left_text = left.get("source_text", "").strip()
        if not left_text.endswith("-"):
            continue
        left_box = left.get("box_pixels")
        if not left_box:
            continue
        for right in regions:
            if left is right:
                continue
            right_box = right.get("box_pixels")
            if not right_box or not _is_next_wrapped_line(left_box, right_box):
                continue
            candidates.append(
                DetectionCandidate(
                    text=left_text[:-1] + right.get("source_text", "").lstrip(),
                    region_ids=frozenset(
                        {
                            int(left.get("id", 0)),
                            int(right.get("id", 0)),
                        }
                    ),
                )
            )
    return candidates


def _is_next_wrapped_line(left: list[int], right: list[int]) -> bool:
    left_y1, left_x1, left_y2, left_x2 = left
    right_y1, right_x1, right_y2, right_x2 = right
    vertical_gap = right_y1 - left_y2
    max_height = max(left_y2 - left_y1, right_y2 - right_y1, 1)
    overlap = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
    min_width = max(1, min(left_x2 - left_x1, right_x2 - right_x1))
    return -2 <= vertical_gap <= max(8, round(max_height * 0.75)) and overlap / min_width >= 0.65


def _required_score(target: str, default_threshold: int) -> int:
    compact_length = len(re.sub(r"[^a-zа-я0-9]", "", target))
    if compact_length <= 5:
        return max(default_threshold, 96)
    if compact_length <= 8:
        return max(default_threshold, 92)
    return default_threshold


def _match_score(target: str, detected: str) -> float:
    if target == detected:
        return 100.0
    score = float(ratio(target, detected))

    # Numbered notes may include a leading item number not present in gold.
    stripped = re.sub(r"^\s*\d+\s*[.)*:-]*\s*", "", detected)
    if stripped != detected:
        score = max(score, float(ratio(target, stripped)))
    return score


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("№", " no ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
