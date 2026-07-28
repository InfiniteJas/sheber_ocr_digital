"""Regression checks for row-aware region deduplication."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drawing_localizer.pipeline import DetectedRegion, PixelBox, deduplicate_regions


def region(text: str, y_min: int, y_max: int) -> DetectedRegion:
    """Create a compact table-row fixture for deduplication tests."""
    return DetectedRegion(
        source_text=text,
        target_text=text,
        operation="transliterate",
        box=PixelBox(y_min, 184, y_max, 381),
        rotation_degrees=0,
        text_kind="table_value",
        confidence=1.0,
        is_partial=False,
        source_pass="test",
    )


class DeduplicationRegressionTest(unittest.TestCase):
    """Protect row-aware deduplication from merging neighboring codes."""

    def test_adjacent_codes_are_not_merged(self) -> None:
        rows = [
            region("ИГ 02.2407.11.002", 902, 929),
            region("ИГ 02.2407.11.003", 942, 969),
        ]
        self.assertEqual(len(deduplicate_regions(rows)), 2)

    def test_overlapping_same_code_is_merged(self) -> None:
        rows = [
            region("ИГ 02.2407.11.002", 902, 929),
            region("ИГ 02.2407.11.002", 904, 931),
        ]
        self.assertEqual(len(deduplicate_regions(rows)), 1)


if __name__ == "__main__":
    unittest.main()
