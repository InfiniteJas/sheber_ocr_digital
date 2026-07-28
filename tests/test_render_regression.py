"""Regression checks for Pillow text layout compatibility."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import ImageFont

from drawing_localizer.render import _line_layouts, _render_text_layer, resolve_font


class RenderRegressionTest(unittest.TestCase):
    """Protect the renderer from float bounds and nested line layouts."""

    def test_two_line_layout_contains_strings(self) -> None:
        layouts = _line_layouts("Reference dimensions")
        self.assertEqual(layouts[0], ["Reference dimensions"])
        self.assertTrue(all(isinstance(line, str) for line in layouts[1]))

    def test_text_layer_uses_integer_dimensions(self) -> None:
        font = ImageFont.truetype(str(resolve_font()), size=17)
        layer = _render_text_layer(["Reference", "dimensions"], font)
        self.assertIsInstance(layer.width, int)
        self.assertIsInstance(layer.height, int)
        self.assertGreater(layer.width, 0)
        self.assertGreater(layer.height, 0)


if __name__ == "__main__":
    unittest.main()
