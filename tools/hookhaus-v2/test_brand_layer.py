import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand_layer as b  # noqa: E402
from PIL import Image  # noqa: E402


class HookChecks(unittest.TestCase):
    def test_good_hook_has_no_problems(self):
        self.assertEqual(b.check_hook("*Head in the sand*, you lose"), [])
        self.assertEqual(b.check_hook("*$30* x 300,000 = *$100M*"), [])

    def test_flags_long_unmarked_hook(self):
        problems = b.check_hook("so basically what I wanted to say today is about retention and customers")
        self.assertTrue(any("words" in p for p in problems))
        self.assertTrue(any("highlighted" in p for p in problems))

    def test_flags_hook_without_tension(self):
        self.assertTrue(b.check_hook("*Nice* weather today"))

    def test_markup_and_wrap_keep_highlights(self):
        words = b.hook_words("*Head in the sand*, you lose")
        self.assertEqual([w for w, hl in words if hl], ["Head", "in", "the", "sand,"])
        self.assertEqual([w for w, _ in words], ["Head", "in", "the", "sand,", "you", "lose"])
        lines = b.wrap_words(words, 15)
        self.assertTrue(all(sum(len(w) for w, _ in ln) + len(ln) - 1 <= 15 for ln in lines))


class AssAndCaption(unittest.TestCase):
    def test_ass_contains_hook_and_lime_subtitles(self):
        words = [{"text": "Hello", "start": 0.2, "end": 0.5}, {"text": "world.", "start": 0.6, "end": 1.0}]
        ass = b.build_branded_ass(words, "*Head in the sand*, you lose")
        self.assertIn("Style: Hook,Inter Display Black", ass)
        self.assertIn("HEAD", ass)
        self.assertIn("Dialogue: 0,", ass)
        self.assertIn(b.ASS_LIME, ass)
        self.assertNotIn("&H00E5FF&", ass)  # old yellow is gone

    def test_caption_keeps_attribution_after_tags(self):
        cap = b.build_caption("*Head in the sand*, you lose", "Do you look?", ["business", "#founder"],
                              {"title": "T", "channel": "C", "url": "https://youtu.be/x"})
        lines = cap.split("\n")
        self.assertEqual(lines[0], "Head in the sand, you lose")
        self.assertEqual(lines[2], "#business #founder")
        self.assertIn("CC BY 3.0", lines[3])


class Visuals(unittest.TestCase):
    def test_mark_is_round_and_lime(self):
        m = b.draw_mark(200)
        self.assertEqual(m.size, (200, 200))
        self.assertEqual(m.getpixel((0, 0))[3], 0)  # transparent corner
        r, g, bl, a = m.getpixel((20, 100))
        self.assertGreater(g, 200)  # lime edge near the circle border

    def test_cover_size_and_title_fit(self):
        frame = Image.new("RGB", (1080, 1920), (40, 60, 90))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "c.jpg"
            info = b.make_cover(frame, "*$30* x 300,000 = *$100M*", out)
            with Image.open(out) as img:
                self.assertEqual(img.size, (1080, 1920))
            self.assertLessEqual(info["lines"], 4)

    def test_cover_rejects_title_that_cannot_fit(self):
        with self.assertRaises(ValueError):
            b.make_cover(Image.new("RGB", (1080, 1920)), " ".join(["SUPERCALIFRAGILISTIC"] * 12), "x.jpg")


if __name__ == "__main__":
    unittest.main()
