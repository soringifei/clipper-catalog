import json
import tempfile
import unittest
from pathlib import Path

import hookhaus_v2 as hh


def w(start, end, text):
    return {"start": start, "end": end, "text": text}


def sentence_words(t0, text, rate=0.4):
    out = []
    for i, tok in enumerate(text.split()):
        out.append(w(round(t0 + i * rate, 3), round(t0 + i * rate + rate * 0.9, 3), tok))
    return out


class TimelineTest(unittest.TestCase):
    def test_maps_cut_time_back_to_source(self):
        tl = {"timebase": "30/1", "v": [[
            {"start": 0, "dur": 30, "offset": 0},
            {"start": 30, "dur": 60, "offset": 90},
        ]]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.v3"
            p.write_text(json.dumps(tl))
            fps, clips = hh.load_timeline(p)
        self.assertAlmostEqual(hh.cut_to_source(0.5, fps, clips), 0.5)
        # 1.5 s in the cut is frame 45 -> second clip, 15 frames in -> source frame 105
        self.assertAlmostEqual(hh.cut_to_source(1.5, fps, clips), 3.5)
        self.assertAlmostEqual(hh.cut_to_source(99, fps, clips), 5.0)


class MomentTest(unittest.TestCase):
    def test_split_sentences_on_punctuation_and_pause(self):
        words = [w(0, .3, "Hello"), w(.4, .7, "there."), w(.8, 1, "Next"), w(3, 3.2, "after"), w(3.3, 3.5, "pause")]
        s = hh.split_sentences(words, max_gap=1.2)
        self.assertEqual([x["text"] for x in s], ["Hello there.", "Next", "after pause"])

    def test_weak_opener_is_penalised(self):
        strong, _ = hh.score_window([{"start": 0, "end": 4, "text": "Why is the universe dark?",
                                       "words": [{}] * 5}])
        weak, reasons = hh.score_window([{"start": 0, "end": 4, "text": "and then it was dark",
                                           "words": [{}] * 5}])
        self.assertGreater(strong, weak)
        self.assertIn("weak opener 'and'", reasons)

    def test_pick_moments_non_overlapping_and_bounded(self):
        words = []
        t = 0.0
        for k in range(12):
            text = ("Why does this matter to you?" if k % 4 == 0 else "and it keeps going on for a while here.")
            words += sentence_words(t, text)
            t = words[-1]["end"] + 0.3
        sents = hh.split_sentences(words)
        moments = hh.pick_moments(sents, n=3, min_dur=5, max_dur=9)
        self.assertEqual(len(moments), 3)
        for m in moments:
            self.assertGreaterEqual(m["end"] - m["start"], 5)
            self.assertLessEqual(m["end"] - m["start"], 9)
            self.assertTrue(m["text"].startswith("Why"))
        for a, b in zip(moments, moments[1:]):
            self.assertLessEqual(a["end"], b["start"])


    def test_pad_bounds_never_overlap_neighbours(self):
        ms = [{"start": 0.05, "end": 10.0}, {"start": 10.1, "end": 20.0}]
        (a0, a1), (b0, b1) = hh.pad_bounds(ms, 20.1)
        self.assertEqual((a0, b1), (0.0, 20.1))
        self.assertAlmostEqual(a1, 10.05)
        self.assertLessEqual(a1, b0)


class ReframeMathTest(unittest.TestCase):
    def test_fill_track_interpolates_and_holds_edges(self):
        self.assertEqual(hh.fill_track({2: 10.0, 4: 20.0}, 6, 0), [10, 10, 10, 15, 20, 20])
        self.assertEqual(hh.fill_track({}, 3, 7), [7, 7, 7])

    def test_camera_ignores_jitter_inside_deadzone(self):
        cam = hh.virtual_camera([100, 103, 97, 102, 99], deadzone=10)
        self.assertEqual(cam, [100] * 5)

    def test_camera_follows_large_move_with_capped_speed(self):
        cam = hh.virtual_camera([0] + [500] * 200, deadzone=10, gain=0.2, max_step=12)
        steps = [b - a for a, b in zip(cam, cam[1:])]
        self.assertLessEqual(max(steps), 12 + 1e-9)
        self.assertAlmostEqual(cam[-1], 490, delta=1)

    def test_crop_box_is_9_16_and_inside_frame(self):
        for cx in (0, 640, 1280):
            x, y, cw, ch = hh.crop_box(cx, 360, 1.0, 1280, 720)
            self.assertEqual((cw, ch), (405, 720))
            self.assertTrue(0 <= x <= 1280 - cw and y == 0)
        x, y, cw, ch = hh.crop_box(640, 700, 1.2, 1280, 720)
        self.assertEqual(ch, 600)
        self.assertEqual(y, 120)

    def test_zoom_alternates_per_sentence(self):
        self.assertEqual(hh.zoom_schedule([0, 1, 2.5, 4], [0, 2, 3.5], punch=1.2), [1.0, 1.0, 1.2, 1.0])


class SubtitleTest(unittest.TestCase):
    def test_one_event_per_word_with_single_highlight(self):
        words = [w(1.0, 1.2, "the"), w(1.3, 1.6, "universe,"), w(1.7, 2.0, "is"), w(2.1, 2.4, "{big}")]
        ass = hh.build_ass(words, offset=1.0, labels=[(0, 1, "BEFORE")])
        events = [l for l in ass.splitlines() if l.startswith("Dialogue: 0")]
        self.assertEqual(len(events), 4)
        for e in events:
            self.assertEqual(e.count(hh.HIGHLIGHT), 1)
        self.assertIn("0:00:00.00,0:00:00.30", events[0])
        self.assertIn("BIG", events[3])
        self.assertNotIn("{BIG}", ass)
        self.assertIn("Dialogue: 1,0:00:00.00,0:00:01.00,Label,,0,0,0,,BEFORE", ass)

    def test_word_events_never_overlap(self):
        words = [w(0.0, 0.3, "form"), w(0.35, 0.5, "of"), w(0.55, 0.9, "light,"), w(1.0, 1.2, "which"), w(1.25, 1.4, "is")]
        ass = hh.build_ass(words)
        spans = []
        for line in ass.splitlines():
            if line.startswith("Dialogue: 0"):
                _, a, b = line.split(",")[:3]
                spans.append((a, b))
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start)

    def test_chunks_break_on_comma_and_size(self):
        words = [w(i, i + .5, t) for i, t in enumerate(["a", "b,", "c", "d", "e", "f"])]
        self.assertEqual([len(c) for c in hh.chunk_words(words, max_words=3)], [2, 3, 1])

    def test_ass_time_format(self):
        self.assertEqual(hh.ass_time(3723.456), "1:02:03.46")


if __name__ == "__main__":
    unittest.main()
