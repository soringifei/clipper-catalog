import unittest

import jev_gate


def fake(hook=3, whole=0.9, slop=0.1, desc=3, calls=None):
    def ev(state, questions, purpose):
        if calls is not None:
            calls.append(questions)
        out = {}
        for k, q in questions.items():
            kind = k.split("_")[1]
            if kind == "hook":
                out[k] = {"type": "score", "score": hook(k) if callable(hook) else hook}
            elif kind == "desc":
                out[k] = {"type": "score", "score": desc}
            else:
                out[k] = {"type": "boolean", "probability": whole if kind == "whole" else slop}
        return out
    return ev


def cand(n, copy=None, desc="Why Mars dust matters"):
    return {"start": 10.0, "words": [{"w": "Mars", "s": 10.0}, {"w": "is", "s": 10.5}, {"w": "later", "s": 13.0}],
            "text": f"Mars is later {n}", "hook": "Mars is", "llm": {"description": desc}, "copy": copy}


class GateTest(unittest.TestCase):
    def test_batches_two_candidates_per_call_with_8_questions(self):
        calls = []
        jev_gate.gate([cand(i) for i in range(5)], evaluate=fake(calls=calls))
        self.assertEqual([len(q) for q in calls], [8, 8, 4])

    def test_drops_weak_auto_candidates_keeps_editor_picks_flagged(self):
        weak = fake(hook=lambda k: 0.5 if k.startswith("c0") else 3)
        cs = [cand(0), cand(1)]
        kept = jev_gate.gate(cs, evaluate=weak)
        self.assertEqual(kept, [cs[1]])
        self.assertFalse(cs[0]["jev"]["pass"]); self.assertIn("hook 0.5", cs[0]["jev"]["why"])
        ed = [cand(0, copy={"description": "x"}), cand(1)]
        self.assertEqual(len(jev_gate.gate(ed, evaluate=weak)), 2)
        self.assertFalse(ed[0]["jev"]["pass"])

    def test_slop_incomplete_and_description_thresholds(self):
        for kw, why in [({"slop": 0.8}, "formulaic text"), ({"whole": 0.2}, "incomplete"), ({"desc": 0.4}, "description")]:
            c = cand(0)
            self.assertEqual(jev_gate.gate([c], evaluate=fake(**kw)), [])
            self.assertTrue(any(w.startswith(why) for w in c["jev"]["why"]))

    def test_borderline_hook_is_flagged_not_dropped(self):
        c = cand(0)
        self.assertEqual(jev_gate.gate([c], evaluate=fake(hook=1.2)), [c])
        self.assertEqual((c["jev"]["pass"], c["jev"]["drop"]), (False, False))

    def test_fail_open_when_jev_unavailable(self):
        cs = [cand(0), cand(1)]
        self.assertEqual(jev_gate.gate(cs, evaluate=lambda *a: None), cs)
        self.assertIsNone(cs[0]["jev"])

    def test_no_description_question_without_description(self):
        calls = []
        jev_gate.gate([cand(0, desc="")], evaluate=fake(calls=calls))
        self.assertEqual(sorted(calls[0]), ["c0_hook", "c0_slop", "c0_whole"])

    def test_opening_helpers(self):
        self.assertEqual(jev_gate.first_seconds(cand(0)["words"], 10.0), "Mars is")
        it = {"quality_notes": ["x", "2.5 words/s"], "transcript": "one two three four five six seven eight"}
        self.assertEqual(jev_gate.opening_from_item(it), "one two three four five")


if __name__ == "__main__":
    unittest.main()
