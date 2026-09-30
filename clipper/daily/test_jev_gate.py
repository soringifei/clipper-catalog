import unittest
from unittest.mock import patch
import jev_gate


def candidate(description="Why Mars dust matters"):
    return {"start": 10.0, "words": [{"w": "Mars", "s": 10.0}],
            "text": "Mars dust affects equipment.", "llm": {"description": description}}


def reviewed(c):
    c["sourceEvidence"] = {"status": "VERIFIED_PUBLIC", "sourceUrl": "https://www.nasa.gov/podcasts/example/",
        "contentSha256": jev_gate.payload_sha256(jev_gate.candidate_item(c)),
        "reviewedAt": "2026-09-01T00:00:00Z", "reviewedBy": "fixture-reviewer"}
    return c


def fake(**values):
    def evaluate(state, questions, purpose):
        return {key: ({"score": values.get(key.split("_")[1], 3)} if q["type"] == "score"
                     else {"probability": values.get(key.split("_")[1], 0.1 if key.endswith("slop") else 0.9)})
                for key, q in questions.items()}
    return evaluate


class GateTests(unittest.TestCase):
    def test_missing_provenance_does_not_load_or_call_backend(self):
        c = candidate()
        with patch.object(jev_gate, "_default_evaluate", side_effect=AssertionError("must not load")):
            self.assertEqual(jev_gate.gate([c]), [c])
        self.assertIsNone(c["jev"])
        self.assertEqual(c["jev_status"], "UNVERIFIED")

    def test_injected_backend_also_requires_provenance(self):
        c = candidate()
        self.assertEqual(jev_gate.gate([c], evaluate=lambda *a: self.fail("private text sent")), [c])

    def test_url_and_receipt_are_both_required(self):
        invalid = ["https://nasa.gov.evil.test/x", "http://www.nasa.gov/x", "https://user@nasa.gov/x", "https://nasa.gov:8443/x"]
        for url in invalid:
            c = reviewed(candidate()); c["sourceEvidence"]["sourceUrl"] = url
            self.assertFalse(jev_gate.source_evidence_valid(jev_gate.candidate_item(c)))
        for field in ("reviewedAt", "reviewedBy", "contentSha256", "status"):
            c = reviewed(candidate()); del c["sourceEvidence"][field]
            self.assertFalse(jev_gate.source_evidence_valid(jev_gate.candidate_item(c)))

    def test_future_or_naive_timestamp_is_not_reviewed(self):
        for timestamp in ("2999-01-01T00:00:00Z", "2026-09-01", "invalid"):
            c = reviewed(candidate()); c["sourceEvidence"]["reviewedAt"] = timestamp
            self.assertFalse(jev_gate.source_evidence_valid(jev_gate.candidate_item(c)))

    def test_changed_outgoing_field_invalidates_receipt(self):
        for field in ("opening", "text", "description"):
            item = jev_gate.candidate_item(reviewed(candidate()))
            item[field] += " private addition"
            self.assertFalse(jev_gate.source_evidence_valid(item))

    def test_valid_receipt_and_complete_scores_are_text_only(self):
        c = reviewed(candidate())
        self.assertEqual(jev_gate.gate([c], evaluate=fake()), [c])
        self.assertTrue(c["jev"]["pass"])
        self.assertEqual(c["jev_status"], "SCORED_TEXT_ONLY")

    def test_evaluation_unavailable_or_raises_is_unverified(self):
        def broken(*args): raise RuntimeError("unavailable")
        for ev in (lambda *a: None, broken):
            c = reviewed(candidate())
            self.assertEqual(jev_gate.gate([c], evaluate=ev), [c])
            self.assertIsNone(c["jev"])
            self.assertEqual(c["jev_status"], "UNVERIFIED")

    def test_partial_malformed_or_nonfinite_result_never_passes(self):
        for ev in (lambda *a: {"c0_hook": {"score": 4}}, lambda *a: {"c0_hook": "bad"},
                   fake(hook=float("nan")), fake(slop=float("inf")), fake(whole=True), fake(hook=5)):
            c = reviewed(candidate())
            jev_gate.gate([c], evaluate=ev)
            self.assertIsNone(c["jev"])

    def test_bad_auto_candidate_dropped_editor_retained_with_failure(self):
        c = reviewed(candidate())
        self.assertEqual(jev_gate.gate([c], evaluate=fake(hook=0.5)), [])
        editor = candidate(); editor["copy"] = {"description": "Editor draft"}; reviewed(editor)
        self.assertEqual(jev_gate.gate([editor], evaluate=fake(hook=0.5)), [editor])
        self.assertFalse(editor["jev"]["pass"])

    def test_no_description_does_not_require_description_score(self):
        c = reviewed(candidate(""))
        self.assertEqual(jev_gate.gate([c], evaluate=fake()), [c])
        self.assertTrue(c["jev"]["pass"])

    def test_mixed_public_private_only_sends_reviewed_text(self):
        private = candidate(); private["text"] = "PRIVATE SENTINEL"
        public = reviewed(candidate())
        def evaluate(state, questions, purpose):
            self.assertNotIn("PRIVATE SENTINEL", state)
            return fake()(state, questions, purpose)
        jev_gate.gate([private, public], evaluate=evaluate)
        self.assertIsNone(private["jev"])
        self.assertTrue(public["jev"]["pass"])


if __name__ == "__main__":
    unittest.main()
