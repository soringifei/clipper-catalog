"""Offline checks of reviewed pure guards; never imports the pipeline or calls any model/media runtime."""
import ast
import datetime as dt
from pathlib import Path
import re
import tempfile
import unittest

SOURCE = Path(__file__).with_name("make_daily.py")
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMES = {"batch_directory", "source_path", "attach_source_evidence", "load_cover_module"}
FUNCTIONS = ast.Module(body=[node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in NAMES], type_ignores=[])

class SafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clipper = Path(self.temp.name) / "clipper"
        self.root = self.clipper / "daily"
        self.root.mkdir(parents=True)
        self.ns = {"Path": Path, "dt": dt, "re": re, "ROOT": self.root, "CLIPPER": self.clipper}
        exec(compile(FUNCTIONS, str(SOURCE), "exec"), self.ns)

    def test_date_exact_calendar_child(self):
        self.assertEqual(self.ns["batch_directory"]("2026-09-30"), self.root / "2026-09-30")
        for invalid in ["../escape", "2026-9-30", "2026-02-30", "2026-09-30/../x", str(self.clipper), ""]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.ns["batch_directory"](invalid)

    def test_source_id_and_path_containment(self):
        source = {"id": "crippen-sts1", "path": "../sources-master4/crippen-sts1.mp4"}
        self.assertEqual(self.ns["source_path"](source), self.clipper / "sources-master4/crippen-sts1.mp4")
        for bad_id in ["../escape", "a/b", "", "UPPER", "a\nb"]:
            with self.subTest(value=bad_id), self.assertRaises(ValueError):
                self.ns["source_path"]({**source, "id": bad_id})
        for bad_path in ["../../outside.mp4", str(Path(self.temp.name) / "outside.mp4"), ""]:
            with self.subTest(value=bad_path), self.assertRaises(ValueError):
                self.ns["source_path"]({**source, "path": bad_path})

    def test_only_manifest_receipt_propagates_no_attestation_created(self):
        receipt = {"status": "VERIFIED_PUBLIC", "contentSha256": "reviewed-elsewhere"}
        candidate = {"source": "s", "sourceEvidence": {"forged": True}}
        self.ns["attach_source_evidence"]([candidate], {"s": {"sourceEvidence": receipt}})
        self.assertIs(candidate["sourceEvidence"], receipt)
        self.assertEqual(candidate["jev_status"], "UNVERIFIED")
        self.ns["attach_source_evidence"]([candidate], {})
        self.assertIsNone(candidate["sourceEvidence"])

    def test_missing_cover_dependency_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "not included"):
            self.ns["load_cover_module"]()

    def test_no_daemon_or_dynamic_cover_loader(self):
        calls = [node.func.attr for node in ast.walk(TREE) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("Popen", calls)
        self.assertNotIn("exec_module", calls)
        self.assertNotIn("Register-ScheduledTask", SOURCE.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
