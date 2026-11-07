"""Focused checks for completion evidence and resumed-verification selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .historical_coverage import CORE, ANNUAL, MATRIX, EvidenceError, MODELS, digest, snapshot, verify_fold


class CompletedFoldEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "outputs/multiscale"
        self.fold = self.out / "fold_02"
        self.fold.mkdir(parents=True)
        self.source = self.root / "multiscale/training.py"
        self.source.parent.mkdir()
        self.source.write_text("source version\n", encoding="utf8")
        self.standard = self.out / "fold_02_verification.json"
        self.alternative = self.out / "fold_02_verification_night_resumed.json"
        self.expected = ("minute", 5, "future_close", 2)
        self.job = {"validation_origins": 2, "split_sha256": {"validation": "same-origins"}}
        self.receipt = {
            "status": "completed", "phase": "development", "Q3_values_read": False,
            "independent_test": False, "config": {"profile": "minute", "horizon": 5,
                "target": "future_close", "protocol_identity": "protocol-identity"},
            "validation_origins": 2, "identity_fields": {"validation_indices_sha256": "same-origins"},
            "models": {model: {} for model in MODELS}, "targets_sha256": "target-digest",
            "source_sha256": {"multiscale/training.py": digest(self.source)}
        }
        self.write_receipt()
        self.write_verification(self.alternative)
        self.entry = {"status": "completed", "exit_code": 0,
                      "receipt_sha256": digest(self.fold / "fold_receipt.json"),
                      "verification": str(self.alternative), "verification_sha256": digest(self.alternative)}

    def write_receipt(self):
        (self.fold / "fold_receipt.json").write_text(json.dumps(self.receipt), encoding="utf8")

    def write_verification(self, path, *, receipt_hash=None):
        value = {"status": "verified_saved_fold", "fold_directory": str(self.fold),
                 "fold_receipt_sha256": receipt_hash or digest(self.fold / "fold_receipt.json"),
                 "targets_sha256": "target-digest", "protocol_identity": "protocol-identity",
                 "development_cutoff": "2026-06-30", "independent_test": False,
                 "models": {model: {} for model in MODELS},
                 "current_source_matches_recorded": {"multiscale/training.py": True}}
        path.write_text(json.dumps(value), encoding="utf8")

    def verify(self):
        return verify_fold(self.entry, self.expected, self.job, self.standard, self.fold,
                           protocol_hash="protocol-identity", project_root=self.root, output_root=self.out)

    def rebind_receipt(self):
        self.write_receipt()
        self.write_verification(self.alternative)
        self.entry["receipt_sha256"] = digest(self.fold / "fold_receipt.json")
        self.entry["verification_sha256"] = digest(self.alternative)

    def test_resumed_verification_is_used_when_original_is_stale(self):
        self.write_verification(self.standard, receipt_hash="stale-receipt")
        self.assertEqual(self.verify()["verification"], str(self.alternative))

    def test_completed_status_does_not_mask_a_changed_receipt(self):
        self.receipt["validation_origins"] = 3
        self.write_receipt()
        with self.assertRaises(EvidenceError):
            self.verify()

    def test_unmatched_validation_origins_are_rejected(self):
        self.job["split_sha256"]["validation"] = "different-origins"
        with self.assertRaises(EvidenceError):
            self.verify()

    def test_unmatched_protocol_is_rejected(self):
        self.receipt["config"]["protocol_identity"] = "other-protocol"
        self.rebind_receipt()
        with self.assertRaises(EvidenceError):
            self.verify()

    def test_holdout_exposure_is_rejected_even_with_matching_hashes(self):
        self.receipt["Q3_values_read"] = True
        self.rebind_receipt()
        with self.assertRaises(EvidenceError):
            self.verify()

    def test_changed_source_is_rejected(self):
        self.source.write_text("different source\n", encoding="utf8")
        with self.assertRaises(EvidenceError):
            self.verify()

    def test_noncompleted_worker_is_not_counted(self):
        self.entry["status"] = "paused_night"
        with self.assertRaises(EvidenceError):
            self.verify()


class CurrentInventoryTest(unittest.TestCase):
    @unittest.skipUnless(all(p.is_dir() for p in (CORE, ANNUAL, MATRIX)), 'Recorded research outputs are local-only')
    def test_every_planned_task_fold_has_a_unique_status(self):
        result = snapshot()
        keys = {(r["profile"], r["horizon"], r["target"], r["fold"]) for r in result["folds"]}
        self.assertEqual((result["planned_tasks"], result["planned_folds"], len(keys)), (20, 100, 100))
        for row in result["folds"]:
            self.assertEqual("evidence" in row, row["status"] == "completed")


if __name__ == "__main__":
    unittest.main()
