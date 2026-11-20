"""报价快照、接受、回执、复检处置与重判的工作流测试。"""
import json
import pathlib
import tempfile
import unittest

from recovery.config import DEFAULT
from recovery.context import build_context
from recovery.quotes import accept_with_snapshot, build_candidate_snapshot, reapproval_report
from recovery.state import DataError, Workspace


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.local = pathlib.Path(self._tmp.name) / ".local"
        self.workspace = Workspace(local=self.local)
        self.cfg = DEFAULT.with_(sims_portfolio=300)

    def tearDown(self):
        self._tmp.cleanup()

    def _quote(self, pending):
        ctx = build_context(self.cfg, self.workspace)
        evaluation, snapshot = build_candidate_snapshot(ctx, self.cfg, self.workspace, pending=pending)
        self.assertTrue(snapshot["id"])
        return ctx, evaluation, snapshot

    def test_snapshot_is_immutable_and_contains_evidence(self):
        _, evaluation, snapshot = self._quote(["O1"])
        path = self.local / "snapshots" / f"{snapshot['id']}.json"
        before = path.read_text(encoding="utf-8")
        with self.assertRaises(DataError):
            self.workspace.save_snapshot(snapshot)
        self.assertEqual(before, path.read_text(encoding="utf-8"))
        self.assertIn("lots", snapshot["evidence"])
        self.assertIn("assumptions", snapshot)
        self.assertIn("sensitivity", snapshot)

    def test_cannot_accept_same_order_twice(self):
        _, _, snapshot = self._quote(["O1"])
        accept_with_snapshot(self.workspace, snapshot, ["O1"], self.cfg)
        with self.assertRaises(DataError):
            self.workspace.register_acceptance(snapshot, ["O1"])

    def test_cancel_releases_unprocessed_but_not_processed(self):
        _, _, snapshot = self._quote(["O1"])
        accept_with_snapshot(self.workspace, snapshot, ["O1"], self.cfg)
        self.workspace.register_receipt({
            "id": "P1", "lot": "L009", "order": "O1", "kind": "processed",
            "quantity_kg": 500, "known_at": "2026-09-19T10:00:00+08:00",
        })
        ctx = build_context(self.cfg, self.workspace)
        self.assertAlmostEqual(ctx.available_wet["L009"], ctx.lots["L009"].wet_kg - 500, places=3)
        self.workspace.register_cancellation("O1", "客户改期")
        ctx_after = build_context(self.cfg, self.workspace)
        self.assertNotIn("O1", ctx_after.orders)
        self.assertNotIn("O1", ctx_after.accepted)

    def test_customer_assay_disposition_changes_remaining_not_snapshot(self):
        _, _, snapshot = self._quote(["O1"])
        accept_with_snapshot(self.workspace, snapshot, ["O1"], self.cfg)
        original = dict(snapshot["verdict"])
        receipt = self.workspace.register_receipt({
            "id": "C1", "lot": "L009", "order": "O1", "kind": "customer_assay",
            "quantity_kg": 2100, "observed_impurity": 0.095,
            "known_at": "2026-09-21T10:00:00+08:00",
        })
        self.workspace.register_disposition("C1", "dispute", reason="双方检测差异", return_kg=2100)
        disposition = self.workspace.state["dispositions"]["C1"]
        self.assertEqual(disposition["disposition"], "dispute")
        # 旧快照的报价证据不被改写
        reloaded = self.workspace.load_snapshot(snapshot["id"])
        self.assertEqual(reloaded["verdict"], original)
        self.assertEqual(receipt["kind"], "customer_assay")

    def test_recheck_flags_expired_price(self):
        _, _, snapshot = self._quote(["O1"])
        ctx = build_context(self.cfg, self.workspace)
        report = reapproval_report(self.workspace, snapshot["id"], ctx, self.cfg, plan_date="2026-09-25")
        self.assertTrue(report["已过期"])
        self.assertFalse(report["仍可兑现"])

    def test_recheck_sees_new_assay(self):
        _, _, snapshot = self._quote(["O1"])
        accept_with_snapshot(self.workspace, snapshot, ["O1"], self.cfg)
        self.workspace.register_assay({
            "id": "QX", "lot": "L009", "sample_stratum": "layer9", "sampling_basis": "random_mass",
            "represented_wet_kg": 2800, "moisture_fraction": 0.12, "impurity_fraction_dry": 0.09,
            "known_at": "2026-09-19T10:00:00+08:00",
        })
        ctx = build_context(self.cfg, self.workspace, knowledge_cut="2026-09-19T23:59:59+08:00")
        report = reapproval_report(self.workspace, snapshot["id"], ctx, self.cfg, plan_date="2026-09-19")
        self.assertFalse(report["仍可兑现"])

    def test_receipt_requires_accepted_order(self):
        with self.assertRaises(DataError):
            self.workspace.register_receipt({
                "id": "P2", "lot": "L001", "order": "O1", "kind": "processed", "quantity_kg": 100,
            })

    def test_decision_log_is_append_only(self):
        _, _, snapshot = self._quote(["O1"])
        accept_with_snapshot(self.workspace, snapshot, ["O1"], self.cfg)
        self.workspace.register_cancellation("O1", "测试")
        lines = (self.local / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line)["type"] for line in lines]
        self.assertEqual(kinds[:2], ["accept", "cancel"])


if __name__ == "__main__":
    unittest.main()
