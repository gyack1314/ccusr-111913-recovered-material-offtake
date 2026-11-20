"""命令行端到端测试：临时工作区，不写项目 .local、不联网。"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.local = pathlib.Path(self._tmp.name) / ".local"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "recovery", "--local", str(self.local), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )

    def test_check_reads_original_materials(self):
        result = self.run_cli("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recovered-material-offtake", result.stdout)

    def test_quote_creates_snapshot_and_accept_flow(self):
        result = self.run_cli("quote", "--orders", "O1", "--format", "json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        snapshot_id = payload["快照"]["id"]
        self.assertTrue((self.local / "snapshots" / f"{snapshot_id}.json").exists())
        accepted = self.run_cli("accept", snapshot_id, "--orders", "O1")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        again = self.run_cli("accept", snapshot_id, "--orders", "O1")
        self.assertNotEqual(again.returncode, 0)

    def test_receipt_and_disposition(self):
        quote = self.run_cli("quote", "--orders", "O1", "--format", "json")
        snapshot_id = json.loads(quote.stdout)["快照"]["id"]
        self.run_cli("accept", snapshot_id, "--orders", "O1")
        receipt = self.run_cli(
            "receipt", "--id", "C9", "--lot", "L009", "--order", "O1", "--kind", "customer_assay",
            "--quantity", "2100", "--observed-impurity", "0.09", "--known-at", "2026-09-21T10:00:00+08:00",
        )
        self.assertEqual(receipt.returncode, 0, receipt.stderr)
        disposition = self.run_cli("disposition", "C9", "dispute", "--reason", "检测差异")
        self.assertEqual(disposition.returncode, 0, disposition.stderr)
        realized = self.run_cli("realized")
        payload = json.loads(realized.stdout)
        self.assertEqual(payload["客户复检与处置"][0]["处置"], "dispute")

    def test_recheck_detects_expired_quote(self):
        quote = self.run_cli("quote", "--orders", "O1", "--format", "json")
        snapshot_id = json.loads(quote.stdout)["快照"]["id"]
        result = self.run_cli("--plan-date", "2026-09-25", "recheck", snapshot_id)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["仍可兑现"])

    def test_demo_runs_without_touching_project_local(self):
        before = (ROOT / ".local" / "state.json").exists()
        result = self.run_cli("demo", "--orders", "O1", "O3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("候选方案与风险比较", result.stdout)
        self.assertEqual(before, (ROOT / ".local" / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
