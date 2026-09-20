import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "campaign", Path(__file__).resolve().parents[1] / "head-only-campaign.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kit = self.root / "target/head-only-kit"
        self.kit.mkdir(parents=True)
        (self.kit / "build-id").write_text("20260920T120000")
        (self.kit / "run").write_text("runner")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/run-head-only-on-robot").write_text("runner")
        self.destination = self.root / "logs/head-only/robot.test/20260920T120000"
        self.calls = []

    def execute(self, failure=None, reason="completed", extra=()):
        def run(command):
            self.calls.append(command)
            if command[1] == "run":
                if failure == "interrupt":
                    raise KeyboardInterrupt
                if failure == "exit":
                    return 1
                if failure == "missing":
                    return 0
                run_id = command[command.index("--run-id") + 1]
                directory = self.destination / run_id
                directory.mkdir(parents=True)
                (directory / "result.json").write_text(json.dumps({
                    "stop_reason": reason, "test_error": None,
                    "recording_error": None, "damping_error": None,
                }))
                if failure == "kit_change":
                    (self.kit / "build-id").write_text("20260920T130000")
            if command[1] == "fetch" and failure == "fetch":
                return 1
            return 0

        with patch.object(campaign, "ROOT", self.root), \
             patch.object(campaign, "run_process", side_effect=run), \
             patch.object(campaign.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            code = campaign.main(["robot.test", *extra])
        manifests = list((self.destination / "campaigns").glob("*.json"))
        return code, json.loads(manifests[0].read_text()) if manifests else None

    def test_matrix_changes_only_selected_joint_and_has_unique_recording_ids(self):
        for joint in ["yaw", "pitch"]:
            steps = campaign.plan(campaign.arguments(["robot.test", "--joint", joint]), "test")
            self.assertEqual(len(steps), 16)
            self.assertEqual(len({s["run_id"] for s in steps}), 16)
            self.assertEqual(
                [(s["gains"][f"{joint}_kp"], s["gains"][f"{joint}_kd"]) for s in steps[::4]],
                [(10, 1.2), (12, 1.2), (10, 1.4), (12, 1.4)],
            )
            other = "pitch" if joint == "yaw" else "yaw"
            self.assertTrue(all(s["gains"][f"{other}_kp"] == 10 and
                                s["gains"][f"{other}_kd"] == 1.2 for s in steps))
            self.assertEqual([(s["yaw"], s["pitch"]) for s in steps[:3]],
                             [(0, .7), (.95, .5), (-.95, .5)])
            self.assertEqual(steps[3]["pattern"], "scan")

    def test_success_downloads_and_verifies_every_run(self):
        code, manifest = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(all(s["status"] == "completed" for s in manifest["steps"]))
        self.assertEqual([c[1] for c in self.calls], ["run", "fetch"] * 16)

    def test_failure_stops_without_starting_next_run_and_fetches(self):
        code, manifest = self.execute(failure="exit")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual([c[1] for c in self.calls], ["run", "fetch"])
        self.assertTrue(all(s["status"] == "planned" for s in manifest["steps"][1:]))

    def test_remote_stop_with_zero_exit_does_not_rearm(self):
        code, manifest = self.execute(reason="sigterm")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(sum(c[1] == "run" for c in self.calls), 1)

    def test_missing_result_does_not_rearm(self):
        code, manifest = self.execute(failure="missing")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(sum(c[1] == "run" for c in self.calls), 1)

    def test_interrupt_requests_damping_and_fetches_without_restarting(self):
        code, manifest = self.execute(failure="interrupt")
        self.assertEqual(code, 130)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([c[1] for c in self.calls], ["run", "stop", "fetch"])

    def test_dry_run_never_contacts_robot_or_writes_logs(self):
        code, manifest = self.execute(extra=["--dry-run"])
        self.assertEqual(code, 0)
        self.assertIsNone(manifest)
        self.assertEqual(self.calls, [])

    def test_failed_download_prevents_next_run(self):
        code, manifest = self.execute(failure="fetch")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(sum(c[1] == "run" for c in self.calls), 1)

    def test_kit_change_prevents_mixed_builds_and_wrong_fetch(self):
        code, manifest = self.execute(failure="kit_change")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual([c[1] for c in self.calls], ["run"])


if __name__ == "__main__":
    unittest.main()
