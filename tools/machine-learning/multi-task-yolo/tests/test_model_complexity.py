from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from click.testing import CliRunner

from utils import model_complexity


class FakeYOLO:
    def __init__(self, _path: Path) -> None:
        self.model = model_complexity.nn.Sequential(
            model_complexity.nn.Linear(1, 1),
        )


class ModelComplexityTest(TestCase):
    def test_checkpoint_report_path_uses_checkpoint_stem(self) -> None:
        checkpoint_path = Path.cwd() / "assets" / "yolo26m.pt"

        report_path = model_complexity.checkpoint_report_path(
            Path("runs"),
            checkpoint_path,
        )

        self.assertEqual(
            report_path,
            Path("runs/complexity/yolo26m/report.json"),
        )

    def test_profile_checkpoint_writes_report(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            checkpoint_path.write_bytes(b"weights")
            report_path = Path(tmp_dir) / "report.json"

            with (
                patch.object(model_complexity, "YOLO", FakeYOLO),
                patch.object(model_complexity, "get_flops", return_value=1.0),
            ):
                result = model_complexity.profile_checkpoint(
                    checkpoint_path,
                    imgsz=32,
                    device="cpu",
                    report_path=report_path,
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result.report_path,
            model_complexity.display_path(
                report_path,
            ),
        )
        self.assertEqual(
            report["path"],
            model_complexity.display_path(
                checkpoint_path,
            ),
        )
        self.assertEqual(report["parameters"], 2)
        self.assertEqual(report["gflops"], 1.0)

    def test_direct_checkpoint_paths_write_reports(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            assets_dir = Path("assets")
            assets_dir.mkdir()
            checkpoint_path = assets_dir / "yolo26m.pt"
            checkpoint_path.write_bytes(b"weights")

            with (
                patch.object(model_complexity, "YOLO", FakeYOLO),
                patch.object(model_complexity, "get_flops", return_value=1.0),
            ):
                result = runner.invoke(
                    model_complexity.main,
                    [str(checkpoint_path)],
                )

            self.assertEqual(result.exit_code, 0)
            report_path = Path("runs/complexity/yolo26m/report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["path"], str(checkpoint_path))
        self.assertEqual(report["input_size"], 640)
        self.assertEqual(report["report_path"], str(report_path))
