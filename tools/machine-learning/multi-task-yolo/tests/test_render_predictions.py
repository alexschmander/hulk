from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np
from click.testing import CliRunner

from utils.model_naming import TaskType
from validation import render_predictions


class RenderPredictionsTest(TestCase):
    def test_load_val_images_samples_uniformly_from_val_split(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            val_dir = root / "images" / "val"
            train_dir = root / "images" / "train"
            val_dir.mkdir(parents=True)
            train_dir.mkdir(parents=True)
            for index in range(5):
                write_image(val_dir / f"{index}.jpg")

            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "val: images/val",
                        "names:",
                        "  0: person",
                    ]
                ),
                encoding="utf-8",
            )

            _, images = render_predictions.load_val_images(
                yaml_path,
                num_images=3,
            )

        self.assertEqual(
            [image.name for image in images],
            ["0.jpg", "2.jpg", "4.jpg"],
        )

    def test_relative_image_path_flattens_images_val_segment(self) -> None:
        dataset_root = Path("/datasets")
        image_path = (
            dataset_root
            / "smd-fieldmarks-balls-robots-persons"
            / "images"
            / "val"
            / "example.png"
        )

        relative_path = render_predictions.relative_image_path(
            image_path,
            dataset_root,
        )

        self.assertEqual(
            relative_path,
            Path("smd-fieldmarks-balls-robots-persons/example.png"),
        )

    def test_load_source_images_samples_uniformly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source"
            source_dir.mkdir()
            for index in range(5):
                write_image(source_dir / f"{index}.jpg")

            images = render_predictions.load_source_images(
                source_dir,
                num_images=3,
            )

        self.assertEqual(
            [image.name for image in images],
            ["0.jpg", "2.jpg", "4.jpg"],
        )

    def test_parse_detect_and_pose_labels_as_boxes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "image.txt"
            label_path.write_text(
                "0 0.5 0.5 0.4 0.2 0.1 0.2 2\n",
                encoding="utf-8",
            )

            labels = render_predictions.parse_ground_truth_labels(
                label_path,
                image_shape=(100, 200),
                task_type=TaskType.POSE,
            )

        self.assertEqual(labels[0].cls, 0)
        np.testing.assert_allclose(
            labels[0].box_xyxy,
            np.array([60, 40, 140, 60], dtype=np.float32),
        )
        self.assertIsNone(labels[0].polygon_xy)

    def test_parse_segment_labels_as_polygons(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "image.txt"
            label_path.write_text(
                "1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n",
                encoding="utf-8",
            )

            labels = render_predictions.parse_ground_truth_labels(
                label_path,
                image_shape=(100, 200),
                task_type=TaskType.SEGMENTATION,
            )

        self.assertEqual(labels[0].cls, 1)
        np.testing.assert_allclose(
            labels[0].polygon_xy,
            np.array(
                [[20, 10], [100, 10], [100, 50], [20, 50]],
                dtype=np.float32,
            ),
        )

    def test_resolve_finetuned_head_path_from_training_run(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = (
                root
                / "runs"
                / "train"
                / "yolo26m-seg=f17+yolo26m~judge"
                / "weights"
                / "best.pt"
            )
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"weights")
            hydra_model_name = render_predictions.HydraModelName.parse(
                "yolo26m-seg=f17+yolo26m~judge",
            )

            resolved_path = render_predictions.resolve_head_path(
                hydra_model_name,
                assets_dir=root / "assets",
                runs_dir=root / "runs",
                train_dir=Path("train"),
            )

        self.assertEqual(resolved_path, checkpoint_path)

    def test_same_class_bbox_iou_filtering(self) -> None:
        labels = [
            render_predictions.GroundTruthLabel(
                cls=0,
                box_xyxy=np.array([12, 12, 48, 48], dtype=np.float32),
                polygon_xy=None,
            )
        ]
        predictions = [
            render_predictions.Prediction(
                cls=0,
                conf=0.9,
                box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            ),
            render_predictions.Prediction(
                cls=1,
                conf=0.9,
                box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            ),
            render_predictions.Prediction(
                cls=0,
                conf=0.9,
                box_xyxy=np.array([60, 60, 75, 75], dtype=np.float32),
            ),
        ]

        filtered = render_predictions.filter_predictions(
            predictions,
            labels,
            task_type=TaskType.OBJECT,
            image_shape=(100, 100),
            conf_threshold=0.25,
            iou_threshold=0.5,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].cls, 0)

    def test_mask_iou_filtering(self) -> None:
        polygon = np.array(
            [[10, 10], [50, 10], [50, 50], [10, 50]],
            dtype=np.float32,
        )
        labels = [
            render_predictions.GroundTruthLabel(
                cls=0,
                box_xyxy=render_predictions.polygon_to_box(polygon),
                polygon_xy=polygon,
            )
        ]
        predictions = [
            render_predictions.Prediction(
                cls=0,
                conf=0.9,
                box_xyxy=render_predictions.polygon_to_box(polygon),
                mask_xy=polygon,
            ),
            render_predictions.Prediction(
                cls=0,
                conf=0.9,
                box_xyxy=np.array([60, 60, 80, 80], dtype=np.float32),
                mask_xy=np.array(
                    [[60, 60], [80, 60], [80, 80], [60, 80]],
                    dtype=np.float32,
                ),
            ),
        ]

        filtered = render_predictions.filter_predictions(
            predictions,
            labels,
            task_type=TaskType.SEGMENTATION,
            image_shape=(100, 100),
            conf_threshold=0.25,
            iou_threshold=0.9,
        )

        self.assertEqual(len(filtered), 1)
        self.assertGreaterEqual(filtered[0].iou, 0.9)

    def test_custom_renderer_draws_all_task_boxes(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.array([10, 10, 70, 70], dtype=np.float32)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = 30
        keypoints[:, 1] = 30
        keypoints[:, 2] = 1
        polygon = np.array(
            [[10, 10], [70, 10], [70, 70], [10, 70]],
            dtype=np.float32,
        )

        task_predictions = {
            TaskType.OBJECT: render_predictions.Prediction(0, 0.9, box),
            TaskType.POSE: render_predictions.Prediction(
                0,
                0.9,
                box,
                keypoints=keypoints,
            ),
            TaskType.SEGMENTATION: render_predictions.Prediction(
                0,
                0.9,
                box,
                mask_xy=polygon,
            ),
        }

        for task_type, prediction in task_predictions.items():
            rendered = render_predictions.render_predictions(
                image,
                [prediction],
                names={0: "person"},
                task_type=task_type,
            )
            self.assertGreater(int(rendered.sum()), 0)

    def test_small_box_label_floats_outside_for_readability(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        prediction = render_predictions.Prediction(
            cls=0,
            conf=0.99,
            box_xyxy=np.array([5, 5, 14, 14], dtype=np.float32),
        )

        rendered = render_predictions.render_predictions(
            image,
            [prediction],
            names={0: "Ball"},
            task_type=TaskType.OBJECT,
        )

        self.assertEqual(
            render_predictions.label_font_scale(9),
            render_predictions.LABEL_MIN_FONT_SCALE,
        )
        self.assertEqual(
            render_predictions.label_position(
                box=(5, 5, 14, 14),
                label_size=(50, 20),
                image_size=(80, 120),
            ),
            (5, 14),
        )
        self.assertGreater(int(rendered[15:30, 5:70].sum()), 0)

    def test_small_box_label_is_drawn_over_larger_label(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        large_prediction = render_predictions.Prediction(
            cls=1,
            conf=0.90,
            box_xyxy=np.array([6, 5, 100, 70], dtype=np.float32),
        )
        small_prediction = render_predictions.Prediction(
            cls=0,
            conf=0.99,
            box_xyxy=np.array([5, 5, 14, 14], dtype=np.float32),
        )

        rendered = render_predictions.render_predictions(
            image,
            [small_prediction, large_prediction],
            names={0: "Ball", 1: "Robot"},
            task_type=TaskType.OBJECT,
        )

        small_color = np.array(render_predictions.class_color(0))
        large_color = np.array(render_predictions.class_color(1))
        sampled_pixel = rendered[16, 7].astype(np.int16)
        self.assertLess(
            np.linalg.norm(sampled_pixel - small_color),
            np.linalg.norm(sampled_pixel - large_color),
        )

    def test_cli_writes_rendered_image_and_manifest(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            cwd = Path.cwd()
            assets_dir = cwd / "assets"
            assets_dir.mkdir()
            (assets_dir / "yolo26m.pt").write_bytes(b"weights")

            dataset_root = cwd / "dataset"
            image_path = dataset_root / "images" / "val" / "image.jpg"
            label_path = dataset_root / "labels" / "val" / "image.txt"
            image_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            write_image(image_path)
            label_path.write_text(
                "0 0.375 0.375 0.5 0.5\n",
                encoding="utf-8",
            )
            yaml_path = dataset_root / "data.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        f"path: {dataset_root}",
                        "train: images/val",
                        "val: images/val",
                        "names:",
                        "  0: person",
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.object(render_predictions, "YOLO", fake_yolo_class()),
                patch.object(
                    render_predictions,
                    "get_backbone",
                    return_value=[],
                ),
                patch.object(render_predictions, "set_backbone"),
            ):
                result = runner.invoke(
                    render_predictions.main,
                    [
                        "--hydra_model_name",
                        "yolo26m=f11+yolo26m",
                        "--data",
                        str(yaml_path),
                        "--assets_dir",
                        str(assets_dir),
                        "--output_dir",
                        "outputs",
                        "--num-images",
                        "1",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            output_image = (
                cwd
                / "outputs"
                / "yolo26m=f11+yolo26m"
                / "val"
                / "image.jpg"
            )
            manifest_path = (
                cwd / "outputs" / "yolo26m=f11+yolo26m" / "manifest.json"
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
            )
            output_image_exists = output_image.exists()

        self.assertTrue(output_image_exists)
        self.assertEqual(manifest["image_count"], 1)
        self.assertEqual(manifest["prediction_count"], 1)
        self.assertEqual(manifest["kept_prediction_count"], 1)

    def test_cli_source_mode_writes_same_image_for_each_head(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            cwd = Path.cwd()
            assets_dir = cwd / "assets"
            assets_dir.mkdir()
            (assets_dir / "yolo26m.pt").write_bytes(b"weights")
            (assets_dir / "yolo26m-pose.pt").write_bytes(b"weights")
            source_dir = cwd / "source"
            source_dir.mkdir()
            write_image(source_dir / "image.jpg")

            with (
                patch.object(render_predictions, "YOLO", fake_yolo_class()),
                patch.object(
                    render_predictions,
                    "get_backbone",
                    return_value=[],
                ),
                patch.object(render_predictions, "set_backbone"),
            ):
                result = runner.invoke(
                    render_predictions.main,
                    [
                        "--hydra_model_name",
                        "yolo26m=f11+yolo26m+yolo26m-pose",
                        "--source-image-dir",
                        str(source_dir),
                        "--assets_dir",
                        str(assets_dir),
                        "--output_dir",
                        "outputs",
                        "--num-images",
                        "1",
                    ],
                )

            object_manifest_path = (
                cwd / "outputs" / "yolo26m=f11+yolo26m" / "manifest.json"
            )
            pose_manifest_path = (
                cwd / "outputs" / "yolo26m=f11+yolo26m-pose" / "manifest.json"
            )
            object_manifest = json.loads(
                object_manifest_path.read_text(encoding="utf-8"),
            )
            object_output_exists = (
                cwd
                / "outputs"
                / "yolo26m=f11+yolo26m"
                / "source"
                / "image.jpg"
            ).exists()
            pose_output_exists = (
                cwd
                / "outputs"
                / "yolo26m=f11+yolo26m-pose"
                / "source"
                / "image.jpg"
            ).exists()
            pose_manifest_exists = pose_manifest_path.exists()

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(object_output_exists)
        self.assertTrue(pose_output_exists)
        self.assertTrue(pose_manifest_exists)
        self.assertEqual(object_manifest["mode"], "source")
        self.assertIsNone(object_manifest["config"]["iou_threshold"])

    def test_cli_source_mode_rejects_iou_threshold(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            source_dir = Path("source")
            source_dir.mkdir()

            result = runner.invoke(
                render_predictions.main,
                [
                    "--hydra_model_name",
                    "yolo26m=f11+yolo26m",
                    "--source-image-dir",
                    str(source_dir),
                    "--iou-threshold",
                    "0.5",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "--source-image-dir and --iou-threshold are mutually exclusive",
            result.output,
        )


def write_image(path: Path) -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def fake_yolo_class() -> type:
    class FakeYOLO:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.model = SimpleNamespace()
            self.names = {0: "person"}

        def eval(self) -> None:
            return None

        def predict(
            self,
            source: list[str],
            **kwargs: object,
        ) -> object:
            self.kwargs = kwargs
            image_path = Path(source[0])
            image = cv2.imread(str(image_path))
            result = SimpleNamespace(
                boxes=SimpleNamespace(
                    data=np.array(
                        [[10, 10, 50, 50, 0.9, 0]],
                        dtype=np.float32,
                    )
                ),
                masks=None,
                keypoints=None,
                names={0: "person"},
                orig_img=image,
                orig_shape=image.shape[:2],
                path=str(image_path),
            )
            return iter([result])

    return FakeYOLO
