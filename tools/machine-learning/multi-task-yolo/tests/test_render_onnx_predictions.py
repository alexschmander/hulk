from __future__ import annotations

from unittest import TestCase

import numpy as np

from validation import render_onnx_predictions

Prediction = render_onnx_predictions.Prediction


class RenderOnnxPredictionsTest(TestCase):
    def test_model_colors_match_twix_overlays(self) -> None:
        colors = render_onnx_predictions.model_class_colors()

        self.assertEqual(
            render_onnx_predictions.CLASS_COLORS_RGB,
            (
                (255, 140, 56),
                (66, 190, 80),
                (67, 112, 255),
                (110, 40, 170),
                (255, 225, 25),
                (40, 202, 255),
                (185, 35, 35),
                (255, 100, 190),
                (0, 109, 119),
            ),
        )
        self.assertEqual(
            tuple(colors),
            render_onnx_predictions.MODEL_CLASS_NAMES,
        )
        self.assertEqual(
            tuple(colors.values()),
            render_onnx_predictions.CLASS_COLORS_BGR,
        )
        self.assertEqual(len(set(colors.values())), len(colors))
        self.assertNotEqual(colors["Ball"], colors["Person pose"])
        self.assertNotEqual(colors["Ball"], colors["Robot pose"])
        self.assertNotEqual(colors["XSpot"], colors["Person pose"])
        self.assertNotEqual(colors["Robot pose"], colors["Person pose"])

    def test_bgr_to_nv12_uses_runtime_input_layout(self) -> None:
        image = np.zeros((64, 96, 3), dtype=np.uint8)

        nv12 = render_onnx_predictions.bgr_to_nv12(image)

        self.assertEqual(nv12.shape, (32, 48, 6))
        self.assertEqual(nv12.dtype, np.uint8)

    def test_auto_provider_prefers_gpu(self) -> None:
        provider = render_onnx_predictions.choose_execution_provider(
            "auto",
            available=("CPUExecutionProvider", "CUDAExecutionProvider"),
        )

        self.assertEqual(provider, "CUDAExecutionProvider")

    def test_label_bounds_avoid_an_occupied_label(self) -> None:
        first = render_onnx_predictions.label_bounds(
            box=(20, 0, 100, 80),
            label_size=(60, 16),
            image_size=(96, 120),
            occupied=(),
        )
        second = render_onnx_predictions.label_bounds(
            box=(24, 0, 104, 80),
            label_size=(60, 16),
            image_size=(96, 120),
            occupied=(first,),
        )

        self.assertFalse(
            render_onnx_predictions.rectangles_overlap(first, second)
        )

    def test_non_maximum_suppression_retains_best_overlap(self) -> None:
        predictions = [
            Prediction(
                cls=0,
                conf=0.9,
                box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            ),
            Prediction(
                cls=0,
                conf=0.7,
                box_xyxy=np.array([12, 12, 48, 48], dtype=np.float32),
            ),
        ]

        retained = render_onnx_predictions.non_maximum_suppression(
            predictions,
            maximum_iou=0.4,
        )

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].conf, 0.9)

    def test_person_pose_overlapping_robot_pose_is_filtered(self) -> None:
        person_pose = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
        )
        robot_pose = Prediction(
            cls=0,
            conf=0.8,
            box_xyxy=np.array([11, 11, 49, 49], dtype=np.float32),
        )

        retained = (
            render_onnx_predictions.filter_person_poses_overlapping_robots(
                [person_pose],
                [robot_pose],
                maximum_iou=0.8,
            )
        )

        self.assertEqual(retained, [])

    def test_person_pose_rendering_draws_square_bounding_box(self) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        prediction = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([10, 10, 80, 80], dtype=np.float32),
            keypoints=keypoints,
        )

        rendered = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[],
            person_poses=[prediction],
            robot_poses=[],
            draw_labels=False,
            keypoint_confidence_threshold=0.25,
        )

        self.assertGreater(int(rendered[10, 10].sum()), 0)
        self.assertGreater(int(rendered[80, 10].sum()), 0)

    def test_robot_pose_rendering_keeps_skeleton_only(self) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        keypoints = np.zeros((14, 3), dtype=np.float32)
        prediction = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([10, 10, 80, 80], dtype=np.float32),
            keypoints=keypoints,
        )

        rendered = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[],
            person_poses=[],
            robot_poses=[prediction],
            draw_labels=False,
            keypoint_confidence_threshold=0.25,
        )

        np.testing.assert_array_equal(rendered, image)

    def test_no_labels_keeps_object_box_without_text(self) -> None:
        image = np.zeros((96, 120, 3), dtype=np.uint8)
        prediction = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([20, 20, 100, 80], dtype=np.float32),
        )

        without_labels = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[prediction],
            person_poses=[],
            robot_poses=[],
            draw_labels=False,
            keypoint_confidence_threshold=0.25,
        )
        with_labels = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[prediction],
            person_poses=[],
            robot_poses=[],
            draw_labels=True,
            keypoint_confidence_threshold=0.25,
        )

        self.assertGreater(int(without_labels.sum()), 0)
        self.assertGreater(int(with_labels.sum()), int(without_labels.sum()))

    def test_object_boxes_have_square_corners(self) -> None:
        image = np.zeros((96, 120, 3), dtype=np.uint8)
        prediction = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([20, 20, 100, 80], dtype=np.float32),
        )

        rendered = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[prediction],
            person_poses=[],
            robot_poses=[],
            draw_labels=False,
            keypoint_confidence_threshold=0.25,
        )

        self.assertGreater(int(rendered[20, 20].sum()), 0)
        self.assertGreater(int(rendered[80, 20].sum()), 0)
        self.assertGreater(int(rendered[80, 30].sum()), 0)

    def test_outside_label_aligns_to_outer_box_stroke(self) -> None:
        label = render_onnx_predictions.outside_label_bounds(
            box=(20, 30, 100, 80),
            label_size=(60, 16),
            image_size=(96, 120),
            stroke_outset=1,
            occupied=(),
        )

        self.assertIsNotNone(label)
        if label is None:
            self.fail("Expected an outside label placement")
        self.assertEqual(label[0], 19)
        self.assertEqual(label[3], 30)
        self.assertFalse(
            render_onnx_predictions.rectangles_overlap(
                label,
                (20, 30, 100, 80),
            )
        )

    def test_outside_label_never_falls_back_inside_box(self) -> None:
        label = render_onnx_predictions.outside_label_bounds(
            box=(0, 0, 80, 80),
            label_size=(60, 16),
            image_size=(96, 120),
            stroke_outset=1,
            occupied=(),
        )

        self.assertIsNotNone(label)
        if label is None:
            self.fail("Expected an outside label placement")
        self.assertFalse(
            render_onnx_predictions.rectangles_overlap(
                label,
                (0, 0, 80, 80),
            )
        )

    def test_output_scale_doubles_width_and_height(self) -> None:
        image = np.zeros((96, 120, 3), dtype=np.uint8)
        prediction = Prediction(
            cls=0,
            conf=0.9,
            box_xyxy=np.array([20, 20, 100, 80], dtype=np.float32),
        )

        rendered = render_onnx_predictions.render_onnx_predictions(
            image,
            objects=[prediction],
            person_poses=[],
            robot_poses=[],
            draw_labels=False,
            keypoint_confidence_threshold=0.25,
            output_scale=2,
        )

        self.assertEqual(rendered.shape, (192, 240, 3))
        self.assertGreater(int(rendered[40, 40].sum()), 0)
        self.assertGreater(int(rendered[160, 40].sum()), 0)

    def test_bounding_box_thickness_scales_sublinearly(self) -> None:
        native = np.zeros((96, 120, 3), dtype=np.uint8)
        scaled = np.zeros((384, 480, 3), dtype=np.uint8)

        native_thickness = render_onnx_predictions.bounding_box_thickness(
            native,
            output_scale=1,
        )
        scaled_thickness = render_onnx_predictions.bounding_box_thickness(
            scaled,
            output_scale=4,
        )

        self.assertEqual(native_thickness, 2)
        self.assertEqual(scaled_thickness, 4)

    def test_visual_elements_scale_sublinearly(self) -> None:
        self.assertEqual(
            render_onnx_predictions.scaled_visual_size(1, 4),
            2,
        )
        self.assertEqual(
            render_onnx_predictions.scaled_visual_size(2, 4),
            4,
        )
        self.assertEqual(
            render_onnx_predictions.scaled_visual_size(4, 4),
            8,
        )
        self.assertEqual(
            render_onnx_predictions.label_font_scale(220, 4),
            render_onnx_predictions.LABEL_MIN_FONT_SCALE * 2,
        )
