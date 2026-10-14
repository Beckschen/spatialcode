"""CPU regression tests for the public encoder adapter (no weights/downloads)."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Train" / "RL"))

from spatialcode import SceneCode, SpatialEncoder, image_to_code, video_to_code
from spatialcode.encoder import get_encoder
from spatialcode.encoder.spatial_encoder import as_rgb_frames
from spatialcode.pipeline.video_pipeline import _load_video


class FakePredictor:
    image_size = 1024

    def __init__(self, fail=False):
        self.prompts = []
        self.state = {"features": object()}
        self.fail = fail

    def init_state(self, video_frames):
        return self.state

    def add_new_points_or_box(self, state, **kwargs):
        self.prompts.append(kwargs)

    def propagate_in_video(self, state):
        if self.fail:
            raise RuntimeError("inference failed")
        # Same object seen by two translated/rotated cameras. Both camera
        # translation and object coordinates use the checkpoint's 1/2.5 scale.
        extrinsics = [
            [[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, -1, 0, 0], [1, 0, 0, 3], [0, 0, 1, 0]],
        ]
        centers = [[1, 0, 2], [0, 0, 2]]
        quats = [[1, 0, 0, 0], [2**-0.5, 0, 0, 2**-0.5]]  # wxyz
        for frame_idx in range(2):
            yield {
                "frame_idx": frame_idx,
                "obj_ids": [0, 1],
                "has_object": torch.tensor([True, False]),
                "extrinsics": torch.tensor([extrinsics[frame_idx]], dtype=torch.float32),
                "all_pred_bbox_3d": torch.tensor([
                    centers[frame_idx] + [1, 2, 3, 3], [0, 0, 1, 1, 1, 1, 0],
                ], dtype=torch.float32),
                "all_pred_bbox_quat": torch.tensor([quats[frame_idx], [1, 0, 0, 0]]),
            }


class EncoderApiTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.prompts = [
            {"label": "chair", "bbox": [20, 10, 80, 50]},
            {"label": "table", "bbox": [100, 20, 150, 70]},
        ]

    def test_merge_transforms_camera_translation_rotation_and_scale(self):
        predictor = FakePredictor()
        encoder = SpatialEncoder(predictor)
        with patch.object(encoder, "_prepare_video", return_value={}):
            scene = encoder.encode_video([self.frame, self.frame], self.prompts)
        self.assertIsInstance(scene, SceneCode)
        self.assertEqual(len(scene), 1)  # Invisible prompts are not hallucinated.
        self.assertEqual(scene[0].label, "chair")
        self.assertEqual(scene[0].track_id, 0)
        self.assertEqual(scene[0].frame_ids, [0, 1])
        self.assertEqual(scene[0].confidence, 0.75)
        np.testing.assert_allclose(scene[0].position, [2.5, 0, 5], atol=1e-6)
        np.testing.assert_allclose(scene[0].size, [2.5, 5, 7.5])
        np.testing.assert_allclose(scene[0].orientation, [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(predictor.prompts[0]["box"], [102.4, 102.4, 409.6, 512])
        np.testing.assert_allclose(predictor.prompts[0]["points"], [[256, 307.2]])
        self.assertEqual(scene.metadata["coordinate_frame"], "first_camera")
        self.assertEqual(scene.metadata["units"], "meters")
        self.assertEqual(len(scene.to_bbox_list()[0]["bbox_3d"]), 9)
        self.assertEqual(predictor.state, {})

    def test_inference_failure_releases_state(self):
        predictor = FakePredictor(fail=True)
        encoder = SpatialEncoder(predictor)
        with patch.object(encoder, "_prepare_video", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                encoder.encode_video([self.frame], self.prompts)
        self.assertEqual(predictor.state, {})

    def test_invalid_prompts_fail_before_model_load(self):
        for fn, value, module in [
            (video_to_code, [self.frame], "video_pipeline"),
            (image_to_code, self.frame, "image_pipeline"),
        ]:
            with patch(f"spatialcode.pipeline.{module}._get_encoder") as loader:
                for prompts in [None, [], [{"label": "chair", "bbox": [-1, 0, 10, 10]}],
                                [{"label": "", "bbox": [0, 0, 10, 10]}],
                                [{"label": "chair", "bbox": [0, 0, float("nan"), 10]}]]:
                    with self.subTest(fn=fn.__name__, prompts=prompts):
                        with self.assertRaises(ValueError):
                            fn(value, object_prompts=prompts)
                loader.assert_not_called()

    def test_public_pipelines_share_model_and_convert_inputs(self):
        get_encoder.cache_clear()
        self.addCleanup(get_encoder.cache_clear)
        with patch.object(SpatialEncoder, "from_pretrained") as load:
            encoder = load.return_value
            video_to_code(torch.zeros(5, 3, 100, 200, dtype=torch.bfloat16),
                          max_frames=2, object_prompts=self.prompts, device="cpu")
            image_to_code(torch.zeros(3, 100, 200), object_prompts=self.prompts, device="cpu")
            load.assert_called_once_with("qicq1c/SpatialCode", device="cpu")
            frames = encoder.encode_video.call_args.args[0]
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].shape, (100, 200, 3))
            self.assertEqual(frames[0].dtype, np.uint8)
            self.assertEqual(encoder.encode_image.call_args.args[0].shape, (100, 200, 3))

    def test_bad_frames_and_sampling(self):
        for frames in [[], np.array(1), torch.zeros(2, 4, 8, 8),
                       [self.frame, self.frame[:50]], [np.ones((4, 4, 3)) * 255],
                       [np.full((4, 4, 3), np.nan)]]:
            with self.subTest(shape=str(type(frames))):
                with self.assertRaises(ValueError):
                    as_rgb_frames(frames)
        for kwargs in [{"max_frames": 0}, {"max_frames": True}, {"fps": 0}, {"fps": float("nan")}]:
            with self.assertRaises(ValueError):
                video_to_code([self.frame], object_prompts=self.prompts, **kwargs)

    def test_missing_local_checkpoint_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                SpatialEncoder.from_pretrained(directory, device="cpu")

    def test_local_model_config_works_without_training_environment_variables(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        backend = types.ModuleType("modeling.build_sam")
        predictor = FakePredictor()
        root = Path(__file__).resolve().parents[1] / "Train/SpatialEncoder/modeling"
        with tempfile.TemporaryDirectory(prefix="spatialcode weights ") as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            sam_checkpoint = Path(directory) / "sam2.1_hiera_large.pt"
            checkpoint.touch()
            sam_checkpoint.touch()

            def build(config_name, ckpt_path, sam_checkpoint, device, hydra_overrides_extra):
                with initialize_config_dir(config_dir=str(root), version_base="1.2"):
                    config = compose(config_name=config_name, overrides=hydra_overrides_extra)
                    OmegaConf.resolve(config)
                self.assertEqual(config.model.sam_weight_path, sam_checkpoint)
                self.assertEqual(ckpt_path, str(checkpoint))
                self.assertEqual(device, "cpu")
                return predictor

            backend.build_sam2_video_predictor = build
            with patch.dict(sys.modules, {"modeling.build_sam": backend}), patch.dict(os.environ):
                os.environ.pop("BOX_WEIGHTS_PATH", None)
                encoder = SpatialEncoder.from_pretrained(directory, device="cpu")
                self.assertIs(encoder.predictor, predictor)

    def test_video_file_sampling_and_rgb_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sample.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(12):
                    frame = np.zeros((32, 32, 3), dtype=np.uint8)
                    frame[:] = (i * 10, 0, 200)  # BGR
                    writer.write(frame)
            finally:
                writer.release()
            frames = _load_video(path, fps=2, max_frames=3)
            self.assertEqual(len(frames), 3)
            np.testing.assert_allclose([f[16, 16, 2] for f in frames], [0, 50, 100], atol=5)
            np.testing.assert_allclose([f[16, 16, 0] for f in frames], 200, atol=5)
            consecutive = _load_video(path, max_frames=2)
            self.assertEqual(len(consecutive), 2)
            self.assertLess(abs(int(consecutive[1][16, 16, 2]) - 10), 5)
            with self.assertRaises(FileNotFoundError):
                _load_video(str(Path(directory) / "missing.mp4"))
            bad = Path(directory) / "invalid.mp4"
            bad.write_bytes(b"")
            with self.assertRaises(ValueError):
                _load_video(str(bad))


if __name__ == "__main__":
    unittest.main()
