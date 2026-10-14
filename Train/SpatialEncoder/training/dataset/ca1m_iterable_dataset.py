import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Dict
import os
import random
import pickle
from collections import deque
import functools
from glob import glob
import io
import numpy as np
from PIL import Image
import ast
from skimage.transform import resize
import cv2
import webdataset as wds
from scipy.spatial.transform import Rotation as R

from .base_dataset import (
    filter_object,
    NOT_APPEAR_FLAG,
    closed_form_inverse_se3,
    get_alpha_from_bbox_2d,
    DATASET_LOOKUP_TABLE,
    remap_axes,
    project_to_so3,
)
from training.utils.distributed import get_rank, get_world_size


cv2.setNumThreads(0)
torch.set_num_threads(1)

DATASET_FILENAMES = {
    "CA-1M": {"train": "train/ca1m-train-{}.tar", "val": "val/ca1m-val-{}.tar"},
}


def read_image_bytes(image_bytes, expected_size, channels_first=True):
    if image_bytes.startswith(b"\x89PNG"):
        # PNG.
        image = np.array(Image.open(io.BytesIO(image_bytes)))
    else:
        raise ValueError("Unknown image format")

    assert (image.shape[1], image.shape[0]) == expected_size

    if channels_first and (image.ndim > 2):
        image = np.moveaxis(image, -1, 0)
    return image


def group_by_clip_ca1m(samples, clip_len=15, meta_files: List[Dict] = None):
    window = deque()
    last_vid = None
    meta_file = None
    for sample in samples:
        if "fname" not in sample:  # 跳过没有 fname 的 entry
            continue
        key = sample["fname"]

        if (
            not key.endswith("wide/image.png")
            and not key.endswith("wide/depth.png")
            and not key.endswith("_wide/depth/size")
            and not key.endswith("_wide/image/size")
        ):
            continue
        vid, ts_modality = key.split("/", 1)
        ts = ts_modality.split(".")[0]

        if meta_file is None or vid != last_vid:
            meta_file = [_ for _ in meta_files if vid in _][0]
            with open(meta_file, "rb") as f:
                meta_file = pickle.load(f)

        valid_frames = meta_file["frames"]
        if ts not in valid_frames:
            continue

        # 如果换了视频，就清空窗口
        if last_vid is not None and vid != last_vid:
            window.clear()
        last_vid = vid

        # 把当前帧存到窗口
        window.append({"ts": ts, "key": key, "data": sample})

        # 如果窗口够长了，就 yield 出一个 clip
        if len(window) == clip_len * 4:  # 每个 clip 包含 clip_len 帧
            yield {
                "video_id": vid,
                "meta_file": meta_file,
                "frames": list(window),
                "name": "CA-1M",
            }
            # window.clear()
            for _ in range(random.randint(5, 15) * 4):
                window.popleft()  # 滑动窗口


DATASET_COLLECT_FN = {
    "CA-1M": group_by_clip_ca1m,
}


class CA1MTrainDataset(IterableDataset):
    def __init__(
        self,
        meta_dir,
        dataset_name: str,
        default_scale: float = 1.0,
        base_dir: str = ".",
        training: bool = True,
        resolution_aug_scale: List[float] = [0.8, 1.2],
        predict_metric: bool = False,
        norm_scale: float = 1.0,
        spatial_type: str = "vggt",
        resolution: int = 1024,
        spatial_resolution: int = 518,
        train_transforms=None,
        spatial_transforms=None,
        video_split_per_frame: int = 20,
        frame_num_range: List[int] = [20, 64],
        frame_sample_gap_range: List[int] = [1, 3],
        max_num_objects: int = 20,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn=None,
        sample_videos_for_ab=100,
        ablation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.training = training
        self.aug_scale = resolution_aug_scale if training else None
        self.predict_metric = predict_metric
        self.resolution = resolution
        self.spatial_resolution = spatial_resolution
        self.frame_num_range = frame_num_range
        self.frame_sample_gap_range = frame_sample_gap_range
        self.max_num_objects = max_num_objects
        self.train_transforms = train_transforms
        self.spatial_transforms = spatial_transforms
        self.video_split_per_frame = video_split_per_frame
        self.norm_scale = norm_scale
        self.spatial_type = spatial_type
        # =========
        # container for dataloader
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # ========
        self.video_ids = []
        self.names = [dataset_name]

        self.meta_files = []
        self.metric_scales = {dataset_name: default_scale}
        name = dataset_name
        meta_path = sorted(glob(os.path.join(meta_dir, name, "*.pkl")))
        if ablation:
            if self.training:
                meta_path = meta_path[:sample_videos_for_ab]
        if self.training:
            meta_path = [_ for _ in meta_path if "train" in _]
        else:
            meta_path = [_ for _ in meta_path if "val" in _]
        self.meta_files += meta_path
        self.video_ids += [
            (name, os.path.basename(p).split(".pkl")[0].split("-")[-1])
            for p in meta_path
        ]

        self.length = 100 * num_workers  # an arbitrary large number
        # ==========
        if self.spatial_type == "depthanythingv3":
            from depth_anything_3.utils.io.input_processor import InputProcessor
            from depth_anything_3.utils.logger import logger

            self.input_processor = InputProcessor()
            logger.info(f"Using spatial type: {self.spatial_type}")

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rank = get_rank()
        world_size = get_world_size()

        if self.shuffle:
            random.shuffle(self.video_ids)

        # ==========
        # prepare urls and get webdataset pipeline
        urls = {k: [] for k in self.names}
        for name, video_id in self.video_ids:
            if self.training:
                url = os.path.join(
                    self.base_dir,
                    DATASET_FILENAMES[name]["train"].format(video_id),
                )
            else:
                url = os.path.join(
                    self.base_dir,
                    DATASET_FILENAMES[name]["val"].format(video_id),
                )
            urls[name].append(url)

        if worker_info is not None:
            global_worker_id = rank * worker_info.num_workers + worker_info.id
            total_workers = world_size * worker_info.num_workers
        else:
            global_worker_id = rank
            total_workers = world_size

        urls = {k: v[global_worker_id::total_workers] for k, v in urls.items()}
        pipelines = []
        for k, url in urls.items():
            collect_fn = DATASET_COLLECT_FN[k]
            ds = wds.DataPipeline(
                wds.SimpleShardList(url),
                wds.tariterators.url_opener,
                wds.tariterators.tar_file_expander,
                functools.partial(
                    collect_fn,
                    clip_len=self.video_split_per_frame,
                    meta_files=self.meta_files,
                ),  # 传参数,
            )
            pipelines.append(ds)
        # if self.shuffle:
        weights = [1.0 / len(pipelines)] * len(pipelines)
        dataset = wds.RandomMix(pipelines, weights)
        # else:
        #     dataset = wds.ConcatDataset(pipelines)

        count = 0
        target = max(self.length // worker_info.num_workers, 1)
        dataset_iter = iter(dataset)
        while count < target:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                dataset_iter = iter(dataset)  # 重新开一轮
                sample = next(dataset_iter)
            video_id = sample["video_id"]
            meta_file = sample["meta_file"]
            name = sample["name"]
            metric_scale = self.metric_scales[name]
            frame_list = self._load_webdataset_to_numpy(meta_file, sample["frames"])
            data_dict = self._prepare_detection_data(
                meta_file, frame_list, metric_scale
            )
            if data_dict == {}:
                continue
            if self.train_transforms:
                rgb_image = data_dict["rgb_images"]
                data_dict["rgb_images"] = self.train_transforms(rgb_image)
            if self.spatial_transforms:
                spatial_image = data_dict["spatial_images"]
                data_dict["spatial_images"] = self.spatial_transforms(spatial_image)
            data_dict = self._totensor(data_dict)
            data_dict["dataset_id"] = DATASET_LOOKUP_TABLE[name]
            yield data_dict
            count += 1

    #########
    def get_dataloader(self, **kwargs):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=False,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = kwargs.get("prefetch_factor", 2)
        return DataLoader(self, **loader_kwargs)

    def _totensor(self, dict):
        for key in dict:
            if isinstance(dict[key], np.ndarray):
                dict[key] = torch.from_numpy(dict[key])
        return dict

    #############
    # code for loading data
    def _load_webdataset_to_numpy(self, meta_file: Dict, frames: List[Dict]):
        frame_dict = {}
        for ts_data in sorted(frames, key=lambda x: x["ts"]):
            ts = ts_data["ts"]
            if ts not in frame_dict:
                frame_dict[ts] = {}

            key = ts_data["key"]
            if key.endswith("image.png"):
                frame_dict[ts]["image"] = ts_data["data"]
            elif key.endswith("depth.png"):
                frame_dict[ts]["depth"] = ts_data["data"]
            elif key.endswith("image/size"):
                frame_dict[ts]["image_size"] = ts_data["data"]
            elif key.endswith("depth/size"):
                frame_dict[ts]["depth_size"] = ts_data["data"]
            else:
                raise ValueError(f"Unknown key {key}")
        for ts in frame_dict:
            assert "image" in frame_dict[ts], f"Image not found for ts {ts}"
            assert "depth" in frame_dict[ts], f"Depth not found for ts {ts}"
            assert "image_size" in frame_dict[ts], f"Image size not found for ts {ts}"
            assert "depth_size" in frame_dict[ts], f"Depth size not found for ts {ts}"
        # load frames in the order of meta_file["frames"]
        frame_list = []
        for ts in meta_file["frames"]:
            if ts not in frame_dict:
                continue
            frame_data = frame_dict[ts]

            image_bytes = frame_data["image"]["data"]
            depth_bytes = frame_data["depth"]["data"]
            image_size = frame_data["image_size"]["data"].decode("utf-8")
            image_size = tuple(ast.literal_eval(image_size))  # (W, H)
            depth_size = frame_data["depth_size"]["data"].decode("utf-8")
            depth_size = tuple(ast.literal_eval(depth_size))  # (W, H)

            image = read_image_bytes(
                image_bytes, expected_size=image_size, channels_first=False
            )
            depth = read_image_bytes(
                depth_bytes, expected_size=depth_size, channels_first=False
            )
            # depth = np.expand_dims(depth, axis=0)  # (1, H, W)
            frame_list.append({"ts": ts, "image": image, "depth": depth})
        return frame_list

    def _prepare_detection_data(
        self, meta_file: Dict, frames: List[Dict], metric_scale: float
    ):
        # decide frame_num and sample frames
        frame_num = random.randint(self.frame_num_range[0], self.frame_num_range[1])
        all_frame_num = len(frames)
        if all_frame_num > frame_num:
            frame_sample_gap = random.randint(
                self.frame_sample_gap_range[0], self.frame_sample_gap_range[1]
            )
            frames = frames[::frame_sample_gap]
            if len(frames) > frame_num:
                start_idx = random.sample(range(len(frames) - frame_num), 1)[0]
                frames = frames[start_idx : start_idx + frame_num]
        self.max_num_objects = min(
            self.max_num_objects, 50 // len(frames)
        )  # ensure memory fit

        # decide objects based on objects in the first frame
        if len(frames) == 0:
            return {}
        first_frame_id = frames[0]["ts"]
        objects = meta_file["objects"]
        objects_in_frame = {
            k: v
            for k, v in objects.items()
            if (v.get(first_frame_id, None) is not None)
        }
        # first_frame= frames[0]["image"]
        # original_resolution = first_frame.shape[:2] # (H, W)
        objects_in_frame_ = filter_object(
            objects_in_frame, first_frame_id, resolution=(1024, 1024)
        )

        # 如果第一帧没有物体，就重新采样一个视频
        if len(objects_in_frame_) == 0:
            return {}
        objects_in_frame = objects_in_frame_
        if len(objects_in_frame.values()) > self.max_num_objects:
            selected_objs = random.sample(
                list(objects_in_frame.keys()), self.max_num_objects
            )
        elif len(objects_in_frame.values()) < self.max_num_objects:
            # randomly duplicate some objects to reach max_num_objects
            selected_objs = list(objects_in_frame.keys())
            while len(selected_objs) < self.max_num_objects:
                selected_objs.append(random.choice(selected_objs))
        else:
            selected_objs = list(objects_in_frame.keys())
        # load meta data
        if not self.predict_metric:
            distance_norm = meta_file.get("pcd_average_distance", 1.0)
        else:
            distance_norm = self.norm_scale
        original_frame_size = None
        video_bboxes_2d = []
        video_bboxes_center_3d = []
        video_bboxes_dim_3d = []
        video_bboxes_yaw_3d = []
        video_bboxes_R_3d = []
        video_rgb_images = []
        video_depth_images = []
        video_intrinsics = []
        video_spatial_images = []
        # video_spatial_intrinsics = []
        extrinsics = []
        T_gravities = []
        object_ids = []
        object_appearance = []
        for frame in frames:
            frame_id = frame["ts"]
            rgb_image = frame["image"]
            depth_image = frame["depth"].astype(np.float32)  # (1, H, W)
            depth_image[np.isnan(depth_image)] = 0
            depth_image[np.isinf(depth_image)] = 0
            depth_image = resize(
                depth_image,
                (rgb_image.shape[0], rgb_image.shape[1]),
                order=1,  # 1 = 双线性插值
                mode="edge",
                anti_aliasing=False,
                preserve_range=True,  # 保持原始数值范围（避免归一化到0~1）
            )
            depth_image = depth_image / metric_scale
            depth_image = depth_image / distance_norm

            frame_meta = meta_file["frames"][frame_id]
            intrinsic = np.array(frame_meta["rgb_intrinsic"])[0]
            camera_pose = frame_meta.get("camera_pose", None)
            if camera_pose is not None:
                camera_pose = np.array(camera_pose)
            else:
                camera_pose = np.eye(4)
            extrinsic = np.linalg.inv(camera_pose)
            T_gravity = np.array(frame_meta["T_gravity"])
            if original_frame_size is None:
                original_frame_size = rgb_image.shape[:2]

            # load object annotations
            object_bboxes_2d = []
            object_bboxes_center_3d = []
            object_bboxes_dim_3d = []
            object_bboxes_yaw_3d = []
            object_bboxes_R_3d = []
            object_id_frame = []
            object_appearance_frame = []
            for k in selected_objs:
                obj = objects_in_frame[k]
                # record object id for objects appeared in the first frame
                object_id_frame.append(k)

                has_object = obj.get(frame_id, None) is not None
                if has_object:
                    bbox2d = np.array(obj[frame_id]["camera_box_2d_rend"])
                    w_2d = bbox2d[2] - bbox2d[0]
                    h_2d = bbox2d[3] - bbox2d[1]
                    area = w_2d * h_2d
                    if area < 100 and w_2d < 20 and h_2d < 20:
                        has_object = False
                if not has_object:
                    object_appearance_frame.append(False)
                    object_bboxes_2d.append(
                        NOT_APPEAR_FLAG * np.ones((4,), dtype=np.float32)
                    )
                    object_bboxes_center_3d.append(
                        NOT_APPEAR_FLAG * np.ones((3,), dtype=np.float32)
                    )
                    object_bboxes_dim_3d.append(
                        NOT_APPEAR_FLAG * np.ones((3,), dtype=np.float32)
                    )
                    object_bboxes_yaw_3d.append(NOT_APPEAR_FLAG)
                    object_bboxes_R_3d.append(
                        NOT_APPEAR_FLAG * np.ones((3, 3), dtype=np.float32)
                    )
                    continue
                object_appearance_frame.append(True)
                object_bboxes_2d.append(bbox2d)
                object_bboxes_center_3d.append(
                    np.array(obj[frame_id]["camera_center"]) / distance_norm
                )
                camera_scale = np.array(obj[frame_id]["camera_scale"]) / distance_norm
                rotmat = np.array(obj[frame_id]["camera_R"])
                rotmat = project_to_so3(rotmat)
                # if name == "CA-1M":
                #     rotmat, camera_scale = remap_axes(
                #         rotmat, camera_scale, perm=(0, 2, 1), signs=(1, 1, -1)
                #     )
                object_bboxes_dim_3d.append(camera_scale)
                # the axis definition is different from bbox_yaw_3d
                yaw = np.arctan2(-rotmat[0, 2], rotmat[2, 2])
                object_bboxes_yaw_3d.append(yaw)
                object_bboxes_R_3d.append(rotmat)
            object_ids.append(object_id_frame)
            object_appearance.append(np.array(object_appearance_frame))
            object_bboxes_2d_np = np.stack(object_bboxes_2d, axis=0).astype(np.float32)
            object_bboxes_center_3d_np = np.stack(
                object_bboxes_center_3d, axis=0
            ).astype(np.float32)
            object_bboxes_dim_3d_np = np.stack(object_bboxes_dim_3d, axis=0).astype(
                np.float32
            )
            object_bboxes_yaw_3d_np = np.stack(object_bboxes_yaw_3d, axis=0).astype(
                np.float32
            )
            object_bboxes_R_3d_np = np.stack(object_bboxes_R_3d, axis=0).astype(
                np.float32
            )
            rgb_image, depth_image, intrinsic, object_bboxes_2d_np = (
                self._crop_image_depth_and_intrinsic_by_pp(
                    rgb_image, depth_image, intrinsic, object_bboxes_2d_np
                )
            )
            rgb_image, depth_image, intrinsic, object_bboxes_2d_np = (
                self._resize_image_depth_and_intrinsic(
                    rgb_image,
                    depth_image,
                    intrinsic,
                    object_bboxes_2d_np,
                    target_shape=self.resolution,
                    pixel_center=True,
                    square=True,
                )
            )
            if self.spatial_type == "vggt":
                spatial_image = cv2.resize(
                    rgb_image.copy(),
                    (self.spatial_resolution, self.spatial_resolution),
                    interpolation=cv2.INTER_LINEAR,
                )
            elif self.spatial_type == "mapanything":
                # original implementation is actually a list of dicts, where each dict represents a frame
                # here we just process single frame and aggregate the list outside
                from .dataset_utils import load_images

                spatial_image = load_images(
                    Image.fromarray(rgb_image.copy()),
                    resize_mode="square",
                    size=self.spatial_resolution,
                )[0]
            elif self.spatial_type == "depthanythingv3":
                # a placeholder, actual spatial image will be processed outside
                spatial_image = rgb_image.copy()
            else:
                raise NotImplementedError
            # spa_to_ori = self.spatial_resolution / self.resolution
            # spatial_intrinsic = self._resize_intrinsic(
            #     intrinsic, (spa_to_ori, spa_to_ori), pixel_center=True
            # )
            video_bboxes_2d.append(object_bboxes_2d_np)
            video_bboxes_center_3d.append(object_bboxes_center_3d_np)
            video_bboxes_dim_3d.append(object_bboxes_dim_3d_np)
            video_bboxes_yaw_3d.append(object_bboxes_yaw_3d_np)
            video_bboxes_R_3d.append(object_bboxes_R_3d_np)
            video_rgb_images.append(rgb_image)  # (H, W, 3)
            video_depth_images.append(depth_image)
            video_intrinsics.append(intrinsic)
            video_spatial_images.append(spatial_image)  # (H', W', 3)
            # video_spatial_intrinsics.append(spatial_intrinsic)
            extrinsics.append(extrinsic)
            T_gravities.append(T_gravity)
        object_appearance = np.stack(object_appearance, axis=0)  # (N, M) boolean
        video_bboxes_2d = np.stack(video_bboxes_2d, axis=0)  # (N, M, 4)
        video_bboxes_center_3d = np.stack(video_bboxes_center_3d, axis=0)  # (N, M, 3)
        video_bboxes_dim_3d = np.stack(video_bboxes_dim_3d, axis=0)  # (N, M, 3)
        video_bboxes_yaw_3d = np.stack(video_bboxes_yaw_3d, axis=0)  # (N, M)
        video_bboxes_R_3d = np.stack(video_bboxes_R_3d, axis=0)  # (N, M, 3, 3)
        video_rgb_images = np.stack(video_rgb_images, axis=0)  # (N, H, W, 3)
        video_depth_images = np.stack(video_depth_images, axis=0)  # (N, H, W)
        video_intrinsics = np.stack(video_intrinsics, axis=0)  # (N, 3, 3)
        ############
        if self.spatial_type == "vggt":
            video_spatial_images = np.stack(
                video_spatial_images, axis=0
            )  # (N, H', W', 3)
        elif self.spatial_type == "mapanything":
            pass
        elif self.spatial_type == "depthanythingv3":
            from .dataset_utils import preprocess_inputs, prepare_model_inputs

            video_spatial_images, _, _ = preprocess_inputs(
                video_spatial_images,
                process_res=self.spatial_resolution,
                input_processor=self.input_processor,
            )
            video_spatial_images, _, _ = prepare_model_inputs(
                video_spatial_images,
                extrinsics=None,
                intrinsics=None,
                device="cpu",
            )
            # after this, video_spatial_images is a tensor: bs, num_views, 3, h, w
            # we squeeze batch dim here and leave it for later collate_fn or forward to handle
            video_spatial_images = video_spatial_images.squeeze(0)
        # video_spatial_intrinsics = np.stack(
        #     video_spatial_intrinsics, axis=0
        # )  # (N, 3, 3)
        T_gravities = np.stack(T_gravities, axis=0)  # (N, 4, 4)

        extrinsics = np.stack(extrinsics, axis=0)  # (N, 4, 4)
        first_frame_extrinsic = extrinsics[0:1]
        extrinsics = self._normalize_camera_poses(extrinsics, dist_norm=distance_norm)
        
        return {
            "scene_id": meta_file["scene_id"],
            "original_size": np.array(original_frame_size, dtype=np.int32),
            "object_ids": object_ids,
            "object_appearance": object_appearance,
            "bboxes_2d": video_bboxes_2d,
            "bboxes_center_3d": video_bboxes_center_3d,
            "bboxes_dim_3d": video_bboxes_dim_3d,
            "bboxes_alpha_3d": get_alpha_from_bbox_2d(
                video_bboxes_2d, video_intrinsics, video_bboxes_yaw_3d
            ),
            "bboxes_yaw_3d": video_bboxes_yaw_3d,
            "bboxes_R_3d": video_bboxes_R_3d,
            "rgb_images": video_rgb_images,
            "depth_images": video_depth_images,
            "intrinsics": video_intrinsics,
            "spatial_images": video_spatial_images,
            # "spatial_intrinsics": video_spatial_intrinsics,
            "extrinsics": extrinsics,
            "first_frame_extrinsic": first_frame_extrinsic,
            "T_gravities": T_gravities,
            "metric_scale": metric_scale,
            "distance_norm": distance_norm,
        }

    ################
    # code for aggregate boxes in a clip
    ################
    def _aggregate_boxes_for_clip(
        self,
        video_bboxes_center_3d,
        video_bboxes_dim_3d,
        video_bboxes_R_3d,
        extrinsics,
    ):
        import open3d as o3d
        from training.dataset.dataset_utils import compute_3d_bbox_vertices

        N, M, _ = video_bboxes_center_3d.shape

        agg_xyz = np.zeros((N, M, 3), dtype=np.float32)
        agg_dim = np.zeros((N, M, 3), dtype=np.float32)
        agg_R = np.zeros((N, M, 3, 3), dtype=np.float32)

        # world transform of frame0
        extr0 = extrinsics[0]
        R_wc0 = extr0[:3, :3].T
        t_wc0 = -R_wc0 @ extr0[:3, 3]

        for obj_idx in range(M):
            cumulative_corners = []

            # 物体在 frame0 的旋转（世界坐标）
            R_obj0_cam = video_bboxes_R_3d[0, obj_idx]
            R_ref = R_wc0 @ R_obj0_cam

            for frame_idx in range(N):
                box_xyz_cam = video_bboxes_center_3d[frame_idx, obj_idx]
                if (box_xyz_cam < 0).all():
                    continue

                box_dim = video_bboxes_dim_3d[frame_idx, obj_idx]
                box_R_cam = video_bboxes_R_3d[frame_idx, obj_idx]

                extr = extrinsics[frame_idx]
                R_wc = extr[:3, :3].T
                t_wc = -R_wc @ extr[:3, 3]

                # camera → world
                box_xyz_world = R_wc @ box_xyz_cam + t_wc
                box_R_world = R_wc @ box_R_cam

                # 计算世界系角点
                corners_world = compute_3d_bbox_vertices(
                    box_xyz_world[0],
                    box_xyz_world[1],
                    box_xyz_world[2],
                    box_dim[0],
                    box_dim[1],
                    box_dim[2],
                    yaw=0,
                    rotation_matrix=box_R_world,
                )[0]

                cumulative_corners.append(corners_world.reshape(-1, 3))
                all_pts = np.concatenate(cumulative_corners, axis=0)

                # ----- 使用 Open3D 计算最小外接盒尺寸（但不采用其旋转） -----
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                obb = pcd.get_oriented_bounding_box()
                obb_center_world = np.asarray(obb.center)

                # ----- 使用 reference rotation (frame0 rotation in world) -----
                # 将所有点投影到 R_ref 的 local frame 中，重新计算尺寸，确保方向不变
                pts_local = (R_ref.T @ (all_pts - obb_center_world).T).T
                min_l = pts_local.min(axis=0)
                max_l = pts_local.max(axis=0)

                center_local = (min_l + max_l) / 2
                dim_local = max_l - min_l
                center_world = (R_ref @ center_local) + obb_center_world

                # 保存
                agg_xyz[frame_idx, obj_idx] = extr[:3, :3] @ center_world + extr[:3, 3]
                agg_dim[frame_idx, obj_idx] = dim_local
                agg_R[frame_idx, obj_idx] = extr[:3, :3] @ R_ref

        return agg_xyz, agg_dim, agg_R

    ################
    # code for preprocess data
    ################
    def _crop_image_depth_and_intrinsic_by_pp(self, image, depth_map, intrinsic, bbox):
        H, W = image.shape[:2]
        intrinsic = np.copy(intrinsic)

        # aug_size 是 (H', W')
        if self.training and self.aug_scale:
            hs, ws = np.random.uniform(self.aug_scale[0], self.aug_scale[1], 2)
            hs, ws = min(hs, 1.0), min(ws, 1.0)
            target_shape = (int(H * hs), int(W * ws))  # (H', W')
        else:
            target_shape = (H, W)

        cx = intrinsic[0, 2]  # x/u（列方向）
        cy = intrinsic[1, 2]  # y/v（行方向）

        # half_x 用宽度维度，half_y 用高度维度
        half_x = min(target_shape[1] / 2, cx, W - cx)
        half_y = min(target_shape[0] / 2, cy, H - cy)

        start_x = int(cx - half_x)
        start_y = int(cy - half_y)
        end_x = int(start_x + 2 * half_x)
        end_y = int(start_y + 2 * half_y)

        assert 0 <= start_x < end_x <= W
        assert 0 <= start_y < end_y <= H

        # 注意先 y 再 x
        image = image[start_y:end_y, start_x:end_x, :]
        if depth_map is not None:
            depth_map = depth_map[start_y:end_y, start_x:end_x]

        # 主点偏移：cx 减 start_x，cy 减 start_y
        intrinsic[0, 2] = intrinsic[0, 2] - start_x
        intrinsic[1, 2] = intrinsic[1, 2] - start_y

        # bbox 调整：x 用 start_x、宽度 clamp；y 用 start_y、高度 clamp
        Hc, Wc = image.shape[:2]
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]] - start_x, 0, Wc)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]] - start_y, 0, Hc)

        return image, depth_map, intrinsic, bbox

    def _resize_image_depth_and_intrinsic(
        self,
        image,
        depth_map,
        intrinsic,
        bbox,
        target_shape,
        square=True,
        pixel_center=True,
    ):
        # ================
        # adapted from VGGT
        # ================
        original_size = np.array(image.shape[:2])  # (height, width)
        intrinsic = np.copy(intrinsic)

        # Calculate new dimensions while maintaining aspect ratio
        if not square:
            max_dim = np.max(original_size)
            scale = target_shape / max_dim  # Maintain aspect ratio
            new_size = (np.round(original_size * scale)).astype(int)
            scale = scale, scale
        else:
            new_size = (target_shape, target_shape)
            scale = target_shape / original_size[0], target_shape / original_size[1]

        # Convert image to PIL for resizing
        image = Image.fromarray(image)
        image = image.resize(
            (new_size[1], new_size[0]),
            resample=Image.LANCZOS if max(scale) < 1 else Image.BICUBIC,
        )
        image = np.array(image)
        assert image.shape[0] == new_size[0] and image.shape[1] == new_size[1]

        # Resize depth_map
        if depth_map is not None:
            depth_map = cv2.resize(
                depth_map,
                (new_size[1], new_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Update the intrinsic matrix based on the resize scale
        actual_size = np.array(image.shape[:2])
        actual_resize_scale = (
            actual_size[0] / original_size[0],
            actual_size[1] / original_size[1],
        )

        # Adjust intrinsic matrix for pixel center if needed
        intrinsic = self._resize_intrinsic(
            intrinsic, actual_resize_scale, pixel_center=pixel_center
        )
        assert image.shape[:2] == depth_map.shape[:2]

        bbox[:, [0, 2]] = np.clip(
            bbox[:, [0, 2]] * actual_resize_scale[1], 0, new_size[1]
        )
        bbox[:, [1, 3]] = np.clip(
            bbox[:, [1, 3]] * actual_resize_scale[0], 0, new_size[0]
        )
        return image, depth_map, intrinsic, bbox

    def _resize_intrinsic(self, intrinsic, scale, pixel_center=True):
        intrinsic = np.copy(intrinsic)
        sh, sw = scale  # sh = H'/H, sw = W'/W
        if pixel_center:
            intrinsic[0, 2] += 0.5
            intrinsic[1, 2] += 0.5

        # fx/cx 用 sw，fy/cy 用 sh
        intrinsic[0, 0] *= sw  # fx
        intrinsic[1, 1] *= sh  # fy
        intrinsic[0, 2] *= sw  # cx
        intrinsic[1, 2] *= sh  # cy

        if pixel_center:
            intrinsic[0, 2] -= 0.5
            intrinsic[1, 2] -= 0.5
        return intrinsic

    def _normalize_camera_poses(self, extrinsics, dist_norm=1.0):
        """
        extrinsics: S x 3 x 4 numpy array
        dist_norm: float
        """
        S = extrinsics.shape[0]

        # 1. 扩展成 4x4 齐次矩阵
        if not (extrinsics.shape[1] == 4 and extrinsics.shape[2] == 4):
            extrinsics_homog = np.zeros((S, 4, 4), dtype=extrinsics.dtype)
            extrinsics_homog[:, :3, :4] = extrinsics
            extrinsics_homog[:, 3, 3] = 1.0
        else:
            extrinsics_homog = extrinsics

        # 2. 求第一个相机的逆
        first_cam_inv = closed_form_inverse_se3(extrinsics_homog[0:1])
        # 3. 归一化到第一个相机坐标系
        new_extrinsics = np.matmul(extrinsics_homog, first_cam_inv)
        # 4. 缩放平移
        new_extrinsics[:, :3, 3] /= dist_norm
        # 5. 返回 3x4
        return new_extrinsics[:, :3, :]
