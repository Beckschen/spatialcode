from torch.utils.data import Dataset
import os
from glob import glob
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from PIL import Image
from skimage.transform import resize
import PIL
import cv2
import json


from .dataset_utils import (
    load_images,
    rotate_image,
    filter_object,
    NOT_APPEAR_FLAG,
    closed_form_inverse_se3,
    get_alpha_from_bbox_2d,
    DATASET_LOOKUP_TABLE,
    remap_axes,
)
from .path_utils import resolve_existing_path


# from training.utils.transform_utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
def box_xyxy_to_cxcywh(bbox):
    """
    把 (x1, y1, x2, y2) 转换成 (cx, cy, w, h)
    bbox: (..., 4)
    """
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ExampleDataset(Dataset):
    def __init__(
        self,
        sample_meta_path: str,
        name: str = "CA-1M",
        default_scale: float = 1.0,
        base_dir: str = ".",
        predict_metric: bool = False,
        norm_scale: float = 1.0,
        resolution: int = 1024,
        spatial_resolution: int = 518,
        eval_transforms=None,
        spatial_transforms=None,
        clip_length: int = 8,
        spatial_type: str = "vggt",
        use_extrinsics: bool = False,
        max_objects_per_frame: int = 10,
        slide_stride: int = -1,
    ):
        super().__init__()
        self.base_dir = base_dir
        assert os.path.exists(self.base_dir)
        self.predict_metric = predict_metric
        self.resolution = resolution
        self.spatial_resolution = spatial_resolution
        self.norm_scale = norm_scale
        self.spatial_type = spatial_type
        self.use_extrinsics = use_extrinsics
        self.max_objects_per_frame = max_objects_per_frame

        if use_extrinsics:
            sample_meta_path = sample_meta_path.replace("json_wo_pose/val", "val-json")
        sample_meta_path = resolve_existing_path(sample_meta_path)
        with open(sample_meta_path, "r") as f:
            sample_meta = json.load(f)
        self.sample_meta = sample_meta
        self.meta_files = [(name, default_scale, len(sample_meta["frames"]))]
        if slide_stride <= 0:
            slide_stride = clip_length
        meta_file_paths = self._split_videos_by_sliding_window(
            self.meta_files, clip_length, sliding_stride=slide_stride
        )
        self.video_split_per_frame = clip_length
        self.meta_file_paths = meta_file_paths

        self.eval_transforms = eval_transforms
        self.spatial_transforms = spatial_transforms

        if self.spatial_type == "depthanythingv3":
            from depth_anything_3.utils.io.input_processor import InputProcessor

            self.input_processor = InputProcessor()

    def __len__(self):
        return len(self.meta_file_paths)

    def __getitem__(self, idx):
        name, metric_scale, frame_num, split_idx = self.meta_file_paths[idx]
        input_dict = self._load_frames(metric_scale, frame_num, split_idx, name)
        # if input_dict["valid"] is False:
        #     return input_dict
        if self.eval_transforms:
            rgb_image = input_dict["rgb_images"]
            input_dict["rgb_images"] = self.eval_transforms(rgb_image)
        if self.spatial_transforms:
            spatial_image = input_dict["spatial_images"]
            input_dict["spatial_images"] = self.spatial_transforms(spatial_image)

        input_dict = self._totensor(input_dict)
        input_dict["dataset_id"] = DATASET_LOOKUP_TABLE[name]
        return input_dict


    def _split_videos_by_sliding_window(
        self, meta_file_paths, video_split_per_frame, sliding_stride=4
    ):
        """
        meta_file_paths: List[(name, scale, path, num_frames)]  # 来自上游汇总
        产出: List[(name, scale, path, split_len, split_start)]
        """
        new_meta = []
        max_per_split = video_split_per_frame
        for name, scale, num_frames in meta_file_paths:
            if num_frames <= 0:
                continue
            # 按最大帧数直接滑窗切分
            for start in range(
                0,
                num_frames,
                sliding_stride,
            ):
                split_len = min(max_per_split, num_frames - start)
                new_meta.append((name, scale, split_len, start))
        return new_meta

    # ========
    # code for loading a frame
    # ========
    def _totensor(self, dict):
        for key in dict:
            if isinstance(dict[key], np.ndarray):
                dict[key] = torch.from_numpy(dict[key])
        return dict

    def _load_frames(self, metric_scale, all_frame_num, split_idx, name):
        meta_file = self.sample_meta
        scene_id = meta_file["scene_id"]
        frames_items = list(
            sorted(meta_file["frames"].items(), key=lambda item: item[0])
        )
        n_total = len(frames_items)
        start = max(0, min(split_idx, n_total))
        end = max(start, min(start + all_frame_num, n_total))
        frames = frames_items[start:end]  # 先拿到这个 split 内的帧

        frames = dict(frames)
        objects = meta_file["objects"]
        first_frame_id = list(frames.keys())[0]

        # 只保留在第一帧出现的物体
        objects_in_frame = {
            k: v
            for k, v in objects.items()
            if (v.get(first_frame_id, None) is not None)
        }

        # first_frame_rgb_path = frames[first_frame_id]["rgb_path"]
        # first_frame = cv2.imread(os.path.join(self.base_dir, first_frame_rgb_path))

        # original_resolution = first_frame.shape[:2]  # (H, W)
        # objects_in_frame = filter_object(
        #     objects_in_frame, first_frame_id, resolution=(1280,1280)
        # )
        # 如果第一帧没有物体，就重新采样一个视频
        valid = True
        if len(objects_in_frame) == 0:
            valid = False
        if len(objects_in_frame.values()) > self.max_objects_per_frame:
            selected_objs = list(objects_in_frame.keys())[: self.max_objects_per_frame]
        else:
            selected_objs = list(objects_in_frame.keys())

        if not self.predict_metric:
            distance_norm = meta_file["pcd_average_distance"]
        else:
            distance_norm = self.norm_scale

        original_frame_size = None
        video_bboxes_trunc_2d = []
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
        image_paths = []
        for frame_id, frame in frames.items():
            object_trunc_2d = []
            object_bboxes_2d = []
            object_bboxes_center_3d = []
            object_bboxes_dim_3d = []
            object_bboxes_yaw_3d = []
            object_bboxes_R_3d = []
            object_id_frame = []
            object_appearance_frame = []

            rgb_path = os.path.join(self.base_dir, frame["rgb_path"])
            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))  # shape (H, W, 3)
            depth_path = os.path.join(self.base_dir, frame["depth_path"])
            depth_image = np.array(Image.open(depth_path)).astype(np.float32)
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
            orientation = frame.get("img_orientation", 0)
            if orientation is None:
                orientation = 0
            rgb_image = rotate_image(rgb_image, orientation)
            depth_image = rotate_image(depth_image, orientation)
            intrinsic = np.array(frame["rgb_intrinsic"])
            if len(intrinsic.shape) > 2:
                intrinsic = intrinsic[0]
            # x right, y up, z forward to x right, y down, z backward
            camera_pose = frame.get("camera_pose", None)
            if camera_pose is not None:
                camera_pose = np.array(camera_pose)
            else:
                camera_pose = np.eye(4)
            if name == "hyperism":
                T_convert = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )
                camera_pose = camera_pose @ T_convert
            extrinsic = np.linalg.inv(camera_pose)
            T_gravity = np.eye(4)
            # placeholder; currently not used in training/evaluation
            if original_frame_size is None:
                original_frame_size = rgb_image.shape[:2]

            for k, obj in objects_in_frame.items():
                if k not in selected_objs:
                    continue
                object_id_frame.append(k)
                has_object = obj.get(frame_id, None) is not None
                if has_object:
                    bbox2d = np.array(obj[frame_id]["camera_box_2d_rend"])
                    # 过滤极小的 bbox
                    w_2d = bbox2d[2] - bbox2d[0]
                    h_2d = bbox2d[3] - bbox2d[1]
                    pixel_area = w_2d * h_2d
                    if pixel_area < 100 and w_2d < 20 and h_2d < 20:
                        has_object = False
                if not has_object:
                    object_appearance_frame.append(False)
                    object_trunc_2d.append(
                        NOT_APPEAR_FLAG * np.ones((1,), dtype=np.float32)
                    )
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
                    object_ids.append(None)
                    continue
                object_appearance_frame.append(True)
                if obj[frame_id].get("camera_box_2d_proj", None) is not None:
                    bbox2d = np.array(obj[frame_id]["camera_box_2d_rend"])
                    bbox2d_proj = np.array(obj[frame_id]["camera_box_2d_proj"])
                    trunc_y = (bbox2d[3] - bbox2d[1]) / (
                        bbox2d_proj[3] - bbox2d_proj[1]
                    )
                    trunc_x = (bbox2d[2] - bbox2d[0]) / (
                        bbox2d_proj[2] - bbox2d_proj[0]
                    )
                    assert trunc_x <= 1 and trunc_y <= 1, (
                        f"trunc_x: {trunc_x}, trunc_y: {trunc_y}"
                    )
                else:
                    bbox2d = np.array(obj[frame_id]["camera_box_2d_rend"])
                    trunc_x, trunc_y = 1.0, 1.0

                object_trunc_2d.append(
                    trunc_x * trunc_y * np.ones((1,), dtype=np.float32)
                )
                object_bboxes_2d.append(bbox2d)
                object_bboxes_center_3d.append(
                    np.array(obj[frame_id]["camera_center"]) / distance_norm
                )

                camera_scale = np.array(obj[frame_id]["camera_scale"]) / distance_norm
                rotmat = np.array(obj[frame_id]["camera_R"])
                if name == "hyperism":
                    rot_y_90 = R.from_euler("y", 90, degrees=True).as_matrix()
                    rotmat = rotmat @ rot_y_90
                elif name == "ADT":
                    rotmat, camera_scale = remap_axes(
                        rotmat, camera_scale, perm=(2, 1, 0), signs=(1, -1, -1)
                    )
                # elif "arkitscenes" in name.lower():
                #     rotmat, camera_scale = remap_axes(
                #         rotmat, camera_scale, perm=(0, 2, 1), signs=(1, -1, 1)
                #     )
                object_bboxes_dim_3d.append(camera_scale)
                yaw = np.arctan2(-rotmat[0, 2], rotmat[2, 2])
                object_bboxes_yaw_3d.append(yaw)
                object_bboxes_R_3d.append(rotmat)
            if valid is False:
                # padding for all entries
                object_trunc_2d.append(
                    NOT_APPEAR_FLAG * np.ones((1,), dtype=np.float32)
                )
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
                object_ids.append(None)
            # if sum(object_appearance_frame) == 0:  # no valid object in this frame
            #    valid = False
            object_ids.append(object_id_frame)
            image_paths.append(rgb_path)
            object_appearance.append(np.array(object_appearance_frame))
            object_bboxes_2d_np = np.stack(object_bboxes_2d, axis=0).astype(np.float32)
            object_bboxes_center_3d_np = np.stack(
                object_bboxes_center_3d, axis=0
            ).astype(np.float32)
            object_bboxes_trunc_2d_np = np.concatenate(object_trunc_2d, axis=0).astype(
                np.float32
            )
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
            # Note that we only use square images for training
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
                # spa_to_ori = self.spatial_resolution / self.resolution
                # spatial_intrinsic = self._resize_intrinsic(
                #     intrinsic, (spa_to_ori, spa_to_ori), pixel_center=True
                # )
            elif self.spatial_type == "mapanything":
                spatial_image = load_images(
                    rgb_path,
                    resize_mode="square",
                    size=self.spatial_resolution,
                    orientation=orientation,
                )[0]
            elif self.spatial_type == "depthanythingv3":
                # a placeholder, actual spatial image will be processed outside
                spatial_image = rgb_image.copy()
            else:
                spatial_image = np.zeros(
                    (self.spatial_resolution, self.spatial_resolution, 3),
                    dtype=np.uint8,
                )
            # spa_to_ori = self.spatial_resolution / self.resolution
            # spatial_intrinsic = self._resize_intrinsic(
            #     intrinsic, (spa_to_ori, spa_to_ori), pixel_center=True
            # )
            video_bboxes_trunc_2d.append(object_bboxes_trunc_2d_np)
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
        video_bboxes_trunc_2d = np.stack(video_bboxes_trunc_2d, axis=0)  # (N, M)
        video_bboxes_2d = np.stack(video_bboxes_2d, axis=0)  # (N, M, 4)
        video_bboxes_center_3d = np.stack(video_bboxes_center_3d, axis=0)  # (N, M, 3)
        video_bboxes_dim_3d = np.stack(video_bboxes_dim_3d, axis=0)  # (N, M, 3)
        video_bboxes_yaw_3d = np.stack(video_bboxes_yaw_3d, axis=0)  # (N, M)
        video_bboxes_R_3d = np.stack(video_bboxes_R_3d, axis=0)  # (N, M, 3, 3)
        video_rgb_images = np.stack(video_rgb_images, axis=0)  # (N, H, W, 3)
        video_depth_images = np.stack(video_depth_images, axis=0)  # (N, H, W)
        video_intrinsics = np.stack(video_intrinsics, axis=0)  # (N, 3, 3)
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
        # video_spatial_intrinsics = np.stack(
        #     video_spatial_intrinsics, axis=0
        # )  # (N, 3, 3)
        T_gravities = np.stack(T_gravities, axis=0)  # (N, 4, 4)

        extrinsics = np.stack(extrinsics, axis=0)  # (N, 4, 4)
        first_frame_extrinsic = extrinsics[0:1]
        normalized_extrinsics = self._normalize_camera_poses(
            extrinsics, dist_norm=distance_norm
        )
        ######################
        ######################
        # load world box if needed
        world_centers = []
        world_dims = []
        world_Rs = []
        if (
            self.use_extrinsics
            and len(selected_objs)
            and "world_center" in objects_in_frame[selected_objs[0]]
        ):
            for obj_name in selected_objs:
                obj = objects_in_frame[obj_name]
                world_centers.append(
                    first_frame_extrinsic[0, :3, :3] @ obj["world_center"]
                    + first_frame_extrinsic[0, :3, 3]
                )
                world_dims.append(obj["world_scale"])
                world_Rs.append(first_frame_extrinsic[0, :3, :3] @ obj["world_R"])
            world_centers = np.array(world_centers)
            world_dims = np.array(world_dims)
            world_Rs = np.array(world_Rs)
        else:
            world_centers = np.zeros(
                (len(selected_objs), 3), dtype=np.float32
            )  # placeholder
            world_dims = np.zeros((len(selected_objs), 3), dtype=np.float32)
            world_Rs = np.zeros((len(selected_objs), 3, 3), dtype=np.float32)

        return {
            "valid": valid,
            "scene_id": scene_id,
            "image_paths": image_paths,
            "original_size": np.array(original_frame_size, dtype=np.int32),
            "object_ids": object_ids,
            "object_appearance": object_appearance,
            "bboxes_trunc_2d": video_bboxes_trunc_2d,
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
            "extrinsics": normalized_extrinsics,
            "original_extrinsics": extrinsics,
            "first_frame_extrinsic": first_frame_extrinsic,
            "T_gravities": T_gravities,
            "metric_scale": metric_scale,
            "distance_norm": distance_norm,
            "world_centers": world_centers,
            "world_dims": world_dims,
            "world_Rs": world_Rs,
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

    def _crop_image_depth_and_intrinsic_by_pp(self, image, depth_map, intrinsic, bbox):
        H, W = image.shape[:2]
        intrinsic = np.copy(intrinsic)

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
