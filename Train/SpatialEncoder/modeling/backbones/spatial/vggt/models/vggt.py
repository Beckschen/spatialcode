# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead
from ..heads.track_head import TrackHead


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(
                pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device
            )
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        output_idx: List[int] = [4, 11, 17, 23],
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_idx=output_idx,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                return_feat=True
            )
            if enable_depth
            else None
        )
        self.track_head = (
            TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
            if enable_track
            else None
        )

    def pose_encoding_to_extri_intri(
        self,
        pose_encoding,
        image_size_hw=None,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,  # e.g., (256, 512)
    ):
        """Convert a pose encoding back to camera extrinsics and intrinsics.

        This function performs the inverse operation of extri_intri_to_pose_encoding,
        reconstructing the full camera parameters from the compact encoding.

        Args:
            pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
                where B is batch size and S is sequence length.
                For "absT_quaR_FoV" type, the 9 dimensions are:
                - [:3] = absolute translation vector T (3D)
                - [3:7] = rotation as quaternion quat (4D)
                - [7:] = field of view (2D)
            image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
                Required for reconstructing intrinsics from field of view values.
                For example: (256, 512).
            pose_encoding_type (str): Type of pose encoding used. Currently only
                supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
            build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
                If False, only extrinsics are returned and intrinsics will be None.

        Returns:
            tuple: (extrinsics, intrinsics)
                - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
                In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
                transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
                a 3x1 translation vector.
                - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
                or None if build_intrinsics is False. Defined in pixels, with format:
                [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]]
                where fx, fy are focal lengths and (cx, cy) is the principal point,
                assumed to be at the center of the image (W/2, H/2).
        """

        intrinsics = None

        if pose_encoding_type == "absT_quaR_FoV":
            T = pose_encoding[..., :3]
            quat = pose_encoding[..., 3:7]
            fov_h = pose_encoding[..., 7]
            fov_w = pose_encoding[..., 8]

            R = quat_to_mat(quat)
            extrinsics = torch.cat([R, T[..., None]], dim=-1)

            if build_intrinsics:
                H, W = image_size_hw
                fy = (H / 2.0) / torch.tan(fov_h / 2.0)
                fx = (W / 2.0) / torch.tan(fov_w / 2.0)
                intrinsics = torch.zeros(
                    pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device
                )
                intrinsics[..., 0, 0] = fx
                intrinsics[..., 1, 1] = fy
                intrinsics[..., 0, 2] = W / 2
                intrinsics[..., 1, 2] = H / 2
                intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
        else:
            raise NotImplementedError

        return extrinsics, intrinsics

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        assert len(images.shape)==5,"currently only supports b,s,c,h,w"

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[
                    -1
                ]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = (
                images  # store the images for visualization during inference
            )

        return predictions
