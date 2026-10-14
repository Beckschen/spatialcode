# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0, **kwargs):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        self.interactive_2d_3d = kwargs.get("interactive_2d_3d", None)
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        )

    def forward(
        self,
        sample: torch.Tensor,
        video_feature_dict: dict = None,
        frame_idx: Optional[int] = None,
    ):
        # Forward through backbone
        if self.trunk.is_frozen:
            with torch.no_grad():
                outputs = self.trunk(sample)
        else:
            outputs = self.trunk(sample)
        if video_feature_dict is not None:
            features_3d = video_feature_dict["selected_level_features"]
            frame_indices = video_feature_dict.get("frame_indices", [])
            if len(frame_indices) > 0:
                # TODO: currently only support single frame indexing for 2D-3D interaction
                # spatial encoder may have sliding window inputs
                raise NotImplementedError()
                spatial_frame_idx = (
                    frame_indices.index(frame_idx)
                    if isinstance(frame_idx, int) and frame_idx >= 0
                    else -1
                )
            else:
                spatial_frame_idx = frame_idx
            if len(features_3d[0].shape) == 4:
                # apply for mapanything features
                # (F, C, H, W) -> (1, F, C, H, W)
                features_3d = [_.unsqueeze(0) for _ in features_3d]
            if isinstance(spatial_frame_idx, int) and spatial_frame_idx >= 0:
                # Convert global frame_idx to relative index using spatial_frame_offset
                spatial_offset = video_feature_dict.get("spatial_frame_offset", 0)
                relative_idx = spatial_frame_idx - spatial_offset
                # Clamp to valid range [0, num_frames-1]
                max_idx = features_3d[0].shape[1] - 1
                clamped_idx = max(0, min(relative_idx, max_idx))
                features_3d = [_[:, clamped_idx : clamped_idx + 1] for _ in features_3d]
            elif isinstance(spatial_frame_idx, int) and spatial_frame_idx == -1:
                pass
            elif isinstance(spatial_frame_idx, torch.Tensor):
                assert len(spatial_frame_idx.shape) == 1, (
                    "spatial_frame_idx should be a 1D tensor"
                )
                features_3d = [_[:, spatial_frame_idx] for _ in features_3d]
            else:
                raise ValueError(
                    "frame_idx must be specified and >=0 when using 2D-3D interaction"
                )
            assert self.interactive_2d_3d is not None
            mixed_outputs = self.interactive_2d_3d(outputs, features_3d)
            features, pos = self.neck(mixed_outputs)
            ori_features, ori_pos = self.neck(outputs)
        else:
            features, pos = self.neck(outputs)
            ori_features, ori_pos = features, pos
        assert self.scalp > 0, "Currently only support remove the last level of feature"
        # Discard the lowest resolution features
        features, pos = features[: -self.scalp], pos[: -self.scalp]
        # Also apply scalp to original features
        ori_features = ori_features[: -self.scalp]
        ori_pos = ori_pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
            "original_sam2": {
                "vision_features": ori_features[-1],
                "vision_pos_enc": ori_pos,
                "backbone_fpn": ori_features,
            },
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
        is_frozen: bool = False,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)
        self.is_frozen = is_frozen
        if self.is_frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
