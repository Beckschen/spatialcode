# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def add_noise_to_bbox(
    box_coords: torch.Tensor,
    noise_center_bound: tuple = (-0.2, 0.2),  # 相对于box尺寸的中心偏移比例
    noise_size_bound: tuple = (0.5, 1.5),     # 尺寸缩放比例
) -> torch.Tensor:
    """
    Add random noise to bbox coordinates.

    Inputs:
    - box_coords: [B, 2, 2], (x, y) coordinates of top-left and bottom-right corners
    - noise_center_bound: (min, max) fraction of box size for center offset
    - noise_size_bound: (min, max) scale factor for width/height

    Returns:
    - noisy_box_coords: [B, 2, 2], perturbed coordinates
    """
    device = box_coords.device
    B = box_coords.shape[0]

    # 1. 计算每个 box 的宽高和中心
    tl, br = box_coords[:, 0], box_coords[:, 1]
    widths = (br[:, 0] - tl[:, 0]).clamp(min=1e-6)   # [B]
    heights = (br[:, 1] - tl[:, 1]).clamp(min=1e-6)  # [B]
    cx = (tl[:, 0] + br[:, 0]) / 2
    cy = (tl[:, 1] + br[:, 1]) / 2

    # 2. 为每个 box 生成独立的噪音
    cx_noise = torch.empty(B, device=device).uniform_(noise_center_bound[0], noise_center_bound[1])
    cy_noise = torch.empty(B, device=device).uniform_(noise_center_bound[0], noise_center_bound[1])
    w_noise = torch.empty(B, device=device).uniform_(noise_size_bound[0], noise_size_bound[1])
    h_noise = torch.empty(B, device=device).uniform_(noise_size_bound[0], noise_size_bound[1])

    # 3. 应用噪音 (中心偏移相对于box尺寸)
    cx = cx + widths * cx_noise
    cy = cy + heights * cy_noise
    widths = widths * w_noise
    heights = heights * h_noise

    # 4. 重新计算 noisy box 坐标
    noisy_tl_x = cx - widths / 2
    noisy_tl_y = cy - heights / 2
    noisy_br_x = cx + widths / 2
    noisy_br_y = cy + heights / 2

    noisy_box_coords = torch.stack(
        [
            torch.stack([noisy_tl_x, noisy_tl_y], dim=-1),
            torch.stack([noisy_br_x, noisy_br_y], dim=-1),
        ],
        dim=1,
    )  # [B, 2, 2]
    return noisy_box_coords


def sample_points_around_box_center(
    bboxes: torch.Tensor,  # [B,4] = (x1,y1,x2,y2)
    noise: float = 0.1,  # 噪声占 bbox 尺寸的比例
    noise_bound: float = 20.0,  # 噪声最大像素范围
    center_label: int = 1,  # 给中心点的 label
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 bbox 中心点周围 sample 点，保证在 bbox 内部

    Inputs:
    - bboxes: [B,4], 每个 bbox = (x1,y1,x2,y2)
    - noise: 噪声占 bbox 宽高比例
    - noise_bound: 噪声最大像素
    - center_label: 给中心点的类别标签

    Returns:
    - point_coords: [B,1,2], 每个 bbox 的一个采样点 (x,y)
    - point_labels: [B,1], 每个采样点的 label
    """
    device = bboxes.device
    B = bboxes.shape[0]

    # 中心点坐标
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

    if noise == 0:
        point_coords = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # [B,1,2]
        point_labels = torch.full(
            (B, 1), center_label, dtype=torch.int32, device=device
        )
        return point_coords, point_labels

    # bbox 宽高
    bw = bboxes[:, 2] - bboxes[:, 0]
    bh = bboxes[:, 3] - bboxes[:, 1]

    # 最大噪声范围（取宽/高比例和像素上限的最小值）
    noise_bound_tensor = torch.tensor(noise_bound, dtype=bw.dtype, device=device)
    max_dx = torch.minimum(bw * noise, noise_bound_tensor)
    max_dy = torch.minimum(bh * noise, noise_bound_tensor)

    # 随机偏移 (uniform in [-max, +max])
    dx = (2 * torch.rand(B, device=device) - 1) * max_dx
    dy = (2 * torch.rand(B, device=device) - 1) * max_dy

    # 加噪声后的采样点
    xs = cx + dx
    ys = cy + dy

    # 保证点不超过 bbox (requires PyTorch >= 1.9 for tensor bounds)
    xs = xs.clamp(min=bboxes[:, 0], max=bboxes[:, 2])
    ys = ys.clamp(min=bboxes[:, 1], max=bboxes[:, 3])

    # 组装返回
    point_coords = torch.stack([xs, ys], dim=-1).unsqueeze(1)  # [B,1,2]
    point_labels = torch.full((B, 1), center_label, dtype=torch.int32, device=device)

    return point_coords, point_labels


def sample_negative_points(
    bboxes: torch.Tensor,  # [B, 4], 每个 bbox = (x1, y1, x2, y2)
    image_size: Tuple[int, int],  # (H, W)
    num_points: int = 1,  # 每个 bbox 采多少负样本点
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 bbox 外部采样负样本点，label=0

    Returns:
    - point_coords: [B, num_points, 2] (x, y)
    - point_labels: [B, num_points] (全为0)
    """
    device = bboxes.device
    B = bboxes.shape[0]
    H, W = image_size

    coords = []
    for i in range(B):
        x1, y1, x2, y2 = bboxes[i]
        neg_pts = []
        while len(neg_pts) < num_points:
            # 在整张图随机采样
            xs = torch.randint(0, W, (num_points,), device=device)
            ys = torch.randint(0, H, (num_points,), device=device)
            mask_outside = (xs < x1) | (xs > x2) | (ys < y1) | (ys > y2)
            valid = torch.stack([xs[mask_outside], ys[mask_outside]], dim=-1)
            for pt in valid:
                neg_pts.append(pt)
                if len(neg_pts) == num_points:
                    break
        coords.append(torch.stack(neg_pts, dim=0))

    point_coords = torch.stack(coords, dim=0)  # [B, num_points, 2]
    point_labels = torch.zeros((B, num_points), dtype=torch.int32, device=device)
    return point_coords, point_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")


def sample_points_in_bboxes(
    bboxes: torch.Tensor, num_points: int, device=None, dtype=None
):
    """
    在 B 个 bbox 内均匀采样点 (Torch版本)

    Args:
        bboxes: (B, 2, 2) Tensor，每个框是 [[x1,y1],[x2,y2]]
        num_points: 每个 bbox 采样点数
        device: 输出设备
        dtype: 输出数据类型

    Returns:
        points: (B, num_points, 2) Tensor
    """
    if device is None:
        device = bboxes.device
    if dtype is None:
        dtype = bboxes.dtype

    B = bboxes.shape[0]
    x1, y1 = bboxes[:, 0, 0], bboxes[:, 0, 1]
    x2, y2 = bboxes[:, 1, 0], bboxes[:, 1, 1]

    # 网格大小（尽量接近正方形）
    grid_size = int(
        torch.ceil(
            torch.sqrt(torch.tensor(num_points, device=device, dtype=dtype))
        ).item()
    )

    # 生成 [0,1] 范围的均匀网格
    xs = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    xv, yv = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1)  # (G^2, 2)

    # 缩放到每个 bbox 范围
    grid = grid.unsqueeze(0).repeat(B, 1, 1)  # (B, G^2, 2)

    mins = torch.stack([x1, y1], dim=-1).unsqueeze(1)  # (B,1,2)
    maxs = torch.stack([x2, y2], dim=-1).unsqueeze(1)

    scaled_points = mins + grid * (maxs - mins)  # (B, G^2, 2)

    # 随机打乱 & 选择前 num_points 个
    G2 = scaled_points.shape[1]
    if G2 > num_points:
        idx = torch.randperm(G2, device=device)[:num_points]
        scaled_points = scaled_points[:, idx, :]
    else:
        # 如果不足，直接返回所有点
        scaled_points = scaled_points

    return scaled_points  # (B, num_points, 2)


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
                pose_encoding.shape[:1] + (3, 3), device=pose_encoding.device
            )  # F,3,3
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics


def quat_to_rotmat(q: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Quaternion -> Rotation Matrix (Shoemake, batch-safe)
    Input: q of shape (B, 4) in [w, x, y, z] format
    Output: R of shape (B, 3, 3)
    """
    if isinstance(q, np.ndarray):
        is_numpy = True
        q = torch.tensor(q, dtype=torch.float32)
    else:
        is_numpy = False

    # ensure normalized quaternion
    q = q / q.norm(dim=-1, keepdim=True)

    w, x, y, z = q.unbind(dim=-1)

    B = q.shape[0]
    R = torch.zeros((B, 3, 3), dtype=q.dtype, device=q.device)

    # Precompute repeated terms
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    # Fill rotation matrix
    # Following standard quaternion → rotation matrix formula
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)

    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)

    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    if is_numpy:
        return R.numpy()
    return R


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R -> quaternion  (Shoemake, batch-safe)
    Returns [w, x, y, z]
    """
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    R00 = R[:, 0, 0]
    R11 = R[:, 1, 1]
    R22 = R[:, 2, 2]
    trace = R00 + R11 + R22

    eps = 1e-8

    # 分支0：trace > 0
    cond0 = trace > 0
    S0 = torch.sqrt(torch.clamp(trace[cond0] + 1.0, min=eps)) * 2
    q[cond0, 0] = 0.25 * S0
    q[cond0, 1] = (R[cond0, 2, 1] - R[cond0, 1, 2]) / S0
    q[cond0, 2] = (R[cond0, 0, 2] - R[cond0, 2, 0]) / S0
    q[cond0, 3] = (R[cond0, 1, 0] - R[cond0, 0, 1]) / S0

    # 分支1：R00 最大
    cond1 = (~cond0) & (R00 > R11) & (R00 > R22)
    S1 = (
        torch.sqrt(torch.clamp(1.0 + R00[cond1] - R11[cond1] - R22[cond1], min=eps)) * 2
    )
    q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / S1
    q[cond1, 1] = 0.25 * S1
    q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / S1
    q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / S1

    # 分支2：R11 最大
    cond2 = (~cond0) & (~cond1) & (R11 > R22)
    S2 = (
        torch.sqrt(torch.clamp(1.0 + R11[cond2] - R00[cond2] - R22[cond2], min=eps)) * 2
    )
    q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / S2
    q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / S2
    q[cond2, 2] = 0.25 * S2
    q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / S2

    # 分支3：剩下的一切 = R22 最大
    cond3 = (~cond0) & (~cond1) & (~cond2)
    S3 = (
        torch.sqrt(torch.clamp(1.0 + R22[cond3] - R00[cond3] - R11[cond3], min=eps)) * 2
    )
    q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / S3
    q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / S3
    q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / S3
    q[cond3, 3] = 0.25 * S3

    q = q / q.norm(dim=-1, keepdim=True)
    return q


def project_to_so3(R):
    if isinstance(R, torch.Tensor):
        is_tensor = True
        R_ = R.cpu().numpy()
    else:
        R_ = R
        is_tensor = False
    # R: (...,3,3)
    U, S, Vh = np.linalg.svd(R_)

    # initial projection
    R_proj = U @ Vh

    # check determinant (last axis: i.e., R_proj[..., :, :])
    det = np.linalg.det(R_proj)

    # mask: where det < 0
    neg_mask = det < 0

    if np.any(neg_mask):
        # flip last column of U for those matrices
        U[neg_mask, :, -1] *= -1
        R_proj = U @ Vh
    if is_tensor:
        R_proj = torch.tensor(R_proj, dtype=R.dtype, device=R.device)
    return R_proj
