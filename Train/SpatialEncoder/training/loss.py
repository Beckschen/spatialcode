# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List
from dataclasses import asdict
from scipy.spatial.transform import Rotation as R
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbones.spatial.vggt.utils.pose_enc import extri_intri_to_pose_encoding

from training.utils.transform_utils import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    project_to_image,
    giou_diag,
    so3_relative_angle,
    class2angle,
    rotmat_to_quat,
    quat_to_rotmat,
    check_and_fix_inf_nan,
    decode_bbox3d,
    box3d_iou,
    compute_3d_bbox_vertices,
)

from training.trainer import CORE_LOSS_KEY


def bbox_2d_giou_loss(inputs, targets, has_object_indicator, resolution):
    """
    IoU (or GIoU) loss for 2D bounding boxes.

    Args:
        inputs: Tensor [N, num_preds, 4], predicted bboxes (x0,y0,x1,y1)
        targets: Tensor [N, 4], gt bboxes (x0,y0,x1,y1)
        num_objects: (optional) int, number of valid objects in batch
        has_object_indicator: (optional) Tensor [N], 1=valid, 0=invalid

    Returns:
        Tensor [num_preds], each entry is the scalar loss for that branch
    """
    assert inputs.dim() == 3 and inputs.size(-1) == 4
    assert targets.dim() == 2 and targets.size(-1) == 4
    num_preds = inputs.size(1)
    targets = targets / resolution

    branch_losses = []
    for i in range(num_preds):
        # [N], each pred[i] with its own gt
        giou = giou_diag(inputs[:, i], targets)  # already elementwise
        loss = 1.0 - giou  # IoU/GIoU loss in [0,2]

        loss = loss * has_object_indicator.float()
        denom = has_object_indicator.sum().clamp(min=1)

        branch_loss = loss / denom
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses, dim=0)  # [N, num_preds]


def bbox_2d_l1_loss(inputs, targets, has_object_indicator, resolution):
    """
    L1 loss for 2D bounding boxes.

    Args:
        inputs: A float tensor of shape [N, num_preds, 4] or [N, 4].
                The predicted bounding boxes in (cx, cy, w, h) format.
        targets: A float tensor of shape [N, 4].
                 The target bounding boxes in (cx, cy, w, h) format.
        num_objects: Number of objects in the batch
        has_object_indicator: Optional tensor of shape [N],
                              1 for valid object, 0 for padded/invalid

    Returns:
        Tensor of shape [num_preds, 1] if multiple preds,
        otherwise scalar tensor.
    """
    targets = targets / resolution
    # [N, num_preds, 4] case
    num_preds = inputs.size(1)
    branch_losses = []
    for i in range(num_preds):
        loss_l1_bbox = F.smooth_l1_loss(
            inputs[:, i], targets, reduction="none", beta=1.0 / 9.0
        )
        if has_object_indicator is not None:
            loss_l1_bbox = loss_l1_bbox * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_l1_bbox.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [N, num_preds,1]


def bbox_3d_center_in_2d_loss(
    inputs, targets, intrinsics, resolution, has_object_indicator
):
    """
    L1 loss for 3D bounding boxes (center projected to 2D).

    Args:
        inputs: Tensor, shape [N, 7] or [N, num_preds, 7]
                Predicted bboxes (cx, cy, cz, w, h, l, confidence)
        targets: Tensor, shape [N, 6]
                 GT bboxes (cx, cy, cz, w, h, l)
        intrinsics: Tensor, shape [3, 3]
                    Camera intrinsic matrix
        num_objects: Number of objects in the batch
        has_object_indicator: Optional Tensor [N], 1 = valid object, 0 = ignore

    Returns:
        - If single prediction: scalar tensor
        - If multiple predictions: [num_preds,1] tensor
    """
    num_preds = inputs.size(1)
    branch_losses = []
    for i in range(num_preds):
        pred_center_2d = project_to_image(inputs[:, i, :3], intrinsics)  # [N,2]
        target_center_2d = project_to_image(targets[:, :3], intrinsics)  # [N,2]
        pred_center_2d = pred_center_2d / resolution
        target_center_2d = target_center_2d / resolution
        loss_l1_bbox = F.smooth_l1_loss(
            pred_center_2d, target_center_2d, reduction="none", beta=1.0 / 9.0
        )  # [N,2]
        loss_l1_bbox = loss_l1_bbox * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_l1_bbox.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [N, num_preds,1]


def bbox_3d_depth_loss(inputs, targets, has_object_indicator, log_depth_error=False):
    """
    L1 loss for 3D bounding boxes depth (z) with confidence weighting.

    Args:
        inputs: Tensor, shape [N,7] or [N,num_preds,7]
                Predicted bboxes (cx, cy, cz, w, h, l, confidence)
        targets: Tensor, shape [N,6]
                 GT bboxes (cx, cy, cz, w, h, l)
        num_objects: int, number of objects in the batch
        has_object_indicator: Optional Tensor [N], 1=valid, 0=ignore

    Returns:
        - scalar tensor if single prediction
        - [num_preds,1] tensor if multiple predictions
    """
    num_preds = inputs.size(1)
    branch_losses = []
    for i in range(num_preds):
        pred_depth = inputs[:, i, 2].clamp(min=1e-6).log()
        real_depth = targets[:, 2].clamp(min=1e-6).log()
        confidence = inputs[:, i, -1].clamp(min=1e-6).log()  # clamp to avoid log(0)

        loss_confidence_depth = (
            1.4142 * torch.exp(confidence) * torch.abs(pred_depth - real_depth)
            - confidence
        ).unsqueeze(-1)  # [N,1]

        if has_object_indicator is not None:
            loss_confidence_depth = (
                loss_confidence_depth * has_object_indicator.unsqueeze(-1)
            )

        branch_loss = loss_confidence_depth.sum(-1) / has_object_indicator.sum().clamp(
            min=1
        )
        branch_losses.append(branch_loss)
    if log_depth_error:
        # log average depth error in meters
        depth_error = (
            torch.abs(inputs[:, :, 2] - targets[:, 2][..., None])
            * has_object_indicator[..., None]
        )
        avg_depth_error = depth_error.mean()
        return torch.stack(branch_losses), depth_error, avg_depth_error
    return torch.stack(branch_losses)  # [N, num_preds,1]


def bbox_3d_dim_loss(inputs, targets, has_object_indicator):
    """
    L1 loss for 3D bounding boxes dimensions (w, h, l).

    Args:
        inputs: Tensor [N,7] or [N,num_preds,7]
                Predicted bboxes (cx, cy, cz, w, h, l, confidence)
        targets: Tensor [N,6]
                 GT bboxes (cx, cy, cz, w, h, l)
        num_objects: int
        has_object_indicator: Optional Tensor [N], 1=valid, 0=ignore

    Returns:
        - scalar tensor if single prediction
        - [num_preds,1] tensor if multiple predictions
    """
    num_preds = inputs.size(1)
    branch_losses = []
    for i in range(num_preds):
        pred_dim = inputs[:, i, 3:6].clamp(min=1e-6).log()
        real_dim = targets[:, 3:6].clamp(min=1e-6).log()
        loss_l1_dim = F.smooth_l1_loss(
            pred_dim, real_dim, reduction="none", beta=1.0 / 9.0
        )  # [N,3]
        loss_l1_dim = loss_l1_dim * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_l1_dim.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [N, num_preds]


def bbox_3d_alpha_loss(
    inputs_cls,
    targets_cls,
    inputs_res,
    targets_res,
    has_object_indicator,
):
    """
    Orientation (alpha) loss for 3D bounding boxes.
    Combines classification (bin) and residual regression.

    Args:
        inputs_cls: Tensor [N,num_bins] or [N,num_preds,num_bins]
        targets_cls: Tensor [N] (bin ids)
        inputs_res: Tensor [N,num_bins] or [N,num_preds,num_bins]
        targets_res: Tensor [N] (residuals)
        num_objects: int
        has_object_indicator: Optional Tensor [N], 1=valid, 0=ignore

    Returns:
        - scalar tensor if single prediction
        - [num_preds,1] tensor if multiple predictions
    """

    num_preds = inputs_cls.size(1)
    branch_losses = []
    for i in range(num_preds):
        cls_logits = inputs_cls[:, i, :]  # [N,num_bins]
        res_logits = inputs_res[:, i, :]  # [N,num_bins]
        # classification loss
        alpha_cls_loss = F.cross_entropy(cls_logits, targets_cls, reduction="none")
        # residual regression
        gt_dir_cls_onehot = F.one_hot(
            targets_cls, num_classes=cls_logits.shape[1]
        ).float()
        pred_res = torch.sum(res_logits * gt_dir_cls_onehot, 1)
        alpha_res_loss = F.smooth_l1_loss(
            pred_res, targets_res, reduction="none", beta=1.0 / 9.0
        )
        loss_alpha = alpha_cls_loss + alpha_res_loss
        loss_alpha = loss_alpha.unsqueeze(-1)  # [N,1]
        loss_alpha = loss_alpha * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_alpha.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [N, num_preds,1]


def bbox_3d_l1_loss(
    inputs,  # [N,3,7]
    gt_bbox_3d,  # [N,6]
    has_object_indicator,
):
    num_preds = inputs.size(1)  # e.g., 3
    branch_losses = []

    for k in range(num_preds):
        loss_l1_bbox = F.smooth_l1_loss(
            inputs[:, k, :6], gt_bbox_3d, reduction="none", beta=1.0 / 9.0
        )  # [N,6]
        loss_l1_bbox = loss_l1_bbox * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_l1_bbox.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [num_preds]


def bbox_3d_iou_loss(
    inputs,  # [N,3,7]
    gt_bbox_3d,  # [N,6]
    pred_ious,  # [N,3]
    alpha_cls,  # [N,3,12]
    alpha_res,  # [N,3,12]
    gt_cls,  # [N]
    gt_res,  # [N]
    intrinsics,  # [3,3]
    has_object_indicator,
    pred_bboxes_quat=None,
    gt_bboxes_quat=None,
    log_iou=False,
    log_yaw_error=False,
):
    num_preds = inputs.size(1)  # e.g., 3
    branch_losses = []

    with torch.no_grad():
        ## compute GT rotation once
        # gt_alpha = class2angle(gt_cls, gt_res).unsqueeze(-1)  # [N,1]
        # target_center_2d = project_to_image(gt_bbox_3d[:, :3], intrinsics)
        # gt_ry = (
        #     torch.atan2(target_center_2d[..., 1] - intrinsics[0, 2], intrinsics[0, 0])[
        #         :, None
        #     ]
        #     + gt_alpha
        # )
        # gt_ry[gt_ry > torch.pi] = gt_ry[gt_ry > torch.pi] - torch.pi * 2
        # gt_ry[gt_ry < -torch.pi] = gt_ry[gt_ry < -torch.pi] + torch.pi * 2

        gt_rot_mat = quat_to_rotmat(gt_bboxes_quat.contiguous())  # [N,3,3]
        gt_ry = torch.atan2(-gt_rot_mat[:, 0, 2], gt_rot_mat[:, 2, 2]).unsqueeze(-1)  # [N,1]
        gt_ry[gt_ry > torch.pi] = gt_ry[gt_ry > torch.pi] - torch.pi * 2
        gt_ry[gt_ry < -torch.pi] = gt_ry[gt_ry < -torch.pi] + torch.pi * 2

        multi_real_ious = []
        multi_yaw_errors = []
        for k in range(num_preds):
            ## orientation prediction for branch k
            # pred_cls = torch.argmax(alpha_cls[:, k, :], dim=-1)  # [N]
            # pred_res = alpha_res[torch.arange(alpha_res.size(0)), k, pred_cls]  # [N]
            # pred_alpha = class2angle(pred_cls, pred_res)  # [N]
            # pred_center_2d = project_to_image(inputs[:, k, :3], intrinsics)

            # pred_ry = (
            #     torch.atan2(pred_center_2d[..., 1] - intrinsics[0, 2], intrinsics[0, 0])
            #     + pred_alpha
            # )  # [N,1]
            # pred_ry[pred_ry > torch.pi] = pred_ry[pred_ry > torch.pi] - torch.pi * 2
            # pred_ry[pred_ry < -torch.pi] = pred_ry[pred_ry < -torch.pi] + torch.pi * 2

            pred_rot_mat = quat_to_rotmat(
                pred_bboxes_quat[:, k, :].contiguous()
            )  # [N,3,3]
            pred_ry = torch.atan2(-pred_rot_mat[:, 0, 2], pred_rot_mat[:, 2, 2]).unsqueeze(-1)  # [N,1]
            pred_ry[pred_ry > torch.pi] = pred_ry[pred_ry > torch.pi] - torch.pi * 2
            pred_ry[pred_ry < -torch.pi] = pred_ry[pred_ry < -torch.pi] + torch.pi * 2
            ## compute real IoUs per sample
            real_ious = []
            yaw_errors = []
            for i in range(inputs.size(0)):
                vals = inputs[i, k, :6].detach().cpu().numpy()
                x, y, z, w, h, l = vals
                ry = pred_ry[i].detach().cpu()
                pred_bbox3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw=ry)

                gvals = gt_bbox_3d[i, :6].detach().cpu().numpy()
                gx, gy, gz, gw, gh, gl = gvals
                gry = gt_ry[i].detach().cpu()
                real_bbox3d = compute_3d_bbox_vertices(gx, gy, gz, gw, gh, gl, yaw=gry)
                real_iou = box3d_iou(pred_bbox3d, real_bbox3d)
                real_ious.append(real_iou)
                # yaw error
                diff = torch.abs(gry - ry)
                diff = torch.min(diff, 2 * torch.pi - diff)
                yaw_errors.append(diff)
            multi_real_ious.append(real_ious)
            multi_yaw_errors.append(yaw_errors)
        multi_real_ious = torch.tensor(
            multi_real_ious, dtype=torch.float32, device=pred_ious.device
        ).permute(1, 0)  # [N,3]
        multi_real_ious = torch.clamp(multi_real_ious, min=0.0, max=1.0)
        multi_yaw_errors = torch.tensor(
            multi_yaw_errors, dtype=torch.float32, device=pred_ious.device
        ).permute(1, 0)  # [N,3]

    # compute IoU prediction loss (outside no_grad to allow gradients through pred_ious)
    target_ious = multi_real_ious.detach()  # ensure target has no gradient
    loss = F.l1_loss(pred_ious, target_ious, reduction="none")  # [N,3]
    loss = loss * has_object_indicator[..., None]

    branch_loss = loss.sum() / has_object_indicator.sum().clamp(min=1)
    branch_losses.append(branch_loss)

    if log_iou or log_yaw_error:
        # log average IoU for this branch
        # confidence is in (0, +inf), normalize to (0, 1) using x/(x+1)
        confidence = inputs[:, :, -1].clamp(min=1e-6)
        confidence = confidence / (confidence + 1)
        pred_best_iou_idx = torch.argmax(pred_ious * confidence, dim=1)  # [N]
        best_ious = multi_real_ious[
            torch.arange(multi_real_ious.size(0)), pred_best_iou_idx
        ]
        avg_iou = (best_ious * has_object_indicator).sum() / has_object_indicator.sum().clamp(min=1)
        best_yaw_errors = (
            multi_yaw_errors[torch.arange(multi_yaw_errors.size(0)), pred_best_iou_idx]
            * has_object_indicator
        )
        avg_yaw_error = best_yaw_errors.sum() / has_object_indicator.sum().clamp(
            min=1
        )
        return (
            torch.stack(branch_losses),
            best_ious,
            avg_iou,
            best_yaw_errors,
            avg_yaw_error,
        )
    return torch.stack(branch_losses)


def depth_map_loss(
    pred_depth_map,
    pred_depth_conf,
    gt_depth_map,
    coefficient=1,
    valid_mask=None,
):
    pred_depth_map = torch.clamp(pred_depth_map, min=1e-6)
    gt_depth_map = torch.clamp(gt_depth_map, min=1e-6)
    pred_depth_conf = torch.clamp(pred_depth_conf, min=-20, max=20)
    if valid_mask is not None:
        # the depth map predicted by DA3 has already masked by confidence and min depth
        mask = valid_mask > 0
        pred_depth_map = pred_depth_map[mask]
        pred_depth_conf = pred_depth_conf[mask]
        gt_depth_map = gt_depth_map[mask]

    if pred_depth_map.numel() == 0:
        return pred_depth_map.new_zeros((), requires_grad=True)

    alpha = 1e-7
    pred_depth_map = pred_depth_map / (pred_depth_map.norm() + 1)
    gt_depth_map = gt_depth_map / (gt_depth_map.norm() + 1)
    g = torch.log(pred_depth_map + alpha) - torch.log(gt_depth_map + alpha)

    precision = torch.exp(pred_depth_conf)  # = 1/σ²
    weighted_g2 = precision * (g**2) - pred_depth_conf

    Dg = weighted_g2.mean() - coefficient * (torch.mean(g)) ** 2
    Dg = torch.clamp(Dg, min=0.0)
    loss = torch.sqrt(Dg + alpha)  # 避免 sqrt(负数)

    if torch.isnan(loss):
        print("Nan SILog loss")
        print("Input min max", torch.min(pred_depth_map), torch.max(pred_depth_map))
        loss = pred_depth_map.new_zeros((), requires_grad=True)
    loss_l1 = F.smooth_l1_loss(
        pred_depth_map, gt_depth_map, reduction="mean", beta=1.0 / 9.0
    )
    # Combine SI-Log loss and L1 loss
    total_loss = loss + loss_l1
    return total_loss


def bbox_3d_quat_loss(
    inputs,  # [N,3,4]
    gt_quat,  # [N,4]
    has_object_indicator,
):
    num_preds = inputs.size(1)  # e.g., 3
    branch_losses = []
    gt_quat = gt_quat / gt_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    for k in range(num_preds):
        pred_quat = inputs[:, k, :]  # [N,4]
        # Normalize the predicted quaternion to ensure it's a valid rotation
        pred_quat = pred_quat / pred_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Compute the quaternion loss (L2 loss between predicted and ground truth quaternions)
        quat_correct_mask = torch.einsum("bi,bi->b", pred_quat, gt_quat) < 0
        # Flip quaternion sign if dot product is negative (q and -q represent same rotation)
        pred_quat = pred_quat * torch.where(quat_correct_mask.unsqueeze(-1), -1.0, 1.0)
        loss_quat = F.l1_loss(pred_quat, gt_quat, reduction="none")  # [N,4]
        loss_quat = loss_quat * has_object_indicator.unsqueeze(-1)
        branch_loss = loss_quat.sum(-1) / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)  # [num_preds]


def bbox_chamfer_loss(
    pred_bbox_3d, pred_bbox_quat, gt_bbox_3d, gt_bbox_quat, has_object_indicator
):
    gt_box_corners = decode_bbox3d(gt_bbox_3d, gt_bbox_quat)
    B = pred_bbox_3d.shape[0]
    pred_num = pred_bbox_3d.shape[1]
    branch_losses = []
    for k in range(pred_num):
        pred_box_corners = decode_bbox3d(pred_bbox_3d[:, k, :], pred_bbox_quat[:, k, :])
        xx = pred_box_corners.view(B, 8, 1, 3)
        yy = gt_box_corners.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        l1 = l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1)
        loss_chamfer = l1 * has_object_indicator
        branch_loss = loss_chamfer / has_object_indicator.sum().clamp(min=1)
        branch_losses.append(branch_loss)
    return torch.stack(branch_losses)


# ==========================================
# code from VGGT (camera loss & depth loss)
# ==========================================
def camera_loss(
    pred_pose_encodings,  # predictions dict, contains pose encodings
    gt_extrinsics,
    gt_intrinsics,
    image_resolution,
    loss_type="l1",  # "l1" or "l2" loss
    gamma=0.6,  # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,  # weight for translation loss
    weight_rot=1.0,  # weight for rotation loss
    weight_focal=0.5,  # weight for focal length loss
    log_pose=False,
    **kwargs,
):
    if (gt_extrinsics[:3, :3] == torch.eye(3).to(gt_extrinsics.device)).all():
        weight_rot = 0.0
        weight_trans = 0.0
    gt_extrinsics = gt_extrinsics[None, None]
    gt_intrinsics = gt_intrinsics[None, None]
    assert len(gt_extrinsics.shape) == 4
    assert len(gt_intrinsics.shape) == 4

    if len(pred_pose_encodings.shape) == 1:
        pred_pose_encodings = pred_pose_encodings[None]
    # Number of prediction stages

    n_stages = len(pred_pose_encodings)
    # assert n_stages == 1, "Only support single stage prediction for camera loss now."

    # Get ground truth camera extrinsics and intrinsics
    image_hw = image_resolution

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        loss_T_stage, loss_R_stage, loss_FL_stage, T_error, R_error, FL_error = (
            camera_loss_single(
                pred_pose_stage,
                gt_pose_encoding,
                loss_type=loss_type,
                log_pose=log_pose,
            )
        )

        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans + avg_loss_R * weight_rot + avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    # return {
    #     "loss_camera": total_camera_loss,
    #     "loss_T": avg_loss_T,
    #     "loss_R": avg_loss_R,
    #     "loss_FL": avg_loss_FL,
    # }
    if log_pose:
        return (
            total_camera_loss.mean(),
            avg_loss_T,
            avg_loss_R,
            avg_loss_FL,
            T_error,
            R_error,
            FL_error,
        )
    return total_camera_loss.mean()


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", log_pose=False):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(
            dim=-1, keepdim=True
        )
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T_avg = loss_T.clamp(max=100).mean()
    loss_R_avg = loss_R.mean()
    loss_FL_avg = loss_FL.mean()

    if log_pose:
        return (
            loss_T_avg,
            loss_R_avg,
            loss_FL_avg,
            loss_T.mean(-1),
            loss_R.mean(-1),
            loss_FL.mean(-1),
        )

    return loss_T_avg, loss_R_avg, loss_FL_avg, None, None, None


# =========================================
# from MapAnything
def compute_gradient_loss(prediction, gt_target, mask=None):
    """
    Compute the gradient loss between the prediction and GT target at valid points.
    References:
    https://docs.nerf.studio/_modules/nerfstudio/model_components/losses.html#GradientLoss
    https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py

    Args:
        prediction (torch.Tensor): Predicted scene representation. Shape: (B, H, W, C).
        gt_target (torch.Tensor): Ground truth scene representation. Shape: (B, H, W, C).
        mask (torch.Tensor): Mask indicating valid points. Shape: (B, H, W).
    """
    # Expand mask to match number of channels in prediction
    if mask is None:
        mask = torch.ones(
            prediction.shape[0],
            prediction.shape[1],
            prediction.shape[2],
            dtype=prediction.dtype,
            device=prediction.device,
        )
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    summed_mask = torch.sum(mask, (1, 2, 3))

    # Compute the gradient of the prediction and GT target
    diff = prediction - gt_target
    diff = torch.mul(mask, diff)

    # Gradient in x direction
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Gradient in y direction
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp the outlier gradients
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Compute the total loss
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    num_valid_pixels = torch.sum(summed_mask)
    if num_valid_pixels > 0:
        image_loss = torch.sum(image_loss) / num_valid_pixels
    else:
        image_loss = 0 * torch.sum(image_loss)

    return image_loss


# =========================================


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1)  # average over spatial dims
    return loss.mean(1).sum()


# ==========================================


class MultiStepLossFns(nn.Module):
    def __init__(
        self,
        weight_dict,
        depth_sil_coefficient=1.0,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        resolution=1024,
        log_iou=False,
        log_depth_error=False,
        log_yaw_error=False,
        log_pose=False,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.depth_sil_coefficient = depth_sil_coefficient
        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.resolution = resolution

        self.log_iou = log_iou
        self.log_depth_error = log_depth_error
        self.log_yaw_error = log_yaw_error
        self.log_pose = log_pose

    def _organize_into_list(self, targets_batch: Dict) -> List[Dict]:
        """
        Convert each item in targets_batch to a list of length batch_size,
        where each element in the list is a dict containing the data for one sample.
        """
        batch_size = targets_batch["gt_bboxes_2d"].shape[0]
        num_objects = targets_batch["gt_bboxes_2d"].shape[1]
        organized_targets = []
        for i in range(batch_size):
            target = {
                k: v[i].unsqueeze(0) if v is not None else None
                for k, v in targets_batch.items()
            }
            organized_targets.append(target)
        return organized_targets, num_objects

    def get_global_num_objects(self, local_num_objects: int) -> int:
        # 转成 tensor
        num_objects_tensor = torch.tensor(
            [local_num_objects], dtype=torch.float32, device="cuda"
        )
        # 所有进程求和
        dist.all_reduce(num_objects_tensor, op=dist.ReduceOp.SUM)
        # 转成 python int
        return int(num_objects_tensor.item())

    def forward(self, outs_batch: Dict, targets_batch: Dict):
        self.meter_container = {}
        targets_batch = asdict(targets_batch)
        targets_batch, local_num_objects = self._organize_into_list(targets_batch)

        assert len(outs_batch) == len(targets_batch)
        # Number of objects is fixed within a batch

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            # calculate the loss for some of the objects in all frames
            cur_losses = self._forward(outs, targets)
            for k, v in cur_losses.items():
                losses[k] += v
        losses = {k: v / len(outs_batch) for k, v in losses.items()}
        return losses

    def _forward(self, outputs: Dict, targets: Dict):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # N, num pred, 4
        pred_bboxes_2d = outputs["multistep_pred_multibbox_2d"].permute(1, 0, 2)
        pred_bboxes_3d = outputs["multistep_pred_multibbox_3d"].permute(1, 0, 2)
        pred_bboxes_3d_alpha_cls = outputs[
            "multistep_pred_multibbox_alpha_cls"
        ].permute(1, 0, 2)
        pred_bboxes_3d_alpha_res = outputs[
            "multistep_pred_multibbox_alpha_res"
        ].permute(1, 0, 2)
        pred_bbox_quat = outputs["multistep_pred_multibbox_quat"].permute(1, 0, 2)
        pred_depth_maps = outputs["multistep_pred_depth_map"]  # N, 1, H, W
        pred_depth_confs = outputs["multistep_pred_depth_conf"]  # N, 1, H, W
        pred_pose_encs = outputs["multistep_pred_pose_enc"]  # N, 9
        pred_ious = outputs["multistep_pred_bbox_ious"]  # N, num_preds
        object_score_logits = (
            outputs["multistep_object_score_logits"][0]
            if self.pred_obj_scores
            else None
        )

        gt_bboxes_2d = targets["gt_bboxes_2d"][0]
        gt_intrinsics = targets["gt_intrinsics"][0]
        gt_extrinsics = targets["gt_extrinsics"][0]
        gt_bboxes_3d_center = targets["gt_bboxes_center_3d"][0]
        gt_bboxes_3d_dim = targets["gt_bboxes_dim_3d"][0]
        gt_bboxes_3d = torch.cat([gt_bboxes_3d_center, gt_bboxes_3d_dim], dim=-1)
        gt_bboxes_3d_alpha_cls = targets["gt_bboxes_3d_alpha_cls"][0]
        gt_bboxes_3d_alpha_res = targets["gt_bboxes_3d_alpha_res"][0]
        gt_bboxes_quat = targets["gt_bboxes_quat_3d"][0]
        gt_depth_map = targets["gt_depth_map"]

        # accumulate the loss over prediction steps
        losses = {k: 0 for k in self.weight_dict.keys()}
        if self.log_iou:
            losses.update({"avg_iou": 0})
        if self.log_depth_error:
            losses.update({"avg_depth_error": 0})
        if self.log_yaw_error:
            losses.update({"avg_yaw_error": 0})
        self._update_losses(
            losses=losses,
            gt_intrinsic=gt_intrinsics,
            gt_extrinsic=gt_extrinsics,
            gt_bboxes_2d=gt_bboxes_2d,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_bboxes_3d_alpha_cls=gt_bboxes_3d_alpha_cls,
            gt_bboxes_3d_alpha_res=gt_bboxes_3d_alpha_res,
            gt_bboxes_quat=gt_bboxes_quat,
            gt_depth_map=gt_depth_map,
            # ===========
            pred_bboxes_2d=pred_bboxes_2d,
            pred_bboxes_3d=pred_bboxes_3d,
            pred_bboxes_3d_alpha_cls=pred_bboxes_3d_alpha_cls,
            pred_bboxes_3d_alpha_res=pred_bboxes_3d_alpha_res,
            pred_bboxes_quat=pred_bbox_quat,
            pred_depth_maps=pred_depth_maps.mean(0),
            pred_depth_confs=pred_depth_confs.mean(0),
            pred_pose_encs=pred_pose_encs.mean(0),
            pred_ious=pred_ious,
            # ==========
            object_score_logits=object_score_logits,
        )

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self,
        losses,
        gt_intrinsic,
        gt_extrinsic,
        gt_bboxes_2d,  # xyxy
        gt_bboxes_3d,  # cxcyczwhl
        gt_bboxes_3d_alpha_cls,
        gt_bboxes_3d_alpha_res,
        gt_bboxes_quat,
        gt_depth_map,
        # ===========
        pred_bboxes_2d,  # cxcywh
        pred_bboxes_3d,  # cxcyczwhlu
        pred_bboxes_3d_alpha_cls,
        pred_bboxes_3d_alpha_res,
        pred_bboxes_quat,
        pred_depth_maps,
        pred_depth_confs,
        pred_pose_encs,
        pred_ious,
        object_score_logits,
    ):
        has_object_indicator = (gt_bboxes_2d.mean(-1) > 0).float()  # [N]
        ##############################################################################
        # record some statistics
        gt_existence_list = self.meter_container.get("gt_existence", [])
        gt_existence_list.extend(has_object_indicator.cpu().numpy().tolist())
        self.meter_container["gt_existence"] = gt_existence_list

        pred_existence_list = self.meter_container.get("pred_existence", [])
        pred_existence_list.extend(
            (object_score_logits.sigmoid() > 0.5)[:, 0]
            .float()
            .clone()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        self.meter_container["pred_existence"] = pred_existence_list

        pred_scores_list = self.meter_container.get("pred_scores", [])
        pred_scores_list.extend(
            object_score_logits.sigmoid()[:, 0]
            .float()
            .clone()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        self.meter_container["pred_scores"] = pred_scores_list

        ################################################################################
        loss_2d_bbox_iou = bbox_2d_giou_loss(
            box_cxcywh_to_xyxy(pred_bboxes_2d),
            gt_bboxes_2d,
            has_object_indicator,
            self.resolution,
        )
        loss_2d_bbox_l1 = bbox_2d_l1_loss(
            pred_bboxes_2d,
            box_xyxy_to_cxcywh(gt_bboxes_2d),
            has_object_indicator,
            self.resolution,
        )
        loss_3d_bbox_center_in_2d_l1 = bbox_3d_center_in_2d_loss(
            pred_bboxes_3d,
            gt_bboxes_3d,
            gt_intrinsic,
            self.resolution,
            has_object_indicator,
        )
        loss_3d_bbox_depth = bbox_3d_depth_loss(
            pred_bboxes_3d,
            gt_bboxes_3d,
            has_object_indicator,
            log_depth_error=self.log_depth_error,
        )
        loss_3d_bbox_dim = bbox_3d_dim_loss(
            pred_bboxes_3d, gt_bboxes_3d, has_object_indicator
        )
        loss_3d_bbox_l1 = bbox_3d_l1_loss(
            pred_bboxes_3d,
            gt_bboxes_3d,
            has_object_indicator,
        )
        loss_3d_bbox_alpha = bbox_3d_alpha_loss(
            pred_bboxes_3d_alpha_cls,
            gt_bboxes_3d_alpha_cls,
            pred_bboxes_3d_alpha_res,
            gt_bboxes_3d_alpha_res,
            has_object_indicator,
        )
        loss_3d_bbox_quat = bbox_3d_quat_loss(
            pred_bboxes_quat,
            gt_bboxes_quat,
            has_object_indicator,
        )
        loss_3d_chamfer = bbox_chamfer_loss(
            pred_bboxes_3d,
            pred_bboxes_quat,
            gt_bboxes_3d,
            gt_bboxes_quat,
            has_object_indicator,
        )
        loss_3d_bbox_ious = bbox_3d_iou_loss(
            pred_bboxes_3d,
            gt_bboxes_3d,
            pred_ious,
            pred_bboxes_3d_alpha_cls,
            pred_bboxes_3d_alpha_res,
            gt_bboxes_3d_alpha_cls,
            gt_bboxes_3d_alpha_res,
            gt_intrinsic,
            has_object_indicator,
            pred_bboxes_quat,
            gt_bboxes_quat,
            self.log_iou,
            self.log_yaw_error,
        )
        ################################################################################
        ################################################################################
        gt_h, gt_w = gt_depth_map.shape[-2:]
        if pred_depth_maps.shape != gt_depth_map.shape:
            gt_depth_map = F.interpolate(
                gt_depth_map.unsqueeze(1),
                size=pred_depth_maps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        loss_depth_map = depth_map_loss(
            pred_depth_maps,
            pred_depth_confs,
            gt_depth_map,
            self.depth_sil_coefficient,
        )
        loss_depth_gradient = compute_gradient_loss(
            pred_depth_maps.unsqueeze(-1),
            gt_depth_map.unsqueeze(-1),
        )
        loss_camera = camera_loss(
            pred_pose_encs,
            gt_extrinsic,
            gt_intrinsic,
            (gt_h, gt_w),
            log_pose=self.log_pose,
        )
        loss_class = sigmoid_focal_loss(
            object_score_logits,
            has_object_indicator[..., None],
            alpha=self.focal_alpha_obj_score,
            gamma=self.focal_gamma_obj_score,
        )
        ################################################################################
        # log some statistics
        assert self.log_depth_error, "depth error logging is required"
        loss_3d_bbox_depth, depth_errors, avg_depth_error = loss_3d_bbox_depth
        # confidence is in (0, +inf), normalize to (0, 1) using x/(x+1)
        confidence = pred_bboxes_3d[:, :, -1].clamp(min=1e-6)
        confidence = confidence / (confidence + 1)
        best_pred_ious = (pred_ious * confidence).argmax(dim=-1)
        depth_errors = depth_errors[torch.arange(depth_errors.size(0)), best_pred_ious]
        depth_errors_list = self.meter_container.get("depth_errors", [])
        depth_errors_list.extend(
            depth_errors.float().clone().detach().cpu().numpy().tolist()
        )
        self.meter_container["depth_errors"] = depth_errors_list
        losses["avg_depth_error"] = avg_depth_error.clone().detach()

        # translation error
        translation_errors_list = self.meter_container.get("translation_errors", [])
        translation_error = (
            pred_bboxes_3d[torch.arange(depth_errors.size(0)), best_pred_ious, :3]
            - gt_bboxes_3d[:, :3]
        ).norm(dim=-1)
        translation_error = translation_error * has_object_indicator.squeeze(-1)
        translation_errors_list.extend(
            translation_error.float().clone().detach().cpu().numpy().tolist()
        )
        self.meter_container["translation_errors"] = translation_errors_list
        losses["avg_translation_error"] = translation_error.mean().clone().detach()

        # scale error: compute "dimension IoU" as a proxy for scale similarity
        # Note: This is NOT true 3D volume IoU, but measures how similar the
        # predicted dimensions (w,h,l) are to ground truth dimensions.
        # Formula: 1 - prod(min(pred,gt)) / (prod(pred) + prod(gt) - prod(min(pred,gt)))
        scale_errors_list = self.meter_container.get("scale_errors", [])
        pred_dims = pred_bboxes_3d[
            torch.arange(depth_errors.size(0)), best_pred_ious, 3:6
        ]
        gt_dims = gt_bboxes_3d[:, 3:6]
        min_whl = torch.minimum(pred_dims, gt_dims)
        min_eps = torch.tensor(1e-6, device=gt_dims.device)
        volume_gt = torch.maximum(min_eps, torch.prod(gt_dims, dim=-1))
        volume_pred = torch.maximum(min_eps, torch.prod(pred_dims, dim=-1))
        volume_intersect = torch.maximum(min_eps, torch.prod(min_whl, dim=-1))
        scale_error = 1 - (
            volume_intersect / (volume_gt + volume_pred - volume_intersect)
        )
        scale_error = scale_error.to(pred_bboxes_3d.device)
        scale_error = scale_error * has_object_indicator.squeeze(-1)
        scale_errors_list.extend(
            scale_error.float().clone().detach().cpu().numpy().tolist()
        )
        self.meter_container["scale_errors"] = scale_errors_list
        losses["avg_scale_error"] = scale_error.mean().clone().detach()

        if self.log_yaw_error and self.log_iou:
            (
                loss_3d_bbox_ious,
                ious,
                avg_iou,
                yaw_errors,
                avg_yaw_error,
            ) = loss_3d_bbox_ious
            losses["avg_iou"] = avg_iou.clone().detach()
            ious_list = self.meter_container.get("ious", [])
            ious_list.extend(ious.float().clone().detach().cpu().numpy().tolist())
            self.meter_container["ious"] = ious_list

            yaw_errors_list = self.meter_container.get("yaw_errors", [])
            yaw_errors_list.extend(
                yaw_errors.float().clone().detach().cpu().numpy().tolist()
            )
            self.meter_container["yaw_errors"] = yaw_errors_list
            losses["avg_yaw_error"] = avg_yaw_error.clone().detach()
        if self.log_pose:
            loss_camera, loss_T, loss_R, loss_FL, T_error, R_error, FL_error = (
                loss_camera
            )
            losses["loss_T"] = loss_T.clone().detach()
            losses["loss_R"] = loss_R.clone().detach()
            losses["loss_FL"] = loss_FL.clone().detach()

            T_error_list = self.meter_container.get("T_error", [])
            T_error_list.extend(T_error.float().clone().detach().cpu().numpy().tolist())
            self.meter_container["T_error"] = T_error_list
            R_error_list = self.meter_container.get("R_error", [])
            R_error_list.extend(R_error.float().clone().detach().cpu().numpy().tolist())
            self.meter_container["R_error"] = R_error_list
            FL_error_list = self.meter_container.get("FL_error", [])
            FL_error_list.extend(
                FL_error.float().clone().detach().cpu().numpy().tolist()
            )
            self.meter_container["FL_error"] = FL_error_list
        ################################################################################

        # for multimask prediction, select the mask with the smallest loss for back-prop
        assert loss_2d_bbox_iou.size(0) > 1, (
            "currently only supports multimask prediction"
        )

        # take the mask indices with the smallest focal + dice loss for back propagation
        loss_combo = (
            loss_3d_bbox_center_in_2d_l1
            * self.weight_dict["loss_3d_bbox_center_in_2d_l1"]
            + loss_3d_bbox_depth * self.weight_dict["loss_3d_bbox_depth"]
            + loss_3d_bbox_dim * self.weight_dict["loss_3d_bbox_dim"]
            + loss_3d_bbox_alpha * self.weight_dict["loss_3d_bbox_alpha"]
            + loss_3d_bbox_l1 * self.weight_dict["loss_3d_bbox_l1"]
        )
        # confidence is in (0, +inf), normalize to (0, 1) using x/(x+1)
        confidence = pred_bboxes_3d[:, :, -1].clamp(min=1e-6)
        confidence = confidence / (confidence + 1)
        best_loss_inds = torch.argmin(loss_combo, dim=0)
        best_bbox_conf_inds = torch.argmax(
            pred_ious * confidence, dim=-1
        )  # N, num pred
        arange_N = torch.arange(
            best_bbox_conf_inds.size(0), device=best_bbox_conf_inds.device
        )
        loss_2d_bbox_iou = (
            loss_2d_bbox_iou[best_loss_inds, arange_N]
            + loss_2d_bbox_iou[best_bbox_conf_inds, arange_N]
        )
        loss_2d_bbox_l1 = (
            loss_2d_bbox_l1[best_loss_inds, arange_N]
            + loss_2d_bbox_l1[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_center_in_2d_l1 = (
            loss_3d_bbox_center_in_2d_l1[best_loss_inds, arange_N]
            + loss_3d_bbox_center_in_2d_l1[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_depth = (
            loss_3d_bbox_depth[best_loss_inds, arange_N]
            + loss_3d_bbox_depth[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_dim = (
            loss_3d_bbox_dim[best_loss_inds, arange_N]
            + loss_3d_bbox_dim[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_alpha = (
            loss_3d_bbox_alpha[best_loss_inds, arange_N]
            + loss_3d_bbox_alpha[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_quat = (
            loss_3d_bbox_quat[best_loss_inds, arange_N]
            + loss_3d_bbox_quat[best_bbox_conf_inds, arange_N]
        )
        loss_3d_bbox_l1 = (
            loss_3d_bbox_l1[best_loss_inds, arange_N]
            + loss_3d_bbox_l1[best_bbox_conf_inds, arange_N]
        )
        loss_3d_chamfer = (
            loss_3d_chamfer[best_loss_inds, arange_N]
            + loss_3d_chamfer[best_bbox_conf_inds, arange_N]
        )
        # calculate the iou prediction and slot losses only in the index
        # with the minimum loss for each mask (to be consistent w/ SAM)

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_2d_bbox_iou"] += loss_2d_bbox_iou.sum()
        losses["loss_2d_bbox_l1"] += loss_2d_bbox_l1.sum()
        losses["loss_3d_bbox_center_in_2d_l1"] += loss_3d_bbox_center_in_2d_l1.sum()
        losses["loss_3d_bbox_depth"] += loss_3d_bbox_depth.sum()
        losses["loss_3d_bbox_dim"] += loss_3d_bbox_dim.sum()
        losses["loss_3d_bbox_alpha"] += loss_3d_bbox_alpha.sum()
        losses["loss_3d_bbox_quat"] += loss_3d_bbox_quat.sum()
        losses["loss_3d_bbox_ious"] += loss_3d_bbox_ious.sum()
        losses["loss_3d_chamfer"] += loss_3d_chamfer.sum()
        losses["loss_depth_map"] += loss_depth_map
        losses["loss_depth_gradient"] += loss_depth_gradient
        losses["loss_camera_pose"] += loss_camera
        losses["loss_class"] += loss_class
        losses["loss_3d_bbox_l1"] += loss_3d_bbox_l1.sum()
        # losses["loss_depth_map_aux"] += loss_depth_map_aux
        # losses["loss_camera_pose_aux"] += loss_camera_aux

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
