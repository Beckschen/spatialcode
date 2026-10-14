# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from torch.nn import functional as F
from modeling.sam2_base import SAM2Base
from modeling.sam2_utils import sample_points_around_box_center, add_noise_to_bbox

from utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_center_input_for_train=0.0,
        prob_to_use_box_center_input_for_eval=0.0,
        num_points_to_track=500,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_center_input_for_train = (
            prob_to_use_box_center_input_for_train
        )
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_center_input_for_eval = (
            prob_to_use_box_center_input_for_eval
        )
        if hasattr(self, "spatial_encoder"):
            self.num_points_to_track = num_points_to_track
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        if hasattr(self, "spatial_encoder"):
            spatial_out = self.forward_video_3d(input.spatial_img_batch)[
                "spatial_features"
            ]
            frame_idx = -1  # indicate that all frames are used
        else:
            spatial_out = None
            frame_idx = None
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(
                input.flat_img_batch,
                video_feature_dict=spatial_out,
                frame_idx=frame_idx,
            )
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out.update({"spatial_features": spatial_out})
        # this is gt depth; used for debugging only
        # depth_maps = F.interpolate(
        #     input.depth_batch[None],
        #     size=backbone_out["spatial_features"]["depth_conf"].shape[-2:],
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # backbone_out["spatial_features"].update({"depth_map": depth_maps})
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(
        self, img_batch, img_ids, spatial_out=None
    ):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image, spatial_out, unique_img_ids)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
            (ori_vision_feats, ori_vision_pos_embeds, ori_feat_sizes),
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]
            ori_vision_feats = [x[:, inv_ids] for x in ori_vision_feats]
            ori_vision_pos_embeds = [x[:, inv_ids] for x in ori_vision_pos_embeds]
            ori_feat_sizes = [x for x in ori_feat_sizes]

        return (
            image,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
            (ori_vision_feats, ori_vision_pos_embeds, ori_feat_sizes),
        )

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_bboxes_per_frame = {
            frame_id: bboxes  # [B, 4]
            for frame_id, bboxes in enumerate(input.gt_batch.gt_bboxes_2d)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_bboxes_per_frame"] = gt_bboxes_per_frame
        num_frames = input.num_frames
        object_appearance = input.obj_to_frame_idx

        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_center_input = self.prob_to_use_box_center_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_center_input = self.prob_to_use_box_center_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            additional_frames = self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
            additional_frames.sort()
            init_cond_frames = [start_frame_idx] + additional_frames
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["bbox_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}

        for t in init_cond_frames:
            points = gt_bboxes_per_frame[t].reshape(-1, 2, 2)
            if self.rng.random() < 0.7 and self.training:
                points = add_noise_to_bbox(points)
            labels = torch.tensor(
                [[2, 3]], dtype=torch.int, device=points.device
            ).repeat(points.shape[0], 1)
            if use_pt_input:
                # print("For this batch, we use points as prompt input.")
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_center_input
                use_box_center = self.rng.random() < prob_to_use_box_center_input
                if use_box_center:
                    new_points, new_labels = sample_points_around_box_center(
                        gt_bboxes_per_frame[t], noise=0
                    )
                    points = torch.cat([points, new_points], dim=1)
                    labels = torch.cat([labels, new_labels], dim=1)
                # random sample any additional points around the box center
                add_other_points = (
                    self.rng.random() < prob_to_use_box_center_input * 0.2
                )
                if (
                    add_other_points and self.training
                ):  # add other points only during training
                    num_points = self.rng.integers(1, 3)
                elif (
                    not use_box_center and self.training
                ):  # if not using box center, we at least sample one point
                    num_points = 1
                else:
                    num_points = 0
                if not self.training:
                    assert num_points == 0 and use_box_center
                for _ in range(num_points):
                    new_points, new_labels = sample_points_around_box_center(
                        bboxes=gt_bboxes_per_frame[t], noise=0.3
                    )
                    points = torch.cat([points, new_points], dim=1)
                    labels = torch.cat([labels, new_labels], dim=1)

            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks

        if num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
                (ori_vision_feats, ori_vision_pos_embeds, ori_feat_sizes),
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
                current_sam2_vision_feats = [x[:, img_ids] for x in ori_vision_feats]
                current_sam2_vision_pos_embeds = [
                    x[:, img_ids] for x in ori_vision_pos_embeds
                ]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                spatial_out = backbone_out.get("spatial_features")
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                    (
                        current_sam2_vision_feats,
                        current_sam2_vision_pos_embeds,
                        ori_feat_sizes,
                    ),
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids, spatial_out
                )

            point_inputs_per_frame = backbone_out["point_inputs_per_frame"].get(
                stage_id, None
            )
            if point_inputs_per_frame is not None:
                point_coords = point_inputs_per_frame["point_coords"]
                if hasattr(self, "spatial_encoder") and hasattr(
                    self.spatial_encoder, "track_head"
                ):
                    track_feature = self._forward_3d_tracking_head(
                        backbone_out["spatial_features"],
                        input.spatial_img_batch,
                        point_coords,
                        self.num_points_to_track,
                        input.flat_img_batch.shape[-2:],
                    )
                    backbone_out["spatial_features"].update(
                        {"track_feats": track_feature}
                    )
            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                current_sam2_vision_feats=current_sam2_vision_feats,
                current_sam2_vision_pos_embeds=current_sam2_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs_per_frame,
                mask_inputs=None,
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                spatial_features=backbone_out["spatial_features"]
                if backbone_out.get("spatial_features", None) is not None
                else None,
                gt_2d_bboxes=input.gt_batch.gt_bboxes_2d[stage_id],
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        current_sam2_vision_feats,
        current_sam2_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_2d_bboxes=None,
        spatial_features=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        (
            current_out,
            sam_outputs,
            pix_feat,
            high_res_features,
            ori_pixel_feature,
            original_high_res_features,
        ) = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            current_sam2_vision_feats,
            current_sam2_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            spatial_features,
        )

        low_res_masks = sam_outputs["low_res_masks"]
        high_res_masks = sam_outputs["high_res_masks"]
        obj_ptr = sam_outputs["obj_ptr"]
        object_score_logits = sam_outputs["object_score_logits"]
        low_res_multimasks = sam_outputs["low_res_multimasks"]
        high_res_multimasks = sam_outputs["high_res_multimasks"]
        ious = sam_outputs["mask_ious"]
        current_out["obj_ptr"] = obj_ptr

        # original predicted keys:
        # current_out["multistep_pred_masks"] = low_res_masks
        # current_out["multistep_pred_masks_high_res"] = high_res_masks
        # current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        # current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        # current_out["multistep_pred_multimask_ious"] = [ious]
        # current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        # The keys for bbox detection:
        current_out["multistep_pred_multibbox_2d"] = sam_outputs["pred_multibboxes_2d"]
        current_out["multistep_pred_multibbox_3d"] = sam_outputs["pred_multibboxes_3d"]
        current_out["multistep_pred_multibbox_alpha_cls"] = sam_outputs[
            "pred_multibboxes_alpha_cls"
        ]
        current_out["multistep_pred_multibbox_alpha_res"] = sam_outputs[
            "pred_multibboxes_alpha_res"
        ]
        current_out["multistep_pred_multibbox_quat"] = sam_outputs[
            "pred_multibboxes_quat"
        ]
        current_out["multistep_pred_bbox_2d"] = sam_outputs["pred_bbox_2d"]
        current_out["multistep_pred_bbox_3d"] = sam_outputs["pred_bbox_3d"]
        current_out["multistep_pred_bbox_alpha_cls"] = sam_outputs[
            "pred_bbox_alpha_cls"
        ]
        current_out["multistep_pred_bbox_alpha_res"] = sam_outputs[
            "pred_bbox_alpha_res"
        ]
        current_out["multistep_pred_bbox_quat"] = sam_outputs["pred_bbox_quat"]
        current_out["multistep_pred_depth_map"] = sam_outputs["pred_depth_map"]
        # current_out["multistep_high_res_depth_map"] = sam_outputs["high_res_depth_map"]
        current_out["multistep_pred_depth_conf"] = sam_outputs["pred_depth_conf"]
        current_out["multistep_pred_pose_enc"] = sam_outputs["pred_pose_enc"]
        current_out["multistep_pred_bbox_ious"] = sam_outputs["pred_bbox_ious"]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            track_feat = spatial_features.get("track_feats", None)
            scale_feat = spatial_features.get("scale_feat", None)
            spatial_features = {
                "camera_feat": spatial_features["camera_feat"][0, frame_idx]
                if spatial_features is not None
                else None,
                "pose_enc": spatial_features["pose_enc"][:, frame_idx]  # 1,9
                if spatial_features is not None
                else None,
                "depth_feat": spatial_features["depth_feat"][0, frame_idx]  # c,h,w
                if spatial_features is not None
                else None,
                "depth_map": spatial_features["depth_map"][:, frame_idx, ..., 0][
                    :, None
                ]  # 1,1,h,w
                if spatial_features is not None
                else None,
                "depth_conf": spatial_features["depth_conf"][0, frame_idx]  # h,w
                if spatial_features is not None
                else None,
                "track_feat": track_feat[0, frame_idx]
                if track_feat is not None
                else None,
                "scale_feat": scale_feat[:, :, 0] if scale_feat is not None else None,
                "scale": spatial_features.get("scale", None)
                if spatial_features is not None
                else None,
            }
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_2d_bboxes,
                high_res_features,
                original_high_res_features,
                pix_feat,
                ori_pixel_feature,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
                spatial_features,
            )

            # Use the final prediction (after all correction steps for output and eval)
            current_out["pred_masks"] = final_sam_outputs["low_res_masks"]
            current_out["pred_masks_high_res"] = final_sam_outputs["high_res_masks"]
            current_out["obj_ptr"] = final_sam_outputs["obj_ptr"]

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_2d_bboxes,
        high_res_features,
        original_high_res_features,
        pix_feat_with_mem,
        ori_pixel_feature_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
        spatial_features,
    ):
        assert gt_2d_bboxes is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        # for _ in range(self.num_correction_pt_per_frame):
        # sample a new point from the error between prediction and ground-truth
        # (with a small probability, directly sample from GT masks instead of errors)
        # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
        if self.rng.random() < 0.5 and self.training:
            # add new point around box center
            new_points, new_labels = sample_points_around_box_center(
                bboxes=gt_2d_bboxes, noise=0.3
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
        if self.rng.random() < 0.5 and self.training:
            # add new bbox
            gt_bboxes = gt_2d_bboxes.reshape(-1, 2, 2)
            new_points = add_noise_to_bbox(gt_bboxes)
            new_labels = torch.tensor(
                [[2, 3]], dtype=torch.int, device=new_points.device
            ).repeat(new_points.shape[0], 1)
            point_inputs = concat_points(point_inputs, new_points, new_labels)

        # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
        # For tracking, this means that when the user adds a correction click, we also feed
        # the tracking output mask logits along with the click as input to the SAM decoder.
        mask_inputs = low_res_masks
        multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
        if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
            sam_outputs = torch.utils.checkpoint.checkpoint(
                self._forward_sam_heads,
                backbone_features=pix_feat_with_mem,
                original_backbone_features=ori_pixel_feature_with_mem,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_features,
                original_high_res_features=original_high_res_features,
                multimask_output=multimask_output,
                use_reentrant=False,
                spatial_out=spatial_features,
            )
        else:
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                original_backbone_features=ori_pixel_feature_with_mem,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_features,
                original_high_res_features=original_high_res_features,
                multimask_output=multimask_output,
                spatial_out=spatial_features,
            )

        all_pred_masks.append(sam_outputs["low_res_masks"])
        all_pred_high_res_masks.append(sam_outputs["high_res_masks"])
        all_pred_multimasks.append(sam_outputs["low_res_multimasks"])
        all_pred_high_res_multimasks.append(sam_outputs["high_res_multimasks"])
        all_pred_ious.append(sam_outputs["mask_ious"])
        all_point_inputs.append(point_inputs)
        all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs
