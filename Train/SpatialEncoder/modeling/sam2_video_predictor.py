# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch

from tqdm import tqdm

from modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from modeling.tracking_utils import QuaternionSmoother
from utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames
from modeling.sam2_utils import (
    pose_encoding_to_extri_intri,
    project_to_so3,
    quat_to_rotmat,
    rotmat_to_quat,
)


class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        num_points_to_track: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(
        self,
        video_path=None,
        video_frames=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        if (video_path is None) == (video_frames is None):
            raise ValueError(
                "exactly one of video_path or video_frames should be provided"
            )
        if video_path is not None:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
            )
        elif video_frames is not None:
            images = video_frames["rgb_images"].to(device=compute_device)
            if offload_video_to_cpu:
                images = images.cpu()
            video_height = video_width = 1024
            video_height = int(video_height)
            video_width = int(video_width)

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Extract 3D spatial features (VGGT)
        # (the patch size of VGGT is 14m so the resolution of image must be the multiply of 14)
        # For streaming mode, we allow lazy initialization of spatial features
        inference_state["spatial_features_initialized"] = False
        if video_frames.get("spatial_images", None) is not None:
            # Store spatial images without processing immediately (for streaming)
            spatial_images = video_frames["spatial_images"]
            if isinstance(spatial_images, torch.Tensor) and spatial_images.dim() == 4:
                spatial_images = spatial_images.unsqueeze(0).to(device=compute_device)
            elif isinstance(spatial_images, torch.Tensor) and spatial_images.dim() == 5:
                spatial_images = spatial_images.to(device=compute_device)
            elif isinstance(spatial_images, list):
                assert self.spatial_encoder.spatial_name == "mapanything"
            inference_state["image_for_spatial"] = spatial_images

            # Only compute spatial features immediately if not in streaming mode
            # (i.e., if all frames are provided at once)
            if not video_frames.get("streaming_mode", False) and hasattr(self, "spatial_encoder"):
                spatial_out_dict = self.forward_video_3d(spatial_images)
                inference_state.update(spatial_out_dict)
                inference_state["spatial_features_initialized"] = True

        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from modeling.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects (including after tracking starts).
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["frames_tracked_per_obj"][obj_idx] = {}
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)
        if point_inputs_per_frame is not None:
            point_coords = point_inputs["point_coords"]
            if hasattr(self, "spatial_encoder") and hasattr(self.spatial_encoder, "track_head"):
                track_feature = self._forward_3d_tracking_head(
                    inference_state["spatial_features"],
                    inference_state["image_for_spatial"],
                    point_coords,
                    self.num_points_to_track,
                    (self.image_size, self.image_size),
                )
                inference_state["spatial_features"].update(
                    {"track_feats": track_feature}
                )

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def update_spatial_features(
        self, inference_state, spatial_images, clip_frame_offset=0
    ):
        """
        Update spatial features for a new clip in streaming mode.

        Args:
            inference_state: The inference state dict
            spatial_images: Spatial images for the current clip (can be tensor or list)
            clip_frame_offset: Frame offset for this clip in the global video timeline
        """
        compute_device = inference_state["device"]

        # Normalize input type and shape
        if isinstance(spatial_images, torch.Tensor):
            if spatial_images.dim() == 4:
                # (F,C,H,W) -> (1,F,C,H,W)
                spatial_images = spatial_images.unsqueeze(0).to(device=compute_device)
            elif spatial_images.dim() == 5:
                # (B,F,C,H,W)
                spatial_images = spatial_images.to(device=compute_device)
        elif isinstance(spatial_images, list):
            # mapanything expects a list; leave as-is
            assert self.spatial_encoder.spatial_name == "mapanything"

        # Determine frame count and early-return if zero to avoid downstream reshape errors
        num_frames = 0
        if isinstance(spatial_images, torch.Tensor):
            # Expect (B,F,C,H,W)
            if spatial_images.dim() == 5:
                num_frames = int(spatial_images.shape[1])
            else:
                num_frames = 0
        elif isinstance(spatial_images, list):
            num_frames = len(spatial_images)

        if num_frames == 0:
            return {}

        # Only keep current clip's spatial features (no accumulation)
        # Store the global frame offset for this clip
        spatial_out_clip = self.forward_video_3d(spatial_images)
        inference_state["image_for_spatial"] = spatial_images
        inference_state["spatial_features"] = spatial_out_clip["spatial_features"]
        inference_state["spatial_frame_offset"] = clip_frame_offset
        inference_state["spatial_features_initialized"] = True

        # Track spatial frames count for alignment diagnostics
        if isinstance(inference_state.get("image_for_spatial"), torch.Tensor):
            if inference_state["image_for_spatial"].dim() == 5:
                inference_state["spatial_num_frames"] = int(
                    inference_state["image_for_spatial"].shape[1]
                )
            else:
                inference_state["spatial_num_frames"] = int(
                    inference_state["image_for_spatial"].shape[0]
                )
        elif isinstance(inference_state.get("image_for_spatial"), list):
            inference_state["spatial_num_frames"] = len(
                inference_state["image_for_spatial"]
            )

        return {"spatial_features": inference_state.get("spatial_features", {})}

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        if not hasattr(self, "pose_tracker") or self.pose_tracker is None:
            self.pose_tracker = QuaternionSmoother(device=inference_state["device"])
        else:
            self.pose_tracker.reset()
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            pred_bbox_2d_per_obj = [None] * batch_size
            pred_bbox_3d_per_obj = [None] * batch_size
            pred_bbox_alpha_res_per_obj = [None] * batch_size
            pred_bbox_alpha_cls_per_obj = [None] * batch_size
            pred_bbox_quat_per_obj = [None] * batch_size
            has_object_per_obj = [False] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    pred_bbox_2d = current_out["pred_bbox_2d"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_3d = current_out["pred_bbox_3d"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_alpha_cls = current_out["pred_bbox_alpha_cls"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_alpha_res = current_out["pred_bbox_alpha_res"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_quat = current_out["pred_bbox_quat"].to(
                        device, non_blocking=True
                    )
                else:
                    device = inference_state["device"]
                    storage_key = "non_cond_frame_outputs"
                    current_out = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    pred_bbox_2d = current_out["pred_bbox_2d"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_3d = current_out["pred_bbox_3d"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_alpha_cls = current_out["pred_bbox_alpha_cls"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_alpha_res = current_out["pred_bbox_alpha_res"].to(
                        device, non_blocking=True
                    )
                    pred_bbox_quat = current_out["pred_bbox_quat"].to(
                        device, non_blocking=True
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
                pred_bbox_2d_per_obj[obj_idx] = pred_bbox_2d
                pred_bbox_3d_per_obj[obj_idx] = pred_bbox_3d
                pred_bbox_alpha_res_per_obj[obj_idx] = pred_bbox_alpha_res
                pred_bbox_alpha_cls_per_obj[obj_idx] = pred_bbox_alpha_cls
                pred_bbox_quat_per_obj[obj_idx] = pred_bbox_quat
                has_object_per_obj[obj_idx] = (
                    current_out["object_score_logits"] > 0
                )
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                # 多个对象 → 按 batch 维拼接
                # 在iou处已经做好了选择
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                all_pred_bbox_2d = torch.cat(pred_bbox_2d_per_obj, dim=0)
                all_pred_bbox_3d = torch.cat(pred_bbox_3d_per_obj, dim=0)
                all_pred_bbox_alpha_res = torch.cat(pred_bbox_alpha_res_per_obj, dim=0)
                all_pred_bbox_alpha_cls = torch.cat(pred_bbox_alpha_cls_per_obj, dim=0)
                all_pred_bbox_quat = torch.cat(pred_bbox_quat_per_obj, dim=0)
                has_object = torch.tensor(
                    has_object_per_obj, dtype=torch.bool, device=all_pred_masks.device
                )
            else:
                # 单对象 → 直接取第一个
                all_pred_masks = pred_masks_per_obj[0]
                all_pred_bbox_2d = pred_bbox_2d_per_obj[0]
                all_pred_bbox_3d = pred_bbox_3d_per_obj[0]
                all_pred_bbox_alpha_res = pred_bbox_alpha_res_per_obj[0]
                all_pred_bbox_alpha_cls = pred_bbox_alpha_cls_per_obj[0]
                all_pred_bbox_quat = pred_bbox_quat_per_obj[0]
                has_object = torch.tensor(
                    has_object_per_obj, dtype=torch.bool, device=all_pred_masks.device
                )
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )

            yield {
                "frame_idx": frame_idx,
                "obj_ids": obj_ids,
                "has_object": has_object,
                "intrinsics": current_out["intrinsics"].clone(),
                "extrinsics": current_out["extrinsics"].clone(),
                "depth_map": current_out["depth_map"].clone(),
                "depth_conf": current_out["depth_conf"].clone(),
                "video_res_masks": video_res_masks,
                "all_pred_bbox_2d": all_pred_bbox_2d,
                "all_pred_bbox_3d": all_pred_bbox_3d,
                "all_pred_bbox_alpha_res": all_pred_bbox_alpha_res,
                "all_pred_bbox_alpha_cls": all_pred_bbox_alpha_cls,
                "all_pred_bbox_quat": all_pred_bbox_quat,
            }

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            # The frame is not a conditioning frame anymore since it's not receiving inputs,
            # so we "downgrade" its output (if exists) to a non-conditioning frame output.
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            # Validate frame index and image availability
            if not isinstance(
                inference_state.get("images"), (list, tuple, torch.Tensor)
            ):
                raise RuntimeError(
                    "inference_state['images'] is not initialized or has invalid type"
                )
            if isinstance(inference_state["images"], (list, tuple)):
                if frame_idx < 0 or frame_idx >= len(inference_state["images"]):
                    raise IndexError(
                        f"Frame index {frame_idx} out of range for images length {len(inference_state['images'])}"
                    )
                image_tensor = inference_state["images"][frame_idx]
            else:
                # Tensor of shape (T, C, H, W)
                if frame_idx < 0 or frame_idx >= inference_state["images"].size(0):
                    raise IndexError(
                        f"Frame index {frame_idx} out of range for images tensor with {inference_state['images'].size(0)} frames"
                    )
                image_tensor = inference_state["images"][frame_idx]
            if image_tensor is None or (
                isinstance(image_tensor, torch.Tensor) and image_tensor.numel() == 0
            ):
                raise RuntimeError(
                    f"Empty image tensor at frame {frame_idx}; check streaming image padding and offsets"
                )
            image = image_tensor.to(device).float().unsqueeze(0)
            video_dict = inference_state.get("spatial_features", None)
            backbone_out = self.forward_image(
                image, video_feature_dict=video_dict, frame_idx=frame_idx
            )
            # Sanity check: ensure backbone features are non-empty
            last_feat = backbone_out["backbone_fpn"][-1]
            if not isinstance(last_feat, torch.Tensor) or last_feat.numel() == 0:
                raise RuntimeError(
                    "Backbone returned empty features on frame "
                    f"{frame_idx}; verify image sizes and streaming indices"
                )
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
            "spatial_features": inference_state.get("spatial_features", None),
            "spatial_frame_offset": inference_state.get("spatial_frame_offset", 0),
            "original_sam2": {
                "vision_pos_enc": backbone_out["original_sam2"]["vision_pos_enc"],
                "backbone_fpn": backbone_out["original_sam2"]["backbone_fpn"],
            },
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            ).contiguous()
            ori_feat = expanded_backbone_out["original_sam2"]["backbone_fpn"][i]
            expanded_backbone_out["original_sam2"]["backbone_fpn"][i] = ori_feat.expand(
                batch_size, -1, -1, -1
            ).contiguous()
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1).contiguous()
            expanded_backbone_out["vision_pos_enc"][i] = pos
            original_pos = expanded_backbone_out["original_sam2"]["vision_pos_enc"][i]
            expanded_backbone_out["original_sam2"]["vision_pos_enc"][i] = original_pos.expand(
                batch_size, -1, -1, -1
            ).contiguous()

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            backbone_out,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            (ori_vision_feats, ori_vision_pos_embeds, ori_feat_sizes),
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            current_sam2_vision_feats=ori_vision_feats,
            current_sam2_vision_pos_embeds=ori_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            spatial_features=backbone_out["spatial_features"],
            spatial_frame_offset=backbone_out.get("spatial_frame_offset", 0),
        )
        for k, v in current_out.items():
            if isinstance(v, torch.Tensor):
                current_out[k] = v.detach()

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"].to(storage_device, non_blocking=True)
        object_score_logits = current_out["object_score_logits"].to(storage_device, non_blocking=True)
        # make a compact version of this frame's output to reduce the state size

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            current_out["pred_pose_enc"].to(storage_device, non_blocking=True),
            image_size_hw=(self.image_size, self.image_size),
        )
        extrinsics[:, :3, :3] = project_to_so3(extrinsics[:, :3, :3])
        compact_current_out = {
            "depth_map": current_out["pred_depth_map"].to(
                storage_device, non_blocking=True
            ),
            "depth_conf": current_out["pred_depth_conf"].to(
                storage_device, non_blocking=True
            ),
            "extrinsics": extrinsics.to(storage_device, non_blocking=True),
            "intrinsics": intrinsics.to(storage_device, non_blocking=True),
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "pred_bbox_2d": current_out["pred_bbox_2d"].to(
                storage_device, non_blocking=True
            ),
            "pred_bbox_3d": current_out["pred_bbox_3d"].to(
                storage_device, non_blocking=True
            ),
            "pred_bbox_alpha_cls": current_out["pred_bbox_alpha_cls"].to(
                storage_device, non_blocking=True
            ),
            "pred_bbox_alpha_res": current_out["pred_bbox_alpha_res"].to(
                storage_device, non_blocking=True
            ),
            "pred_bbox_quat": current_out["pred_bbox_quat"].to(
                storage_device, non_blocking=True
            ),
        }
        del current_out
        return compact_current_out

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, bbox_current_vision_feats, _, feat_sizes, (current_vision_feats, _, _) = (
            self._get_image_feature(inference_state, frame_idx, batch_size)
        )
        # Guard against zero-sized vision features to avoid invalid reshape
        top_feat = current_vision_feats[-1]
        if not isinstance(top_feat, torch.Tensor) or top_feat.numel() == 0:
            raise RuntimeError(
                "Zero-sized vision features before memory encoding on frame "
                f"{frame_idx}; verify image feature computation and streaming indices"
            )
        if feat_sizes[-1][0] == 0 or feat_sizes[-1][1] == 0:
            raise RuntimeError(
                f"Invalid feat_sizes {feat_sizes[-1]} for memory encoder; got zero height/width"
            )

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc
    
    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)


# class SAM2VideoPredictorVOS(SAM2VideoPredictor):
#     """Optimized for the VOS setting"""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._compile_all_components()

#     def _compile_all_components(self):
#         print("Compiling all components for VOS setting. First time may be very slow.")
#         self.memory_encoder.forward = torch.compile(
#             self.memory_encoder.forward,
#             mode="max-autotune",
#             fullgraph=True,
#             dynamic=False,
#         )

#         self.memory_attention.forward = torch.compile(
#             self.memory_attention.forward,
#             mode="max-autotune",
#             fullgraph=True,
#             dynamic=True,  # Num. of memories varies
#         )

#         self.sam_prompt_encoder.forward = torch.compile(
#             self.sam_prompt_encoder.forward,
#             mode="max-autotune",
#             fullgraph=True,
#             dynamic=False,  # Accuracy regression on True
#         )

#         self.sam_mask_decoder.forward = torch.compile(
#             self.sam_mask_decoder.forward,
#             mode="max-autotune",
#             fullgraph=True,
#             dynamic=False,  # Accuracy regression on True
#         )

#     def forward_image(self, img_batch: torch.Tensor, video_tensor: torch.Tensor = None):
#         """
#         Identical to the corresponding method in the parent (SAM2VideoPredictor), but
#         cloning the backbone features and pos encoding to enable compilation.
#         """
#         backbone_out = self.image_encoder(img_batch, video_tensor)
#         if self.use_high_res_features_in_sam:
#             # precompute projected level 0 and level 1 features in SAM decoder
#             # to avoid running it again on every SAM click
#             backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
#                 backbone_out["backbone_fpn"][0]
#             )
#             backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
#                 backbone_out["backbone_fpn"][1]
#             )
#         # Clone to help torch.compile
#         for i in range(len(backbone_out["backbone_fpn"])):
#             backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
#             backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][
#                 i
#             ].clone()
#         return backbone_out

#     def _forward_sam_heads(
#         self,
#         backbone_features,
#         point_inputs=None,
#         mask_inputs=None,
#         high_res_features=None,
#         multimask_output=False,
#     ):
#         """
#         Identical to the corresponding method in the parent (SAM2VideoPredictor), but
#         cloning the outputs of prompt_encoder and mask_decoder to enable compilation.
#         """
#         B = backbone_features.size(0)
#         device = backbone_features.device
#         assert backbone_features.size(1) == self.sam_prompt_embed_dim
#         assert backbone_features.size(2) == self.sam_image_embedding_size
#         assert backbone_features.size(3) == self.sam_image_embedding_size

#         # a) Handle point prompts
#         if point_inputs is not None:
#             sam_point_coords = point_inputs["point_coords"]
#             sam_point_labels = point_inputs["point_labels"]
#             assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
#         else:
#             # If no points are provide, pad with an empty point (with label -1)
#             sam_point_coords = torch.zeros(B, 1, 2, device=device)
#             sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

#         # b) Handle mask prompts
#         if mask_inputs is not None:
#             # If mask_inputs is provided, downsize it into low-res mask input if needed
#             # and feed it as a dense mask prompt into the SAM mask encoder
#             assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
#             if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
#                 sam_mask_prompt = F.interpolate(
#                     mask_inputs.float(),
#                     size=self.sam_prompt_encoder.mask_input_size,
#                     align_corners=False,
#                     mode="bilinear",
#                     antialias=True,  # use antialias for downsampling
#                 )
#             else:
#                 sam_mask_prompt = mask_inputs
#         else:
#             # Otherwise, simply feed None (and SAM's prompt encoder will add
#             # a learned `no_mask_embed` to indicate no mask input in this case).
#             sam_mask_prompt = None

#         sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
#             points=(sam_point_coords, sam_point_labels),
#             boxes=None,
#             masks=sam_mask_prompt,
#         )
#         # Clone image_pe and the outputs of sam_prompt_encoder
#         # to enable compilation
#         sparse_embeddings = sparse_embeddings.clone()
#         dense_embeddings = dense_embeddings.clone()
#         image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
#         (
#             low_res_multimasks,
#             ious,
#             sam_output_tokens,
#             object_score_logits,
#         ) = self.sam_mask_decoder(
#             image_embeddings=backbone_features,
#             image_pe=image_pe,
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=multimask_output,
#             repeat_image=False,  # the image is already batched
#             high_res_features=high_res_features,
#         )
#         # Clone the output of sam_mask_decoder
#         # to enable compilation
#         low_res_multimasks = low_res_multimasks.clone()
#         ious = ious.clone()
#         sam_output_tokens = sam_output_tokens.clone()
#         object_score_logits = object_score_logits.clone()

#         if self.pred_obj_scores:
#             is_obj_appearing = object_score_logits > 0

#             # Mask used for spatial memories is always a *hard* choice between obj and no obj,
#             # consistent with the actual mask prediction
#             low_res_multimasks = torch.where(
#                 is_obj_appearing[:, None, None],
#                 low_res_multimasks,
#                 NO_OBJ_SCORE,
#             )

#         # convert masks from possibly bfloat16 (or float16) to float32
#         # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
#         low_res_multimasks = low_res_multimasks.float()
#         high_res_multimasks = F.interpolate(
#             low_res_multimasks,
#             size=(self.image_size, self.image_size),
#             mode="bilinear",
#             align_corners=False,
#         )

#         sam_output_token = sam_output_tokens[:, 0]
#         if multimask_output:
#             # take the best mask prediction (with the highest IoU estimation)
#             best_iou_inds = torch.argmax(ious, dim=-1)
#             batch_inds = torch.arange(B, device=device)
#             low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
#             high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
#             if sam_output_tokens.size(1) > 1:
#                 sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
#         else:
#             low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

#         # Extract object pointer from the SAM output token (with occlusion handling)
#         obj_ptr = self.obj_ptr_proj(sam_output_token)
#         if self.pred_obj_scores:
#             # Allow *soft* no obj ptr, unlike for masks
#             if self.soft_no_obj_ptr:
#                 lambda_is_obj_appearing = object_score_logits.sigmoid()
#             else:
#                 lambda_is_obj_appearing = is_obj_appearing.float()

#             if self.fixed_no_obj_ptr:
#                 obj_ptr = lambda_is_obj_appearing * obj_ptr
#             obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

#         return (
#             low_res_multimasks,
#             high_res_multimasks,
#             ious,
#             low_res_masks,
#             high_res_masks,
#             obj_ptr,
#             object_score_logits,
#         )

#     def _encode_new_memory(
#         self,
#         current_vision_feats,
#         feat_sizes,
#         pred_masks_high_res,
#         object_score_logits,
#         is_mask_from_pts,
#     ):
#         """
#         Identical to the corresponding method in the parent (SAM2VideoPredictor), but
#         cloning the memories and their pos enc to enable compilation.
#         """
#         B = current_vision_feats[-1].size(1)  # batch size on this frame
#         C = self.hidden_dim
#         H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
#         # top-level feature, (HW)BC => BCHW
#         pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
#         if self.non_overlap_masks_for_mem_enc and not self.training:
#             # optionally, apply non-overlapping constraints to the masks (it's applied
#             # in the batch dimension and should only be used during eval, where all
#             # the objects come from the same video under batch size 1).
#             pred_masks_high_res = self._apply_non_overlapping_constraints(
#                 pred_masks_high_res
#             )
#         # scale the raw mask logits with a temperature before applying sigmoid
#         binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
#         if binarize and not self.training:
#             mask_for_mem = (pred_masks_high_res > 0).float()
#         else:
#             # apply sigmoid on the raw mask logits to turn them into range (0, 1)
#             mask_for_mem = torch.sigmoid(pred_masks_high_res)
#         # apply scale and bias terms to the sigmoid probabilities
#         if self.sigmoid_scale_for_mem_enc != 1.0:
#             mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
#         if self.sigmoid_bias_for_mem_enc != 0.0:
#             mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
#         maskmem_out = self.memory_encoder(
#             pix_feat,
#             mask_for_mem,
#             skip_mask_sigmoid=True,  # sigmoid already applied
#         )
#         # Clone the feats and pos_enc to enable compilation
#         maskmem_features = maskmem_out["vision_features"].clone()
#         maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
#         # add a no-object embedding to the spatial memory to indicate that the frame
#         # is predicted to be occluded (i.e. no object is appearing in the frame)
#         if self.no_obj_embed_spatial is not None:
#             is_obj_appearing = (object_score_logits > 0).float()
#             maskmem_features += (
#                 1 - is_obj_appearing[..., None, None]
#             ) * self.no_obj_embed_spatial[..., None, None].expand(
#                 *maskmem_features.shape
#             )

#         return maskmem_features, maskmem_pos_enc
