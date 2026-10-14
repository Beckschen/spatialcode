# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch

from PIL import Image as PILImage
from tensordict import tensorclass

from training.utils.transform_utils import angle2class, rotmat_to_quat


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedGTData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    gt_bboxes_2d: torch.FloatTensor
    gt_bboxes_center_3d: torch.FloatTensor
    gt_bboxes_dim_3d: torch.FloatTensor
    gt_bboxes_yaw_3d: torch.FloatTensor
    gt_bboxes_R_3d: torch.FloatTensor
    gt_bboxes_3d_alpha_cls: torch.FloatTensor
    gt_bboxes_3d_alpha_res: torch.FloatTensor
    gt_intrinsics: torch.FloatTensor
    # gt_spatial_intrinsics: (
    #     torch.FloatTensor
    # )  # spatial images have different resoltion and intrinsics
    gt_extrinsics: torch.FloatTensor
    gt_depth_map: torch.FloatTensor
    gt_bboxes_quat_3d: torch.FloatTensor
    gt_agg_box_xyz: Optional[torch.FloatTensor] = None
    gt_agg_box_size: Optional[torch.FloatTensor] = None
    gt_agg_box_quat: Optional[torch.FloatTensor] = None


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    spatial_img_batch: torch.FloatTensor
    depth_batch: Optional[torch.FloatTensor]
    obj_to_frame_idx: torch.IntTensor
    metadata: BatchedVideoMetaData
    gt_batch: Optional[BatchedGTData]

    dict_key: str
    video_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """
        rt = self.img_batch.transpose(0, 1).flatten(0, 1)
        return rt
        # Note that spatial feature is always BxTxCxHxW, no need to flatten


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[Dict],
    dict_key: Optional[str] = None,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch:
        dict_key (str): A string key used to identify the batch.
    """
    if batch[0] == {} or batch[0].get("valid", True) is False:  # only for testing
        return BatchedVideoDatapoint(
            img_batch=torch.empty(0),
            spatial_img_batch=torch.empty(0),
            depth_batch=None,
            obj_to_frame_idx=torch.empty(0),
            dict_key="",
            video_key="",
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=torch.empty(0),
                frame_orig_size=torch.empty(0),
            ),
            batch_size=[0],
            gt_batch=None,
        )
    batch = batch[0]  # We only support batch size of 1 for now.
    img_batch = batch["rgb_images"].unsqueeze(1).float()  # Add batch dim
    assert img_batch.shape[-1] == img_batch.shape[-2], (
        "Currently only support square images"
    )
    spatial_img_batch = batch["spatial_images"]
    if isinstance(spatial_img_batch, torch.Tensor):
        spatial_img_batch = spatial_img_batch.float()
    # Add batch dim. Note that frame is always the second dimension for vggt
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    # We only support batch size of 1 for now.
    orig_video_id = f"{batch['dataset_id']}_{batch['scene_id']}"
    orig_frame_size = batch["original_size"]
    for t, frame in enumerate(img_batch):
        objects = batch["object_ids"][t]
        for obj_i in range(len(objects)):
            orig_obj_id = batch["object_ids"][t][obj_i]
            orig_frame_idx = batch["object_appearance"][t][
                obj_i
            ]  # True if appears in this frame
            step_t_obj_to_frame_idx[t].append(torch.tensor([t, 0], dtype=torch.int))
            step_t_objects_identifier[t].extend(
                [orig_video_id, orig_obj_id, orig_frame_idx]
            )
            step_t_frame_orig_size[t].append(orig_frame_size)

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )  # [T, num_objects, 1]

    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    bboxes_3d_alpha_cls, bboxes_3d_alpha_res = angle2class(batch["bboxes_alpha_3d"])
    N, M, _, _ = batch["bboxes_R_3d"].shape
    bboxes_quat_3d = (
        rotmat_to_quat(batch["bboxes_R_3d"].flatten(0, 1).contiguous())
        .reshape(N, M, 4)
        .contiguous()
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        spatial_img_batch=spatial_img_batch,
        depth_batch=batch.get("depth_images", None),
        obj_to_frame_idx=obj_to_frame_idx,
        dict_key=dict_key,
        video_key=batch.get("frame_paths", [None] * T)[0],
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=step_t_objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        batch_size=[T],
        gt_batch=BatchedGTData(
            gt_bboxes_2d=batch["bboxes_2d"].float(),
            gt_bboxes_center_3d=batch["bboxes_center_3d"].float(),
            gt_bboxes_dim_3d=batch["bboxes_dim_3d"].float(),
            gt_bboxes_yaw_3d=batch["bboxes_yaw_3d"].float(),
            gt_bboxes_quat_3d=bboxes_quat_3d,
            gt_bboxes_R_3d=batch["bboxes_R_3d"].float(),
            gt_intrinsics=batch["intrinsics"].float(),
            # gt_spatial_intrinsics=batch["spatial_intrinsics"].float(),
            gt_extrinsics=batch["extrinsics"].float(),
            gt_depth_map=batch["depth_images"].float(),
            gt_bboxes_3d_alpha_cls=bboxes_3d_alpha_cls,
            gt_bboxes_3d_alpha_res=bboxes_3d_alpha_res,
        ),
    )
