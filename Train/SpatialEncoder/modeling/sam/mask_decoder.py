# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type, Dict
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from modeling.sam2_utils import LayerNorm2d, MLP


def _load_dict(source_dict: Dict[str, torch.Tensor], target_modules: nn.ModuleDict):
    for name, module in target_modules.items():
        prefix = f"sam_mask_decoder.{name}."
        sub_state = {
            k[len(prefix) :]: v for k, v in source_dict.items() if k.startswith(prefix)
        }
        missing, unexpected = module.load_state_dict(sub_state, strict=False)
        if missing or unexpected:
            print(f"[SAM load] {name}: missing={missing}, unexpected={unexpected}")


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        intrinsic_fusion_ratio: float = 0.5,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        use_spatial: bool = False,
        spatial_name: int = 2048,
        pred_aggregated_boxs: bool = False,
        sam_weight_path: Optional[str] = None,
        forward_original_sam: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.sam_weight_path = sam_weight_path
        self.forward_original_sam = forward_original_sam
        
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.intrinsic_fusion_ratio = intrinsic_fusion_ratio
        self.pred_aggregated_boxs = pred_aggregated_boxs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.depth_conf_token = nn.Embedding(1, transformer_dim)
        self.depth_tokens = nn.Embedding(1, transformer_dim)
        if self.pred_aggregated_boxs:
            self.agg_box_token = nn.Embedding(1, transformer_dim)
        # self.camera_token = nn.Embedding(1, transformer_dim)
        # self.bbox_token = nn.Embedding(1, transformer_dim)
        # self.bbox_iou_token = nn.Embedding(1, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # if spatial features are providec, use a Linear/Conv to process camera/depth features
        self.use_spatial = use_spatial
        if self.use_spatial:
            self.spatial_name = spatial_name
            if spatial_name == "vggt":
                spatial_dim = 2048
                self.camera_cond = nn.Linear(spatial_dim, transformer_dim)
                self.depth_cond = nn.Conv2d(
                    spatial_dim // 16, transformer_dim, kernel_size=1
                )
                self.track_cond = nn.Conv2d(
                    spatial_dim // 16, transformer_dim, kernel_size=1
                )
            elif spatial_name == "mapanything":
                self.camera_cond = nn.Conv2d(768, transformer_dim, kernel_size=1)
                self.depth_cond = nn.Conv2d(256, transformer_dim, kernel_size=1)
                self.scale_cond = nn.Linear(768, transformer_dim)
            elif spatial_name == "depthanythingv3":
                # self.camera_cond = nn.Conv2d(128, transformer_dim, kernel_size=1)
                self.depth_cond = nn.Conv2d(128, transformer_dim, kernel_size=1)
                self.scale_token = nn.Embedding(1, transformer_dim)
            else:
                raise NotImplementedError
            self._init_cond_transformer()
        self._init_bbox_heads()

    def _init_cond_transformer(self):
        self.control_transformer = deepcopy(self.transformer)
        if self.forward_original_sam:
            sam_transformer = deepcopy(self.transformer)
            sam_output_upscaling = deepcopy(self.output_upscaling)
            sam_output_hypernetworks_mlps = deepcopy(self.output_hypernetworks_mlps)
            sam_iou_prediction_head = deepcopy(self.iou_prediction_head)
            sam_pred_obj_score_head = deepcopy(self.pred_obj_score_head)
            self.original_sam_components = nn.ModuleDict(
                {
                    "transformer": sam_transformer,
                    "output_upscaling": sam_output_upscaling,
                    "output_hypernetworks_mlps": sam_output_hypernetworks_mlps,
                    "iou_prediction_head": sam_iou_prediction_head,
                    "pred_obj_score_head": sam_pred_obj_score_head,
                }
            )

            # load sam weights if provided
            sam_state_dict = torch.load(self.sam_weight_path, map_location="cpu")["model"]
            _load_dict(sam_state_dict, self.original_sam_components)
            # load obj score token
            self.sam_obj_score_token = nn.Embedding(1, self.transformer_dim)
            self.sam_iou_token = nn.Embedding(1, self.transformer_dim)
            self.sam_mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
            self.sam_obj_score_token.weight.data = sam_state_dict[
                "sam_mask_decoder.obj_score_token.weight"
            ].data
            self.sam_iou_token.weight.data = sam_state_dict[
                "sam_mask_decoder.iou_token.weight"
            ].data
            self.sam_mask_tokens.weight.data = sam_state_dict[
                "sam_mask_decoder.mask_tokens.weight"
            ].data

    @torch.no_grad()
    def _forward_original_sam_decoder(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.sam_obj_score_token.weight,
                    self.sam_iou_token.weight,
                    self.sam_mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.sam_iou_token.weight, self.sam_mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, (
            "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        )
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.original_sam_components["transformer"](src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.original_sam_components["output_upscaling"](src)
        else:
            dc1, ln1, act1, dc2, act2 = self.original_sam_components["output_upscaling"]
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.original_sam_components["output_hypernetworks_mlps"][i](
                    mask_tokens_out[:, i, :]
                )
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.original_sam_components["iou_prediction_head"](iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.original_sam_components["pred_obj_score_head"](
                hs[:, 0, :]
            )
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits
    
    @torch.no_grad()
    def forward_segmentation(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self._forward_original_sam_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits


    def _init_bbox_heads(self):
        self.heads = nn.ModuleDict()
        self.heads["depth_downsample"] = MLP(
            self.transformer_dim, self.iou_head_hidden_dim, self.transformer_dim // 8, 3
        )
        self.heads["depth_map"] = nn.Sequential(
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.transformer_dim // 8, 1, 3, 1, 1),
        )
        self.heads["depth_conf"] = nn.Sequential(
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.transformer_dim // 8, 1, 3, 1, 1),
        )

        self.heads["camera_downsample"] = MLP(
            self.transformer_dim,
            self.iou_head_hidden_dim,
            self.transformer_dim // 8,
            self.iou_head_depth,
        )
        self.heads["camera"] = nn.Linear(self.transformer_dim // 8, 9)
        self.heads["bbox_proposal"] = nn.Sequential(
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.transformer_dim, self.transformer_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.heads["bbox_2d_xywh"] = MLP(
            self.transformer_dim,
            self.iou_head_hidden_dim,
            4,
            self.iou_head_depth,
        )  # 2D bbox
        self.heads["bbox_3d_xyzc"] = nn.ModuleList(
            [
                MLP(
                    self.transformer_dim,
                    self.iou_head_hidden_dim,
                    4,
                    self.iou_head_depth,
                )  # 2D bbox
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.heads["bbox_3d_whl"] = nn.ModuleList(
            [
                MLP(
                    self.transformer_dim,
                    self.iou_head_hidden_dim,
                    3,
                    self.iou_head_depth,
                )  # 2D bbox
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.heads["bbox_3d_alpha"] = nn.ModuleList(
            [
                MLP(
                    self.transformer_dim,
                    self.iou_head_hidden_dim,
                    24,
                    self.iou_head_depth,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.heads["bbox_rotation_q"] = nn.ModuleList(
            [
                MLP(
                    self.transformer_dim,
                    self.iou_head_hidden_dim,
                    4,
                    self.iou_head_depth,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.heads["bbox_3d_iou"] = MLP(
            self.transformer_dim,
            self.iou_head_hidden_dim,
            self.num_mask_tokens,
            self.iou_head_depth,
            sigmoid_output=self.iou_prediction_use_sigmoid,
        )
        if self.pred_aggregated_boxs:
            self.heads["agg_box_xyz"] = MLP(
                self.transformer_dim,
                self.iou_head_hidden_dim,
                3,
                self.iou_head_depth,
            )
            self.heads["agg_box_whl"] = MLP(
                self.transformer_dim,
                self.iou_head_hidden_dim,
                3,
                self.iou_head_depth,
            )
            self.heads["agg_box_rotation"] = MLP(
                self.transformer_dim,
                self.iou_head_hidden_dim,
                4,
                self.iou_head_depth,
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        original_image_embeddings: Optional[torch.Tensor] = None,
        high_res_features: Optional[List[torch.Tensor]] = None,
        spatial_features: Optional[torch.Tensor] = None,
        original_high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        if original_image_embeddings is None:
            assert self.forward_original_sam, "original_image_embeddings is required"
            # Use original SAM decoder directly
            # segmentation setting; not detection setting
            return self.forward_segmentation(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                repeat_image=repeat_image,
                high_res_features=high_res_features,
            )
        pred_dict = self.predict_masks(
            image_embeddings=image_embeddings,
            original_image_embeddings=original_image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            spatial_features=spatial_features,
            original_high_res_features=original_high_res_features,
            training=self.training,
        )
        pred_masks = pred_dict["pred_masks"]  # num obj, 4, h, w
        pred_mask_iou = pred_dict["pred_mask_iou"]
        mask_tokens_out = pred_dict["mask_tokens_out"]  # [b, num_mask_tokens, c]
        pred_bbox_2d = pred_dict["pred_bbox_2d_list"]  # list of [b, 4]
        pred_bbox_3d = pred_dict["pred_bbox_3d_list"]  # list of [b, 7]
        pred_bbox_alpha_cls = pred_dict["pred_bbox_alpha_cls_list"]  # list of [b, 12]
        pred_bbox_alpha_res = pred_dict["pred_bbox_alpha_res_list"]  # list of [b, 12]
        pred_bbox_iou = pred_dict["pred_bbox_iou"]
        pred_agg_box_xyz = pred_dict.get("pred_agg_box_xyz", None)  # [b, num obj, 3]
        pred_agg_box_size = pred_dict.get("pred_agg_box_size", None)  # [b, num obj, 3]
        pred_agg_box_quat = pred_dict.get("pred_agg_box_quat", None)  # [b, num obj,12]
        # Select the correct mask or masks for output
        if multimask_output:
            masks = pred_masks[:, 1:, :, :]
            iou_pred = pred_mask_iou[:, 1:]
            pred_bbox_2d = pred_bbox_2d[1:]
            pred_bbox_3d = pred_bbox_3d[1:]
            pred_bbox_alpha_cls = pred_bbox_alpha_cls[1:]
            pred_bbox_alpha_res = pred_bbox_alpha_res[1:]
            pred_bbox_quat = pred_dict["pred_bbox_quat_list"][1:]
            pred_bbox_iou = pred_bbox_iou[:, 1:]
        # elif self.dynamic_multimask_via_stability and not self.training:
        #     # TODO: Select based on bbox IOU!! raise NotImplementedError
        #     masks, iou_pred = self._dynamic_multimask_via_stability(
        #         pred_masks, pred_mask_iou
        #     )
        else:
            masks = pred_masks[:, 0:1, :, :]
            iou_pred = pred_mask_iou[:, 0:1]
            pred_bbox_2d = pred_bbox_2d[0:1]
            pred_bbox_3d = pred_bbox_3d[0:1]
            pred_bbox_alpha_cls = pred_bbox_alpha_cls[0:1]
            pred_bbox_alpha_res = pred_bbox_alpha_res[0:1]
            pred_bbox_quat = pred_dict["pred_bbox_quat_list"][0:1]
            pred_bbox_iou = pred_bbox_iou[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        pred_dict.update(
            {
                "pred_masks": masks,
                "pred_mask_iou": iou_pred,
                "pred_bbox_2d_list": pred_bbox_2d,
                "pred_bbox_3d_list": pred_bbox_3d,
                "pred_bbox_alpha_cls_list": pred_bbox_alpha_cls,
                "pred_bbox_alpha_res_list": pred_bbox_alpha_res,
                "pred_bbox_quat_list": pred_bbox_quat,
                "pred_bbox_iou": pred_bbox_iou,
                "sam_output_tokens": sam_tokens_out,
                "pred_agg_box_xyz": pred_agg_box_xyz,
                "pred_agg_box_size": pred_agg_box_size,
                "pred_agg_box_quat": pred_agg_box_quat,
            }
        )
        return pred_dict

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        spatial_features: Optional[Dict[str, torch.Tensor]] = None,
        original_image_embeddings: Optional[torch.Tensor] = None,
        original_high_res_features: Optional[List[torch.Tensor]] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        # original sam decoder - use original features if provided, otherwise use current features
        if original_image_embeddings is not None and self.forward_original_sam:
            masks, iou_pred, mask_tokens_out, object_score_logits = (
                self._forward_original_sam_decoder(
                    image_embeddings=original_image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    repeat_image=repeat_image,
                    high_res_features=original_high_res_features,
                )
            )
        else:
            masks, iou_pred, mask_tokens_out, object_score_logits = 0,0,0,0

        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores and self.pred_aggregated_boxs:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.depth_tokens.weight,
                    self.depth_conf_token.weight,
                    self.agg_box_token.weight,
                    # self.camera_token.weight,
                    # self.bbox_token.weight,
                    # self.bbox_iou_token.weight,
                ],
                dim=0,
            )
            s = 1
        elif not self.pred_obj_scores and self.pred_aggregated_boxs:
            output_tokens = torch.cat(
                [
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.depth_tokens.weight,
                    self.depth_conf_token.weight,
                    self.agg_box_token.weight,
                    # self.camera_token.weight,
                    # self.bbox_token.weight,
                    # self.bbox_iou_token.weight,
                ],
                dim=0,
            )
        elif not self.pred_aggregated_boxs and self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.depth_tokens.weight,
                    self.depth_conf_token.weight,
                    # self.camera_token.weight,
                    # self.bbox_token.weight,
                    # self.bbox_iou_token.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.depth_tokens.weight,
                    self.depth_conf_token.weight,
                    # self.camera_token.weight,
                    # self.bbox_token.weight,
                    # self.bbox_iou_token.weight,
                ],
                dim=0,
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, (
            "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        )
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        scale = 1.0
        if self.use_spatial:
            spatial_depth_map = spatial_features["depth_map"]
            spatial_depth_conf = spatial_features["depth_conf"].log()
            spatial_pose_enc = spatial_features["pose_enc"]
            if self.spatial_name == "vggt":
                camera_tokens = self.camera_cond(spatial_features["camera_feat"])
                depth_tokens = self.depth_cond(spatial_features["depth_feat"])
                depth_tokens = spatial_features["depth_conf"][None] * depth_tokens
                depth_tokens = F.interpolate(
                    depth_tokens[None],
                    size=src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                track_tokens = self.track_cond(spatial_features["track_feat"])
                track_tokens = F.interpolate(
                    track_tokens[None],
                    size=src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                control_tokens = (
                    camera_tokens[..., None, None] + depth_tokens + track_tokens
                )
            elif self.spatial_name == "mapanything":
                scale = (
                    spatial_features["scale"] / 2.5
                )  # TODO: change it to a parameter
                camera_tokens = self.camera_cond(spatial_features["camera_feat"])
                camera_tokens = F.interpolate(
                    camera_tokens[None],
                    size=src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                depth_tokens = self.depth_cond(spatial_features["depth_feat"])
                depth_tokens = F.interpolate(
                    depth_tokens[None],
                    size=src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                scale_tokens = self.scale_cond(spatial_features["scale_feat"])
                control_tokens = (
                    camera_tokens + depth_tokens + scale_tokens[..., None, None]
                )
            elif self.spatial_name == "depthanythingv3":
                scale = spatial_features["scale"] / 2.5
                scale_token = self.scale_token.weight.expand(b, -1) * scale
                # camera_tokens = self.camera_cond(spatial_features["camera_feat"])
                # camera_tokens = F.interpolate(
                #     camera_tokens[None],
                #     size=src.shape[-2:],
                #     mode="bilinear",
                #     align_corners=False,
                # )
                depth_tokens = self.depth_cond(spatial_features["depth_feat"])
                depth_tokens = F.interpolate(
                    depth_tokens[None],
                    size=src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # control_tokens = (
                #     camera_tokens + depth_tokens + scale_token[..., None, None]
                # )
                control_tokens = depth_tokens + scale_token[..., None, None]
            else:
                raise NotImplementedError
            tokens, src = self.control_transformer(
                src + control_tokens, pos_src, tokens
            )
            src = src.permute(0, 2, 1).view(b, c, h, w).contiguous()
        hs, src = self.transformer(src, pos_src, tokens)

        iou_bbox_token_out = hs[:, s, :]
        mask_bbox_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]
        # detection out
        depth_tokens_ends_idx = s + self.num_mask_tokens + 2
        depth_tokens_out = hs[
            :,
            (s + 1 + self.num_mask_tokens),
            :,
        ]
        depth_conf_token = hs[:, depth_tokens_ends_idx, :]
        # camera_token_out = hs[:, depth_tokens_ends_idx + 1, :]
        # bbox_token_out = hs[:, depth_tokens_ends_idx + 2, :]
        # bbox_iou_token_out = hs[:, depth_tokens_ends_idx + 3, :]
        camera_token_out = depth_tokens_out
        bbox_token_out = depth_tokens_out
        bbox_iou_token_out = depth_conf_token
        if self.pred_aggregated_boxs:
            bbox_agg_token_out = hs[:, depth_tokens_ends_idx + 1, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_bbox_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        bbox_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        bbox_iou_pred = self.iou_prediction_head(iou_bbox_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits_bbox = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits_bbox = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # get depth map prediction & camera prediction
        depth_tokens_out = self.heads["depth_downsample"](depth_tokens_out)
        depth_map = (
            depth_tokens_out[..., None] + upscaled_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)
        depth_map = (
            self.heads["depth_map"](depth_map).mean(dim=1, keepdim=True).exp() * scale
        )

        if torch.isnan(depth_map).any():
            raise ValueError("NaN in pose_enc")
        depth_conf_token = self.heads["depth_downsample"](depth_conf_token)
        depth_conf = (
            depth_conf_token[..., None] + upscaled_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)
        depth_conf = self.heads["depth_conf"](depth_conf).mean(dim=1, keepdim=True)
        if torch.isnan(depth_conf).any():
            raise ValueError("NaN in depth_conf")

        # get camera prediction
        pose_enc_token = self.heads["camera_downsample"](camera_token_out)
        pose_enc_token = (
            pose_enc_token[..., None] + upscaled_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)
        pose_enc_token = F.adaptive_avg_pool2d(pose_enc_token, 1).flatten(1)
        pose_enc = self.heads["camera"](pose_enc_token)
        trans = torch.tanh(pose_enc[..., :3]) * 10 * scale
        rot = pose_enc[..., 3:7]
        rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)  # 保证合法四元数
        focal = F.softplus(pose_enc[..., -2:]).clamp(max=5000)
        pose_enc = torch.cat([trans, rot, focal], dim=-1)
        if torch.isnan(pose_enc).any():
            raise ValueError("NaN in pose_enc")

        # get bbox prediction
        bbox_proposal_feat = self.heads["bbox_proposal"](upscaled_embedding)
        bbox_token_sem = bbox_proposal_feat.flatten(1) + bbox_token_out
        bbox_2d = self.heads["bbox_2d_xywh"](bbox_token_sem)
        xy = bbox_2d[..., :2]
        wh = bbox_2d[..., 2:].exp()
        bbox_2d = torch.cat([xy, wh], dim=-1)

        bbox_3d_list = []
        bbox_alpha_cls = []
        bbox_alpha_res = []
        bbox_rotation_q = []
        for i in range(self.num_mask_tokens):
            bbox_3d_xyzc = self.heads["bbox_3d_xyzc"][i](bbox_token_sem)
            bbox_3d_whl = self.heads["bbox_3d_whl"][i](bbox_token_sem)
            xy = bbox_3d_xyzc[..., :2] * scale
            z = bbox_3d_xyzc[..., 2:3].exp() * scale
            whl = bbox_3d_whl.exp() * scale
            c = bbox_3d_xyzc[..., -1:].exp()
            bbox_3d = torch.cat([xy, z, whl, c], dim=-1)
            bbox_3d_list.append(bbox_3d)

            bbox_alpha = self.heads["bbox_3d_alpha"][i](bbox_token_sem)
            bbox_alpha_cls.append(bbox_alpha[..., :12])
            bbox_alpha_res.append(bbox_alpha[..., 12:])
            rotation_q = self.heads["bbox_rotation_q"][i](bbox_token_sem)
            rotation_q = rotation_q / (rotation_q.norm(dim=-1, keepdim=True) + 1e-8)
            bbox_rotation_q.append(rotation_q)
        bbox3d_iou_pred = self.heads["bbox_3d_iou"](bbox_iou_token_out)

        # get aggregated box prediction
        if self.pred_aggregated_boxs:
            bbox_agg_token_sem = (
                bbox_proposal_feat.flatten(1) + bbox_agg_token_out
            )  # [b, c]
            agg_box_xyz = self.heads["agg_box_xyz"](bbox_agg_token_sem) * scale
            agg_box_size = self.heads["agg_box_whl"](bbox_agg_token_sem).exp() * scale
            agg_box_rotation = self.heads["agg_box_rotation"](bbox_agg_token_sem)
            agg_box_rotation = agg_box_rotation / (
                agg_box_rotation.norm(dim=-1, keepdim=True) + 1e-8
            )
        else:
            agg_box_xyz = None
            agg_box_size = None
            agg_box_rotation = None

        if self.use_spatial:
            spatial_depth_map = F.interpolate(
                spatial_depth_map,
                size=depth_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.spatial_name == "mapanything":
                spatial_depth_map = spatial_depth_map * scale
            # NOTE: depthanythingv3 already scale the depth map in forward head
            elif self.spatial_name == "depthanythingv3":
                spatial_depth_map = spatial_depth_map / 2.5
            spatial_depth_conf = F.interpolate(
                spatial_depth_conf[None, None],
                size=depth_conf.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            spatial_pose_enc = spatial_features["pose_enc"]
            if self.spatial_name == "mapanything":
                spatial_T = spatial_pose_enc[:, :3] * scale
            elif self.spatial_name == "depthanythingv3":
                spatial_T = spatial_pose_enc[:, :3] / 2.5
            spatial_pose_enc = torch.cat([spatial_T, spatial_pose_enc[:, 3:]], dim=-1)
            # depth_map = (
            #     1 - self.depth_map_fusion_ratio
            # ) * depth_map + self.depth_map_fusion_ratio * spatial_depth_map
            # pose_enc = (
            #     1 - self.intrinsic_fusion_ratio
            # ) * pose_enc + self.intrinsic_fusion_ratio * spatial_pose_enc
            # depth_conf = (
            #     1 - self.intrinsic_fusion_ratio
            # ) * depth_conf + self.intrinsic_fusion_ratio * spatial_depth_conf
            fl_enc = pose_enc[:, -2:].mean(dim=0, keepdim=True)
            fl_enc = (
                1 - self.intrinsic_fusion_ratio
            ) * fl_enc + self.intrinsic_fusion_ratio * spatial_pose_enc[:, -2:]
            pose_enc = torch.cat([spatial_pose_enc[:, :-2], fl_enc], dim=-1)
            depth_map = spatial_depth_map
            depth_conf = spatial_depth_conf
        if training:
            masks= (masks + bbox_masks) / 2
            iou_pred= (iou_pred + bbox_iou_pred) / 2
            mask_tokens_out= (mask_tokens_out + mask_bbox_tokens_out) / 2
            object_score_logits= (object_score_logits_bbox + object_score_logits) / 2
        return_dict = {
            "pred_masks": bbox_masks,
            "pred_mask_iou": bbox_iou_pred,
            "mask_tokens_out": mask_bbox_tokens_out,
            "object_score_logits": object_score_logits_bbox,
            "pred_depth_map": depth_map,
            "pred_depth_conf": depth_conf,
            "pred_pose_enc": pose_enc,
            "pred_bbox_quat_list": bbox_rotation_q,
            "pred_bbox_2d_list": [bbox_2d for _ in range(self.num_mask_tokens)],
            "pred_bbox_3d_list": bbox_3d_list,
            "pred_bbox_alpha_cls_list": bbox_alpha_cls,
            "pred_bbox_alpha_res_list": bbox_alpha_res,
            "pred_bbox_iou": bbox3d_iou_pred,
            "pred_agg_box_xyz": agg_box_xyz,
            "pred_agg_box_size": agg_box_size,
            "pred_agg_box_quat": agg_box_rotation,
        }
        return return_dict

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
