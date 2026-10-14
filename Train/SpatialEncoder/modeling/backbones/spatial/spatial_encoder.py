from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        spatial_name: str,
        is_pretrained: bool = True,
        is_encoder_frozen: bool = True,
        is_head_frozen: bool = True,
        img_size: int = 518,
        patch_size: int = 14,
        pretrained_path: Optional[str] = "",
        out_level_index: List = [4, 11, 17, 23],
        multilevel_channel_list: List = [],
        **kwargs,
    ):
        super(SpatialEncoder, self).__init__()
        self.spatial_name = spatial_name
        self.img_size = img_size
        self.out_level_index = out_level_index
        self.multilevel_channel_list = multilevel_channel_list
        self.patch_size = patch_size

        if spatial_name == "vggt":
            from .vggt import VGGT

            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.spatial_encoder = VGGT(
                img_size=img_size,
                patch_size=patch_size,
                enable_point=False,
                output_idx=out_level_index,
            )
            self.spatial_dim = 2048
            if is_pretrained and not pretrained_path:
                self.spatial_encoder.load_state_dict(
                    torch.hub.load_state_dict_from_url(_URL), strict=False
                )
            elif is_pretrained and pretrained_path:
                self.spatial_encoder.load_state_dict(
                    torch.load(pretrained_path, map_location="cpu"), strict=False
                )
            else:
                raise ValueError("spatial encoder must be pretrained")

            if is_encoder_frozen:
                for p in self.spatial_encoder.aggregator.parameters():
                    p.requires_grad = False
            if is_head_frozen:
                for p in self.spatial_encoder.camera_head.parameters():
                    p.requires_grad = False
                for p in self.spatial_encoder.depth_head.parameters():
                    p.requires_grad = False
                for p in self.spatial_encoder.track_head.parameters():
                    p.requires_grad = False

        elif spatial_name == "mapanything":
            from .mapanything import MapAnything
            
            # Monkey patch to ignore unsupported parameters
            import uniception.models.encoders.dinov2 as dinov2_module
            original_init = dinov2_module.DINOv2Encoder.__init__
            
            def patched_init(self, *args, **kwargs):
                # Remove unsupported parameters and enable force_reload
                kwargs.pop('norm_returned_features', None)
                kwargs['torch_hub_force_reload'] = True
                return original_init(self, *args, **kwargs)
            
            dinov2_module.DINOv2Encoder.__init__ = patched_init
            
            try:
                self.spatial_encoder = MapAnything.from_pretrained(
                    "facebook/map-anything",
                    torch_hub_force_reload=True
                )
            finally:
                # Restore original init
                dinov2_module.DINOv2Encoder.__init__ = original_init
            
            self.spatial_dim = 768
            self.is_encoder_frozen = is_encoder_frozen
            self.is_head_frozen = is_head_frozen

        elif spatial_name == "depthanythingv3":
            from .depthanythingv3 import DepthAnything3

            self.spatial_encoder = DepthAnything3.from_pretrained(
                "depth-anything/DA3NESTED-GIANT-LARGE"
            )
            self.spatial_dim = 1024
            self.is_encoder_frozen = is_encoder_frozen
            self.is_head_frozen = is_head_frozen
            if is_encoder_frozen:
                for p in self.spatial_encoder.model.da3.backbone.parameters():
                    p.requires_grad = False
            if is_head_frozen:
                assert is_encoder_frozen, (
                    "If head is frozen, encoder must be frozen too"
                )
                for p in self.spatial_encoder.model.parameters():
                    p.requires_grad = False

        else:
            raise NotImplementedError

    # @torch.no_grad()
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_name == "vggt":
            all_feat_list, selected_out_list, feat_start_idx = (
                self.spatial_encoder.aggregator(x)
            )
            return_dict = {
                "out_level_index": self.out_level_index,
                "aggregated_tokens_list": all_feat_list,
                "selected_out_list": selected_out_list,
                "token_start_idx": feat_start_idx,
            }
            return return_dict
        elif self.spatial_name == "mapanything":
            if self.is_encoder_frozen:
                with torch.no_grad():
                    all_feat_list, scale_feat = self.spatial_encoder.forward_encoder(x)
            else:
                all_feat_list, scale_feat = self.spatial_encoder.forward_encoder(x)
            return_dict = {
                "selected_level_features": all_feat_list,
                "scale_feat": scale_feat,
            }
            return return_dict
        elif self.spatial_name == "depthanythingv3":
            if self.is_encoder_frozen:
                with torch.no_grad():
                    feat_list, _ = self.spatial_encoder.model.da3.backbone(x)
            else:
                feat_list, _ = self.spatial_encoder.model.da3.backbone(x)

            vis_feat_list = []
            camera_feat_list = []
            selected_level_features = []
            for feat_tuple in feat_list:
                vis_feat_list.append(feat_tuple[0])
                camera_feat_list.append(feat_tuple[1])
                vis_feat = feat_tuple[0]
                B, F, N, C = vis_feat.shape
                H = int(math.sqrt(N))
                assert H * H == N, "Feature map is not square!"
                vis_feat = vis_feat.permute(0, 1, 3, 2).reshape(B, F, C, H, H)
                selected_level_features.append(vis_feat)

            return_dict = {
                "selected_level_features": selected_level_features,
                "vis_feat_list": vis_feat_list,
                "camera_feat_list": camera_feat_list,
            }
            return return_dict
        else:
            raise NotImplementedError

    def forward_heads(self, **kwargs) -> torch.Tensor:
        if self.spatial_name == "mapanything":
            assert kwargs.get("dense_head_inputs", None) is not None
            assert kwargs.get("scale_head_inputs", None) is not None
            dense_head_inputs, scale_head_inputs = (
                kwargs["dense_head_inputs"],
                kwargs["scale_head_inputs"],
            )
            if self.is_head_frozen:
                with torch.no_grad():
                    (
                        dense_final_outputs,
                        pose_final_outputs,
                        scale_final_output,
                        (dense_feat, camera_feat, scale_feat),
                    ) = self.spatial_encoder.forward_heads(
                        dense_head_inputs,
                        scale_head_inputs,
                    )
            else:
                (
                    dense_final_outputs,
                    pose_final_outputs,
                    scale_final_output,
                    (dense_feat, camera_feat, scale_feat),
                ) = self.spatial_encoder.forward_heads(
                    dense_head_inputs,
                    scale_head_inputs,
                )
            (
                output_depth,
                output_confidences_per_view,
                output_cam,
                scale_final_output,
            ) = self.spatial_encoder.get_final_output(
                dense_final_outputs,
                pose_final_outputs,
                scale_final_output,
            )
            return_dict = {
                "depth_map": output_depth,
                "depth_conf": output_confidences_per_view,
                "pose_enc": output_cam,
                "scale": scale_final_output,
                "depth_feat": dense_feat[None],
                "camera_feat": camera_feat[None],
                "scale_feat": scale_feat,
            }
            return return_dict

        elif self.spatial_name == "vggt":
            aggregated_tokens_list = kwargs.get("aggregated_tokens_list", None)
            video = kwargs.get("video", None)
            assert aggregated_tokens_list is not None
            assert video is not None
            if self.is_head_frozen:
                with torch.no_grad():
                    depth_map, depth_conf, depth_feat = (
                        self.spatial_encoder.forward_depth_head(
                            aggregated_tokens_list, video, start_index=5
                        )
                    )
                    pose_enc = self.spatial_encoder.forward_camera_head(
                        aggregated_tokens_list
                    )
            else:
                depth_map, depth_conf, depth_feat = (
                    self.spatial_encoder.forward_depth_head(
                        aggregated_tokens_list, video, start_index=5
                    )
                )
                pose_enc = self.spatial_encoder.forward_camera_head(
                    aggregated_tokens_list
                )
            return_dict = {
                "depth_map": depth_map,
                "depth_conf": depth_conf,
                "depth_feat": depth_feat,
                "camera_feat": aggregated_tokens_list[-1][:, :, 0:1],
                "pose_enc": pose_enc,
            }
            return return_dict
        elif self.spatial_name == "depthanythingv3":
            vis_feat_list = kwargs.get("vis_feat_list", None)
            camera_feat_list = kwargs.get("camera_feat_list", None)
            assert vis_feat_list is not None
            assert camera_feat_list is not None
            if self.is_head_frozen:
                with torch.no_grad():
                    head_outputs = self.spatial_encoder.forward_heads(**kwargs)
            else:
                head_outputs = self.spatial_encoder.forward_heads(**kwargs)
            return head_outputs
        else:
            raise NotImplementedError

    # implemented for vggt only
    def forward_camera_head(self, features: torch.Tensor) -> torch.Tensor:
        if self.spatial_name == "vggt":
            pose_enc = self.spatial_encoder.camera_head(features)[-1]
        else:
            raise NotImplementedError
        return pose_enc

    # implemented for vggt only
    def forward_depth_head(
        self, feature_list: List[torch.Tensor], images: torch.Tensor, start_index: int
    ) -> torch.Tensor:
        if self.spatial_name == "vggt":
            depth_map, depth_conf, depth_feat = self.spatial_encoder.depth_head(
                feature_list, images, start_index
            )
        else:
            raise NotImplementedError
        return depth_map, depth_conf, depth_feat
