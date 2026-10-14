import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Tuple


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_dim = in_dim if in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, (
            "num_heads must divide embedding_dim."
        )

        self.q_proj = nn.Linear(self.in_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.in_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        mlp_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(out_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        in_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(embedding_dim, num_heads, in_dim=in_dim, dropout=dropout)
        self.mlp = MLPBlock(mlp_dim, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm1(q + self.attn(q, k, v))
        x = self.norm2(x + self.mlp(x))
        return x


class InteractiveLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        name_3d: str = "vggt",
        backbone_channel_list: List = [128, 256, 512, 1024],
    ) -> None:
        super().__init__()
        if backbone_channel_list[0] > backbone_channel_list[1]:
            backbone_channel_list = backbone_channel_list[::-1]
        self.name_3d = name_3d
        self.attn_layers = [3]
        self.transformer_block = nn.ModuleDict()
        for k in self.attn_layers:
            self.transformer_block[str(k)] = TransformerBlock(
                embedding_dim,
                num_heads,
                embedding_dim * 4,
                in_dim=backbone_channel_list[k],
                dropout=dropout,
            )

        # self.up_convs = nn.ModuleDict(
        #     {
        #         "3-2": nn.Conv2d(
        #             backbone_channel_list[3],
        #             backbone_channel_list[2],
        #             kernel_size=1,
        #             stride=1,
        #             padding=0,
        #         ),
        #         "2-1": nn.Conv2d(
        #             backbone_channel_list[2],
        #             backbone_channel_list[1],
        #             kernel_size=1,
        #             stride=1,
        #             padding=0,
        #         ),
        #         "1-0": nn.Conv2d(
        #             backbone_channel_list[1],
        #             backbone_channel_list[0],
        #             kernel_size=1,
        #             stride=1,
        #             padding=0,
        #         ),
        #     }
        # )
        if name_3d == "vggt":
            self.channel_downsample = nn.ModuleDict()
            for attn_layer in self.attn_layers:
                self.channel_downsample[str(attn_layer)] = nn.Sequential(
                    nn.Linear(2048 * 4, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                    nn.Linear(
                        embedding_dim,
                        backbone_channel_list[attn_layer],
                    ),
                )
        elif name_3d == "mapanything":
            self.encode_dino_feat = nn.Conv2d(
                1024, 768, kernel_size=1, stride=1, padding=0
            )
            self.channel_downsample = nn.ModuleDict()
            for attn_layer in self.attn_layers:
                self.channel_downsample[str(attn_layer)] = nn.Sequential(
                    nn.Linear(768 * 4, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                    nn.Linear(
                        embedding_dim,
                        backbone_channel_list[attn_layer],
                    ),
                )
        elif name_3d == "depthanythingv3":
            self.each_level_downsample = nn.ModuleList(
                [nn.Conv2d(3072, 768, kernel_size=1, stride=1, padding=0)]* 4
            )
            self.channel_downsample = nn.ModuleDict()
            for attn_layer in self.attn_layers:
                self.channel_downsample[str(attn_layer)] = nn.Sequential(
                    nn.Linear(768 * 4, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                    nn.Linear(
                        embedding_dim,
                        backbone_channel_list[attn_layer],
                    ),
                )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, feature_2d_list, feature_3d) -> Tuple[torch.Tensor]:
        # Apply attention layer
        f_2d_0, f_2d_1, f_2d_2, f_2d_3 = feature_2d_list
        if self.name_3d == "mapanything":
            feature_3d_1 = feature_3d[0]
            feature_3d_1 = self.encode_dino_feat(feature_3d_1[0])[None]
            feature_3d = [feature_3d_1] + feature_3d[1:]
            f_3d = (
                torch.cat(feature_3d, dim=2)
                .flatten(3, 4)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
        # Concatenate 3D features along the channel dimension
        elif self.name_3d == "vggt":
            f_3d = torch.cat(feature_3d, dim=-1)
        elif self.name_3d == "depthanythingv3":
            feature_3d=[
                self.each_level_downsample[i](feature_3d[i][0]) for i in range(len(feature_3d))
            ]
            f_3d = torch.cat(feature_3d, dim=1)
            f_3d = (
                f_3d.flatten(2)
                .permute(0, 2, 1)
                .contiguous()
            )
        else:
            raise NotImplementedError
        if len(f_3d.shape) == 4:
            f_3d = f_3d.flatten(0, 1)

        attn_outs = {}
        for attn_layer in self.attn_layers:
            b, c, h, w = feature_2d_list[attn_layer].shape
            f_3d_down = (
                self.channel_downsample[str(attn_layer)](f_3d)
                .permute(0, 1, 2)
                .contiguous()
            )  # n, b, c
            f_2d = (
                feature_2d_list[attn_layer].flatten(2).permute(0, 2, 1).contiguous()
            )  # n, b, c

            f_2d = self.transformer_block[str(attn_layer)](
                q=f_2d, k=f_3d_down, v=f_3d_down
            )
            attn_outs[str(attn_layer)] = (
                f_2d.permute(1, 0, 2).reshape(b, c, h, w).contiguous()
            )
        # Concatenate attention outputs
        f_2d_3 = attn_outs["3"]
        f_2d_2 = attn_outs.get("2", f_2d_2)
        f_2d_1 = attn_outs.get("1", f_2d_1)
        f_2d_0 = attn_outs.get("0", f_2d_0)
        # f_2d_2 = attn_outs["2"] + self.up_convs["3-2"](
        #     F.interpolate(f_2d_3, scale_factor=2, mode="bilinear", align_corners=False)
        # )
        # f_2d_1 = f_2d_1 + self.up_convs["2-1"](
        #     F.interpolate(f_2d_2, scale_factor=2, mode="bilinear", align_corners=False)
        # )
        # f_2d_0 = f_2d_0 + self.up_convs["1-0"](
        #     F.interpolate(f_2d_1, scale_factor=2, mode="bilinear", align_corners=False)
        # )

        return f_2d_0, f_2d_1, f_2d_2, f_2d_3
