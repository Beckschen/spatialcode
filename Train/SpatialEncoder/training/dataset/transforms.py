# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
import numpy as np
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint


def hflip(datapoint, index):
    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint


class ToTensor:
    def __init__(self, v2=False):
        """
        Args:
            v2: whether to use torchvision v2 API
        """
        self.v2 = v2

    def __call__(self, video: torch.Tensor or np.ndarray) -> torch.Tensor:
        """
        Args:
            video: Tensor or ndarray of shape (T, H, W, C) or (T, H, W)
        Returns:
            Tensor of shape (T, C, H, W), dtype float32
        """

        T_frames = video.shape[0]
        out_list = []

        for t in range(T_frames):
            frame = video[t]
            if self.v2:
                # Fv2.to_image_tensor expects (H,W,C)
                frame_tensor = Fv2.to_image_tensor(frame)  # outputs (C,H,W)
            else:
                frame_tensor = F.to_tensor(
                    frame
                )  # outputs (C,H,W), float32 normalized to [0,1]
            out_list.append(frame_tensor)

        out = torch.stack(out_list, dim=0)  # (T, C, H, W)
        return out


class NormalizeTensor:
    def __init__(self, mean, std, v2=False):
        """
        Args:
            mean: list or tuple of 3 floats
            std: list or tuple of 3 floats
            v2: whether to use torchvision v2 API
        """
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: Tensor of shape (T, 3, H, W)
        Returns:
            Tensor of same shape, normalized
        """
        T_frames, C, H, W = video.shape
        assert C == 3, "Input tensor must have 3 channels"

        out = video.clone()

        if self.v2:
            # Convert dtype to float32 first
            out = Fv2.convert_image_dtype(out, torch.float32)
            out = Fv2.normalize(out, mean=self.mean, std=self.std)
        else:
            # torchvision v1 normalize expects (C,H,W), so we normalize per frame
            for t in range(T_frames):
                out[t] = F.normalize(out[t], mean=self.mean, std=self.std)

        return out


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscaleArray:
    """
    Randomly convert a video tensor to grayscale with a given probability.
    Works on NumPy arrays of shape (F, H, W, 3).

    Args:
        consistent_transform (bool): If True, either all frames are converted or none.
        p (float): Probability of applying grayscale.
    """

    def __init__(self, consistent_transform=True, p=0.5):
        self.consistent_transform = consistent_transform
        self.p = p

    def __call__(self, video_array: np.ndarray) -> np.ndarray:
        """
        Args:
            video_array (np.ndarray): Video array of shape (F, H, W, 3)

        Returns:
            np.ndarray: Transformed video array of same shape
        """
        if not isinstance(video_array, np.ndarray):
            raise TypeError("Input must be a NumPy array")

        F_num, H, W, C = video_array.shape
        if C != 3:
            video_array = np.repeat(video_array, 3, axis=-1)

        video_out = video_array.copy()

        if self.consistent_transform:
            # Decide once for all frames
            if random.random() < self.p:
                # Convert all frames to grayscale using luminance formula
                gray = (
                    0.2989 * video_out[:, :, :, 0]
                    + 0.5870 * video_out[:, :, :, 1]
                    + 0.1140 * video_out[:, :, :, 2]
                )
                # Expand back to 3 channels
                video_out = np.repeat(gray[:, :, :, np.newaxis], 3, axis=-1)
        else:
            # Frame-wise random
            for f in range(F_num):
                if random.random() < self.p:
                    gray = (
                        0.2989 * video_out[:, :, :, 0]
                        + 0.5870 * video_out[:, :, :, 1]
                        + 0.1140 * video_out[:, :, :, 2]
                    )
                    video_out[f] = np.repeat(gray[:, :, np.newaxis], 3, axis=-1)

        return video_out


class ColorJitterArray:
    """
    Apply color jitter (brightness, contrast, saturation, hue) to video NumPy arrays.
    Args:
        consistent_transform (bool): If True, all frames share the same random factors
        brightness (float): Max brightness change (0~1)
        contrast (float): Max contrast change (0~1)
        saturation (float): Max saturation change (0~1)
        hue (float): Max hue change (0~0.5, proportion of 1 cycle)
    """

    def __init__(
        self, consistent_transform=True, brightness=0, contrast=0, saturation=0, hue=0
    ):
        self.consistent_transform = consistent_transform
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        if hue == None:
            hue = 0.0
        self.hue = float(hue)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """
        Args:
            video: np.ndarray of shape (F, H, W, 3), float in [0,1] or uint8 in [0,255]
        Returns:
            np.ndarray with same shape and dtype
        """
        if not isinstance(video, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if video.ndim != 4:
            raise ValueError(f"Expected (F,H,W,C), got {video.shape}")

        F_num, H, W, C = video.shape
        if C != 3:
            # 只支持有通道维；若不是 3 通道，这里扩成伪 RGB
            video = np.repeat(video, 3, axis=-1)

        orig_dtype = video.dtype
        # 统一到 float32 做变换
        video_out = (
            (video.astype(np.float32) / 255.0)
            if np.issubdtype(orig_dtype, np.integer)
            else video.astype(np.float32, copy=True)
        )

        def apply_jitter(img, b, c, s, h):
            # img: (H,W,3) float32 in [0,1]
            # Brightness
            img = img * b
            # Contrast（围绕每通道均值）
            mean = img.mean(axis=(0, 1), keepdims=True)  # (1,1,3)
            img = (img - mean) * c + mean
            # Saturation：转灰度并插值
            gray = np.dot(
                img, np.array([0.2989, 0.5870, 0.1140], dtype=img.dtype)
            )  # (H,W)
            gray = gray[..., None]  # (H,W,1)
            img = gray + s * (img - gray)

            # Hue：HWC 下做简化 HSV 变换
            if h != 0:
                rgb = img
                maxc = rgb.max(axis=2)  # (H,W)
                minc = rgb.min(axis=2)  # (H,W)
                v = maxc
                delta = maxc - minc
                s_val = np.where(v > 0, delta / (v + 1e-6), 0.0)

                rc = (v - rgb[..., 0]) / (delta + 1e-6)
                gc = (v - rgb[..., 1]) / (delta + 1e-6)
                bc = (v - rgb[..., 2]) / (delta + 1e-6)

                h_val = np.zeros_like(v, dtype=img.dtype)
                mask = delta > 1e-6
                r_mask = mask & (rgb[..., 0] == v)
                g_mask = mask & (rgb[..., 1] == v)
                b_mask = mask & (rgb[..., 2] == v)

                h_val[r_mask] = (bc - gc)[r_mask]
                h_val[g_mask] = 2.0 + (rc - bc)[g_mask]
                h_val[b_mask] = 4.0 + (gc - rc)[b_mask]
                h_val = (h_val / 6.0 + h) % 1.0

                c_val = s_val * v
                x_val = c_val * (1 - np.abs((h_val * 6) % 2 - 1))
                m = v - c_val

                rgb_new = np.zeros_like(rgb)  # (H,W,3)
                h6 = h_val * 6

                idx = (h6 >= 0) & (h6 < 1)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [c_val[idx], x_val[idx], np.zeros_like(c_val[idx])], axis=1
                    )
                idx = (h6 >= 1) & (h6 < 2)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [x_val[idx], c_val[idx], np.zeros_like(c_val[idx])], axis=1
                    )
                idx = (h6 >= 2) & (h6 < 3)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [np.zeros_like(c_val[idx]), c_val[idx], x_val[idx]], axis=1
                    )
                idx = (h6 >= 3) & (h6 < 4)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [np.zeros_like(c_val[idx]), x_val[idx], c_val[idx]], axis=1
                    )
                idx = (h6 >= 4) & (h6 < 5)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [x_val[idx], np.zeros_like(c_val[idx]), c_val[idx]], axis=1
                    )
                idx = (h6 >= 5) & (h6 < 6)
                if np.any(idx):
                    rgb_new[idx] = np.stack(
                        [c_val[idx], np.zeros_like(c_val[idx]), x_val[idx]], axis=1
                    )

                img = rgb_new + m[..., None]  # 仍是 (H,W,3)

            return np.clip(img, 0.0, 1.0)

        # 采样 jitter 因子
        if self.consistent_transform:
            b = (
                random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                if self.brightness
                else 1.0
            )
            c = (
                random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                if self.contrast
                else 1.0
            )
            s = (
                random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                if self.saturation
                else 1.0
            )
            h = random.uniform(-self.hue, self.hue) if self.hue else 0.0
            for f in range(F_num):
                video_out[f] = apply_jitter(video_out[f], b, c, s, h)
        else:
            for f in range(F_num):
                b = (
                    random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                    if self.brightness
                    else 1.0
                )
                c = (
                    random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                    if self.contrast
                    else 1.0
                )
                s = (
                    random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                    if self.saturation
                    else 1.0
                )
                h = random.uniform(-self.hue, self.hue) if self.hue else 0.0
                video_out[f] = apply_jitter(video_out[f], b, c, s, h)

        # 恢复 dtype
        if np.issubdtype(orig_dtype, np.integer):
            video_out = np.clip(video_out * 255.0 + 0.5, 0, 255).astype(orig_dtype)
        else:
            video_out = video_out.astype(orig_dtype, copy=False)

        return video_out


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint
