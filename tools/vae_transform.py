# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def make_divisible(value, stride):
    return max(stride, int(round(value / stride) * stride))


def apply_scale(width, height, scale, stride):
    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = make_divisible(new_width, stride)
    new_height = make_divisible(new_height, stride)
    return new_width, new_height


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = apply_scale(width, height, scale, self.stride)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = apply_scale(
                new_width, new_height, scale, self.stride)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = apply_scale(
                new_width, new_height, scale, self.stride)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class VAEImageTransform:
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=2048 * 2048,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1, resized_w_h=None):
        img = img.convert('RGB')
        if resized_w_h is not None:
            img = img.resize(resized_w_h)
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = self.normalize_transform(img)
        return img