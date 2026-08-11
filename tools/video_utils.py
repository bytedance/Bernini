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

import io
import math
import os
from typing import List, Union

import decord
import torch
import torchvision
from PIL import Image


def smart_video_nframes(
    total_frames: int,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    add_one: bool = False,
) -> torch.Tensor:
    nframes = total_frames / video_fps * fps

    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * \
            frame_factor + int(add_one)
        nframes = max(nframes, frame_factor + int(add_one))
        if video_fps == fps:
            total_frames = math.floor(total_frames / frame_factor) * \
                frame_factor + int(add_one)
    else:
        nframes = int(nframes + int(add_one))

    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames + int(add_one), nframes)

    while len(idx) < int(nframes):
        idx.append(idx[-1])

    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames + int(add_one), nframes)
    if len(idx) > int(nframes):
        idx = idx[:int(nframes)]

    if frame_factor is not None:
        assert len(idx) % frame_factor == int(
            add_one), f"{len(idx)} % {frame_factor} != {int(add_one)}, total_frames: {total_frames}, video_fps: {video_fps}, fps: {fps}"

    return idx


class VideoReader:
    def __init__(self, video_bytes) -> None:
        video_buffer = io.BytesIO(video_bytes)

        self.vr = decord.VideoReader(
            video_buffer, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
        self.vr.seek(0)
        self._max_frame_id = len(self.vr) - 1
        self._fps = self.vr.get_avg_fps()

    @property
    def max_frame_id(self):
        return self._max_frame_id

    @property
    def length(self):
        return len(self.vr)

    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices) -> (List[Image.Image]):
        frames = self.vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f).convert('RGB') for f in frames]
        return frames


class PathVideoReader(VideoReader):
    def __init__(self, video_path, duration=None, crop_method=None):
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        self.vr = decord.VideoReader(
            video_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)

        self.vr.seek(0)
        self._fps = self.vr.get_avg_fps()
        self._total_frames = len(self.vr)

        if duration is None:
            # 不裁剪：整段视频
            self._start_frame = 0
            self._end_frame = self._total_frames - 1
        else:
            assert duration > 0
            crop_len = int(duration * self._fps)
            # clamp
            crop_len = min(crop_len, self._total_frames)
            assert crop_len > 0
            if crop_method == "left" or crop_method is None:
                start = 0
            elif crop_method == "right":
                start = self._total_frames - crop_len
            elif crop_method == "center":
                start = (self._total_frames - crop_len) // 2
            else:
                raise ValueError(f"Unknown crop_method: {crop_method}")
            end = start + crop_len - 1
            self._start_frame = start
            self._end_frame = end
        # 对外暴露的长度
        self._max_frame_id = self._end_frame - self._start_frame

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def max_frame_id(self) -> int:
        return self._max_frame_id

    @property
    def length(self) -> int:
        return self._max_frame_id + 1

    def sample(self, frame_indices: List[int]) -> List[Image.Image]:
        """
        frame_indices: indices relative to the cropped clip
        """
        real_indices = []
        for idx in frame_indices:
            idx = int(idx)
            idx = max(0, min(idx, self._max_frame_id))
            real_indices.append(self._start_frame + idx)
        frames = self.vr.get_batch(real_indices).asnumpy()
        return [Image.fromarray(f).convert("RGB") for f in frames]


if __name__ == '__main__':
    video_fps = 8
    idx = smart_video_nframes(
        total_frames=video_fps*8,
        video_fps=video_fps,
        fps=16,
        frame_factor=4,
        min_frames=4,
        max_frames=160,
        add_one=True,
    )
    print(len(idx))
    print(idx)