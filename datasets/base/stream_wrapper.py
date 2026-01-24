from typing import List, Tuple, Optional

import torch

from .pixel_source import ScenePixelSource


class StreamWrapper(torch.utils.data.Dataset):
    """
    Sequential (streaming) wrapper over ScenePixelSource.

    Interface mirrors SplitWrapper but returns frames in order via next().
    """

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: Optional[List[int]] = None,
        split: str = "train",
    ):
        super().__init__()
        self.datasource = datasource
        if split_indices is None:
            split_indices = list(range(self.datasource.num_imgs))
        self.split_indices = split_indices
        self.split = split
        self._cursor = 0
        self._loop = False
        self._stride = 1

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def set_stride(self, stride: int) -> None:
        self._stride = max(int(stride), 1)

    def reset(self, cursor: int = 0) -> None:
        self._cursor = max(int(cursor), 0)

    def has_next(self) -> bool:
        return self._cursor < len(self.split_indices)

    def get_image(self, idx, camera_downscale) -> dict:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])
        self.datasource.reset_downscale_factor()
        return image_infos, cam_infos

    def next(self, camera_downscale) -> Tuple[dict, dict]:
        if len(self.split_indices) == 0:
            raise StopIteration("StreamWrapper has no indices to iterate.")

        if self._cursor >= len(self.split_indices):
            if not self._loop:
                raise StopIteration("StreamWrapper reached the end of the stream.")
            self._cursor = 0

        idx = self.split_indices[self._cursor]
        self._cursor += self._stride

        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(idx)
        self.datasource.reset_downscale_factor()

        return image_infos, cam_infos

    def __getitem__(self, idx) -> dict:
        return self.get_image(idx, camera_downscale=1.0)

    def __len__(self) -> int:
        return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
