# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Functions for resampling images."""

import numpy as np
import torch

def resample(data, warp):
    """Resamples input data at user defined coordinates.

    Args:
      data: Tensor of shape `[batch_size, data_height, data_width,
        data_num_channels]` containing 2D data that will be resampled.
      warp: Tensor shape `[batch_size, dim_0, ... , dim_n, 2]` containing the
        coordinates at which resampling will be performed.
      name: Optional name of the op.

    Returns:
      Tensor of resampled values from `data`. The output tensor shape is
      `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
    """
    assert len(data.shape) == 4, "data to warp must be N, C, H, W"
    assert warp.shape[1:3] == data.shape[-2:], "data to warp must be N, C, H, W"
    
    warp_x, warp_y = torch.unbind(warp, axis=-1)
    data = data.permute(0, 2, 3, 1)
    return resampler_with_unstacked_warp(data, warp_x, warp_y).permute(0, 3, 1, 2)

def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y):
    """Resamples input data at user defined coordinates.

    The resampler functions in the same way as `resampler` above, with the
    following differences:
    1. The warp coordinates for x and y are given as separate tensors.
    2. If warp_x and warp_y are known to be within their allowed bounds, (that is,
      0 <= warp_x <= width_of_data - 1, 0 <= warp_y <= height_of_data - 1) we
      can disable the `safe` flag.

    Args:
      data: Tensor of shape `[batch_size, data_height, data_width,
        data_num_channels]` containing 2D data that will be resampled.
      warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
        coordinates at which resampling will be performed.
      warp_y: Tensor of the same shape as warp_x containing the y coordinates at
        which resampling will be performed.
      safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
        Disable only if you know they are within bounds, otherwise a runtime
        exception will be thrown.
      name: Optional name of the op.

    Returns:
      Tensor of resampled values from `data`. The output tensor shape is
      `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

    Raises:
      ValueError: If warp_x, warp_y and data have incompatible shapes.
    """
    device = data.device
    N, H, W, C = data.shape
    warp_shape = warp_x.shape # =[N, H, W]
    
    # compute floor and ceiling sfor bilinear warp weights
    warp_floor_x = torch.floor(warp_x) # N, H, W
    warp_floor_y = torch.floor(warp_y) # N, H, W
    
    # compute weights in 1D
    right_warp_weight = (warp_x - warp_floor_x)[..., None] # N, H, W, 1
    down_warp_weight = (warp_y - warp_floor_y)[..., None] # N, H, W, 1
    left_warp_weight = (1. - right_warp_weight) # N, H, W, 1
    up_warp_weight = (1. - down_warp_weight) # N, H, W, 1
    
    # cast to int and ceiling values to get exact pixel indices
    warp_floor_x = warp_floor_x.type(torch.int32) # N, H, W
    warp_floor_y = warp_floor_y.type(torch.int32) # N, H, W
    warp_ceil_x = torch.ceil(warp_x).type(torch.int32) # N, H, W
    warp_ceil_y = torch.ceil(warp_y).type(torch.int32) # N, H, W
    
    # get batches
    warp_batch = torch.arange(warp_shape[0], dtype=torch.int32, device=device).reshape(warp_shape[0], *[1 for _ in range(len(warp_shape[1:]))]) # arange(N).reshape([N, 1, 1])
    
    floor_y_mask = ((warp_floor_y >= 0) & (warp_floor_y < H))[..., None]
    ceil_y_mask = ((warp_ceil_y >= 0) & (warp_ceil_y < H))[..., None]
    floor_x_mask = ((warp_floor_x >= 0) & (warp_floor_x < W))[..., None]
    ceil_x_mask = ((warp_ceil_x >= 0) & (warp_ceil_x < W))[..., None]
    
    warp_floor_y_safe = torch.clamp(warp_floor_y, 0, H - 1) # N, H, W
    warp_ceil_y_safe = torch.clamp(warp_ceil_y, 0, H - 1) # N, H, W
    warp_floor_x_safe = torch.clamp(warp_floor_x, 0, W - 1) # N, H, W
    warp_ceil_x_safe = torch.clamp(warp_ceil_x, 0, W - 1) # N, H, W

    result = ((data[warp_batch, warp_floor_y_safe, warp_floor_x_safe] * floor_x_mask * left_warp_weight +
               data[warp_batch, warp_floor_y_safe, warp_ceil_x_safe] * ceil_x_mask * right_warp_weight) * floor_y_mask * up_warp_weight +
              (data[warp_batch, warp_ceil_y_safe, warp_floor_x_safe] * floor_x_mask * left_warp_weight + 
               data[warp_batch, warp_ceil_y_safe, warp_ceil_x_safe] * ceil_x_mask * right_warp_weight) * ceil_y_mask * down_warp_weight)
    return result
  
resample_with_unstacked_warp = torch.jit.trace(resampler_with_unstacked_warp,
                                               (torch.rand(4, 640, 640, 3).cuda(),
                                                torch.rand(4, 640, 640).cuda(),
                                                torch.rand(4, 640, 640).cuda()))
  

if __name__ == "__main__":
    data = torch.randn(4, 3, 32, 32).cuda()
    warp = torch.rand(4, 32, 32, 2).cuda() * 32
    resample(data, warp)
    
    