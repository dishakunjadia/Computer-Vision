"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:
  """
  Calculate the disparity value at each pixel by searching a small 
  patch around a pixel from the left image in the right image

  Note: 
  1.  It is important for this project to follow the convention of search
      input in left image and search target in right image
  2.  While searching for disparity value for a patch, it may happen that there
      are multiple disparity values with the minimum value of the similarity
      measure. In that case we need to pick the smallest disparity value.
      Please check the numpy's argmin and pytorch's argmin carefully.
      Example:
      -- diparity_val -- | -- similarity error --
      -- 0               | 5 
      -- 1               | 4
      -- 2               | 7
      -- 3               | 4
      -- 4               | 12

      In this case we need the output to be 1 and not 3.
  3. The max_search_bound is defined from the patch center.

  Args:
  -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be >= 1.
  -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
  -   block_size: the size of the block to be used for searching between
                  left and right image
  -   sim_measure_function: a function to measure similarity measure between
                            two tensors of the same shape; returns the error value
  -   max_search_bound: the maximum horizontal distance (in terms of pixels) 
                        to use for searching
  Returns:
  -   disparity_map: The map of disparity values at each pixel. 
                     Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
  """

  assert left_img.shape == right_img.shape
  disparity_map = torch.zeros(1) #placeholder, this is not the actual size
  ############################################################################
  # Student code begin
  ############################################################################
  height, width, channels = left_img.shape
  half_patch = block_size // 2
  r_range = range(half_patch, height - half_patch)

  c_range = range(half_patch, width - half_patch)
  disparity_map = torch.zeros((height - 2 * half_patch, width - 2 * half_patch))

 
  for r in r_range:
     for c in c_range:
        left_patch = left_img[
            r - half_patch:r + half_patch + 1,
            c - half_patch:c + half_patch + 1,:  ]

        lowest_error = float('inf')
        optimal_disparity = 0

        for disparity in range(max_search_bound + 1):
            c_offset = c - disparity
            if c_offset - half_patch < 0:
                continue 
            right_patch = right_img[
                r - half_patch:r + half_patch + 1,
                c_offset - half_patch:c_offset + half_patch + 1,:]
            error = sim_measure_function(left_patch, right_patch)

            if error < lowest_error:
                lowest_error = error
                optimal_disparity = disparity

        disparity_map[r - half_patch, c - half_patch] = optimal_disparity

  #raise NotImplementedError('calculate_disparity_map not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return disparity_map

def calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          sim_measure_function: Callable,
                          block_size: int = 9):
  """
  Calculate the cost volume. Each pixel will have D=max_disparity cost values
  associated with it. Basically for each pixel, we compute the cost of
  different disparities and put them all into a tensor.

  Note: 
  1.  It is important for this project to follow the convention of search
      input in left image and search target in right image
  2.  If the shifted patch in the right image will go out of bounds, it is
      good to set the default cost for that pixel and disparity to be something
      high(we recommend 255), so that when we consider costs, valid disparities will have a lower
      cost. 

  Args:
  -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be 1 or 3.
  -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
  -   max_disparity:  represents the number of disparity values we will consider.
                  0 to max_disparity-1
  -   sim_measure_function: a function to measure similarity measure between
                  two tensors of the same shape; returns the error value
  -   block_size: the size of the block to be used for searching between
                  left and right image
  Returns:
  -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                dimensions, and D is max_disparity. cost_volume[x,y,d] 
                represents the similarity or cost between a patch around left[x,y]  
                and a patch shifted by disparity d in the right image. 
  """
  #placeholder
  height = left_img.shape[0]
  width = right_img.shape[1]
  cost_volume = torch.zeros(height, width, max_disparity)
  ############################################################################
  # Student code begin
  ############################################################################


  #raise NotImplementedError('calculate_cost_volume not implemented')
  cost_volume = torch.full((height, width, max_disparity), 255.0)

  half_patch_size = block_size // 2
  for r in range(half_patch_size, height - half_patch_size):
    for c in range(half_patch_size, width - half_patch_size):
        
        left_patch = left_img[
           
                       c - half_patch_size:c + half_patch_size + 1,

            r - half_patch_size:r + half_patch_size + 1,
        ]
        
        for disparity in range(max_disparity):
            c_offset = c - disparity
            
            if c_offset - half_patch_size < 0:
                continue
            
            right_patch = right_img[
                r - half_patch_size:r + half_patch_size + 1,
                c_offset - half_patch_size:c_offset + half_patch_size + 1,
            ]
            
            similarity_error = sim_measure_function(
                torch.tensor(left_patch), torch.tensor(right_patch)
            )
            cost_volume[r, c, disparity] = similarity_error
 
  ############################################################################
  # Student code end
  ############################################################################
  return cost_volume
