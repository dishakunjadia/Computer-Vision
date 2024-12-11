'''
Contains functions with different data transforms
'''
import os
import glob
import torch
import torch.utils.data as data
import torchvision

from typing import Tuple, List
from PIL import Image
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt 
import numpy as np
import torchvision.transforms as transforms

from typing import Tuple


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean  of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''

  fundamental_transforms = None

  #############################################################################
  # Student code begin
  #############################################################################

  pixel_mean = np.array(pixel_mean)
  pixel_std = np.array(pixel_std)

    
  fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),  
        transforms.ToTensor(),       
        transforms.Normalize(mean=pixel_mean.tolist(), std=pixel_std.tolist())  
    ])
    #raise NotImplementedError('get_fundamental_transforms not implemented')

  #############################################################################
  # Student code end
  #############################################################################
  return fundamental_transforms


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the data augmentation + core transforms needed to be applied on the
  train set

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean  of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.Compose with all the transforms
  '''

  aug_transforms = None

  #############################################################################
  # Student code begin
  #############################################################################

  aug_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.Resize(inp_size),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=pixel_mean.tolist(), std=pixel_std.tolist())  
    ])
  #raise NotImplementedError('get_data_augmentation_transforms not implemented')

  #############################################################################
  # Student code end
  #############################################################################
  return aug_transforms

pixel_mean = np.array([0.485, 0.456, 0.406])  
pixel_std = np.array([0.229, 0.224, 0.225])   

unnorm = transforms.Normalize(
    mean=(-pixel_mean / pixel_std).tolist(),
    std=(1 / pixel_std).tolist()
)

def tensor_to_image(tensor):
    tensor = unnorm(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    return (img * 255).astype(np.uint8)

inp_size = (224, 224)

img_path = "/Users/u1464153/Documents/CV/proj6_6320/data/train/Bedroom/image_0040.jpg"  
original_img = Image.open(img_path).convert("RGB")
transforms_func = get_data_augmentation_transforms(inp_size, pixel_mean, pixel_std)

plt.figure(figsize=(15, 6))
augmented_imgs = [transforms_func(original_img) for _ in range(5)]
plt.subplot(1, 6, 1)
plt.title("Original")
plt.imshow(original_img)
plt.axis("off")

for i, img in enumerate(augmented_imgs, 2):
    augmented_img = tensor_to_image(img)
    plt.title(f"Augmented {i-1}") 
    plt.imshow(augmented_img)
    plt.subplot(1, 6, i)
    plt.axis("off")
plt.show()



