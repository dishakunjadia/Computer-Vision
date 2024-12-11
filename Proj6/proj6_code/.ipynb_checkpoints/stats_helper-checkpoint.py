import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Tip: You can use any function you want to find mean and standard deviation

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  scaler = StandardScaler()
  image_path = glob.glob(os.path.join(dir_name, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(dir_name, '**', '*.jpg'), recursive=True)
  for image_p in image_path:
      image = Image.open(image_p).convert('L')
      image_ar = np.array(image)/255.0
      reshaped = image_ar.reshape(-1,1)
      scaler.partial_fit(reshaped)
  mean = np.array([scaler.mean_[0]])
  std = np.array([np.sqrt(scaler.var_[0])])


    #raise NotImplementedError('compute_mean_and_std not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
