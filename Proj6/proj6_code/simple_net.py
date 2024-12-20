import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note:
    - Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    - Keep the name of cnn_layers and fc_layers of the network same as given.
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input: (1, H, W) -> Output: (16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> Output: (16, H/2, W/2)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Input: (16, H/2, W/2) -> Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample -> Output: (32, H/4, W/4)
        )

    self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),  # Adjust the input size based on the image size after pooling
            nn.ReLU(),
            nn.Linear(128, 15)  # Output: 15 classes
        )

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    #raise NotImplementedError('__init__ not implemented')

    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    x = self.cnn_layers(x)

    x = x.view(x.size(0), -1)  

    y = self.fc_layers(x)
    return y
      #raise NotImplementedError('forward not implemented')

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output

  # Alternative implementation using functional form of everything
class SimpleNet2(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 10, 5)
      self.pool = nn.MaxPool2d(3, 3)
      self.conv2 = nn.Conv2d(10, 20, 5)
      self.fc1 = nn.Linear(500, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 15)
      self.loss_criterion = nn.CrossEntropyLoss()

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 500)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
