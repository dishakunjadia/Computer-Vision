import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Dropout(p=0.2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Dropout(p=0.2)
        )

        # Fully connected layers
    self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 512),  
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 15)
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
    #print(f"Shape after CNN layers: {x.shape}")  
    x = x.view(x.size(0), -1)
    y = self.fc_layers(x)
    return y
    #raise NotImplementedError('forward not implemented')

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
