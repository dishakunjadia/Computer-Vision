import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    

    pretrained_model = alexnet(pretrained=True)
    self.cnn_layers = pretrained_model.features
    self.fc_layers = pretrained_model.classifier

    for param in pretrained_model.features.parameters():
        param.requires_grad = False
    for param in pretrained_model.classifier[:5].parameters():
        param.requires_grad = False

    pretrained_model.classifier[6] = nn.Linear(in_features=4096, out_features=15)

        
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
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)  
    y = self.fc_layers(x)
    #raise NotImplementedError('forward not implemented')
    return y

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
