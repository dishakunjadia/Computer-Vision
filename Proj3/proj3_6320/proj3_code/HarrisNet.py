#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple
from math import floor

from proj3_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)

"""
Authors: Patsorn Sangkloy, Vijay Upadhya, John Lambert, Cusuh Ham,
Frank Dellaert, September 2019.
"""

class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 7.1.1) in pytorch by
    sequentially stacking several layers together.

    Your task is to implement the combination of pytorch module custom layers
    to perform Harris Corner detector.

    Recall that R = det(M) - alpha(trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.

    You may find the Pytorch function nn.Conv2d() helpful here.
    """

    def __init__(self):
        """
        Create a nn.Sequential() network, using 5 specific layers (not in this
        order):
          - SecondMomentMatrixLayer: Compute S_xx, S_yy and S_xy, the output is
            a tensor of size (num_image, 3, width, height)
          - ImageGradientsLayer: Compute image gradients Ix Iy. Can be
            approximated by convolving with Sobel filter.
          - NMSLayer: Perform nonmaximum suppression, the output is a tensor of
            size (num_image, 1, width, height)
          - ChannelProductLayer: Compute I_xx, I_yy and I_xy, the output is a
            tensor of size (num_image, 3, width, height)
          - CornerResponseLayer: Compute R matrix, the output is a tensor of
            size (num_image, 1, width, height)

        To help get you started, we give you the ImageGradientsLayer layer to
        compute Ix and Iy. You will need to implement all the other layers. You
        will need to combine all the layers together using nn.Sequential, where
        the output of one layer will be the input to the next layer, and so on
        (see HarrisNet diagram). You'll also need to find the right order since
        the above layer list is not sorted ;)

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()

        image_gradients_layer = ImageGradientsLayer()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        #raise NotImplementedError('`HarrisNet.__init__` function in '
          #   + '`HarrisNet.py` needs to be implemented')
        self.net = nn.Sequential(
            image_gradients_layer,
            ChannelProductLayer(),
            SecondMomentMatrixLayer(),
            NMSLayer()
        )# <--replace this with your implementation

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network. We will only test with 1
        image at a time, and the input image will have a single channel.

        Args:
        -   x: input Tensor of shape (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network,
            (num_image, 1, height, width) tensor

        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """
    def __init__(self):
    	super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        I_xx = (x[:, 0] ** 2).unsqueeze(1)
        I_yy = (x[:, 1] ** 2).unsqueeze(1)
        I_xy = (x[:, 0] * x[:, 1]).unsqueeze(1)

        output = torch.cat([I_xx, I_yy, I_xy], dim=1)
        #raise NotImplementedError('`ChannelProductLayer` need to be '
          #  + 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output

class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """
    def __init__(self, ksize: torch.Tensor = 7, sigma: torch.Tensor = 5):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        You may find `groups` parameters in nn.Conv2d() useful.
        Understanding cnn: http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf
        understanding group parametes for nn.Conv2d(): https://discuss.pytorch.org/t/conv2d-certain-values-for-groups-and-out-channels-dont-work/14228/2

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        padding = self.ksize // 2

        self.gaussian_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=self.ksize, 
            bias=False, 
            padding=padding, 
            padding_mode='zeros'
        )

        self.gaussian_conv.weight = get_gaussian_kernel(self.ksize, self.sigma)


      
        #raise NotImplementedError('`SecondMomentMatrixLayer` need to be '
           # + 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        S_xx = self.gaussian_conv(x[:,0,:,:].unsqueeze(1))
        S_yy = self.gaussian_conv(x[:,1,:,:].unsqueeze(1))
        S_xy = self.gaussian_conv(x[:,2,:,:].unsqueeze(1))
        output = torch.cat((S_xx,S_yy,S_xy), dim=1)
        #raise NotImplementedError('`SecondMomentMatrixLayer` needs to be '
           # + 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, 1, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """
    def __init__(self, alpha: int=0.05):
        """
        Don't modify this __init__ function!
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        determinant  = torch.mul(x[:,0,:,:].unsqueeze(1),x[:,1,:,:].unsqueeze(1)) - torch.mul(x[:,2,:,:].unsqueeze(1), x[:,2,:,:].unsqueeze(1))
        trace = x[:,0,:,:].unsqueeze(1) + x[:,1,:,:].unsqueeze(1)
        trace_alpha = self.alpha * (torch.mul(trace, trace))
        output = determinant  - trace_alpha
        #raise NotImplementedError('`CornerResponseLayer` needs to be'
            #+ 'implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compared to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Truncate globally everything below the median to zero with threshold, and
        then MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum, and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, 1, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(), torch.median()
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        padding = int((7-1)/2)
        self.max_pool = nn.MaxPool2d(7, padding=(padding,padding), stride=1)
        x_0 = torch.zeros(*x.size())
        x = torch.where(x>torch.median(x), x,x_0)
        y = self.max_pool(x)
        y_1 = torch.ones(*y.size())
        y = torch.where(x==y, y_1, x_0)        
       
        output = torch.mul(y,x)
        #raise NotImplementedError('`NMSLayer` needs to be implemented')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


def get_interest_points(image: torch.Tensor, num_points: int = 4500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points with the highest confident corner
    score. Note that the return type should be a tensor. Also make sure to
    sort them in order of confidence!

    Important: Before returning the points, make sure to filter them using the 
    `remove_border_vals()` function. This will ensure that all interest points 
    can form a 16x16 window around them to avoid errors in the `get_siftnet_features()` function.


    (Potentially) useful functions: torch.nonzero, torch.masked_select,
    torch.argsort

    Args:
    -   image: A tensor of shape (b,c,m,n). We will provide one image(b=1) of
        (c = 1) for grayscale image.

    Returns:
    -   x: A tensor array of shape (N,) containing x-coordinates of
        interest points
    -   y: A tensor array of shape (N,) containing y-coordinates of
        interest points
    -   confidences (optional): tensor array of dim (N,) containing the
        strength of each interest point
    """

    # We initialize the Harris detector here, you'll need to implement the
    # HarrisNet() class
    harris_detector = HarrisNet()

    # The output of the detector is an R matrix of the same size as image,
    # indicating the corner score of each pixel. After non-maximum suppression,
    # most of R will be 0.
    R = harris_detector(image)
    # Reduce the dimensions whose dim=1
    _,_,_,n = R.shape
    R = R.squeeze()
    R = R.view(-1)

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R_index = torch.argsort(R,descending=True)
    R_index = R_index[:num_points]
    y = (R_index // image.shape[3])
    x = R_index - y*image.shape[3]
    confidences = R[R_index]
    if n>50:
        x,y,confidences = remove_border_vals(image,x,y,confidences)

    #raise NotImplementedError('`get_interest_points` in `HarrisNet.py needs ` '
        #+ 'be implemented')

    # This dummy code will compute random score for each pixel, you can
    # uncomment this and run the project notebook and see how it detects random
    # points.
    # x = torch.randint(0,image.shape[3],(num_points,))
    # y = torch.randint(0,image.shape[2],(num_points,))

    # confidences = torch.arange(num_points,0,-1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, confidences



def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function removes interest points that are too close to a border to allow 
    SIFT feature extraction. It is typically called after interest points have been 
    detected, to ensure that only valid points (those where a 16x16 window can be 
    formed) are returned. This prevents errors during feature extraction in the 
    get_siftnet_features() function.

    Args:
    -   x: Torch tensor of shape (M,)
    -   y: Torch tensor of shape (M,)
    -   c: Torch tensor of shape (M,)

    Returns:
    -   x: Torch tensor of shape (N,), where N <= M (less than or equal after pruning)
    -   y: Torch tensor of shape (N,)
    -   c: Torch tensor of shape (N,)
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

   
    border = 16
    height, width = img.shape[2], img.shape[3]

    valid_x = (x > border) & (x < width - border)
    valid_y = (y > border) & (y < height - border)
    valid_points_mask = valid_x & valid_y
    x = x[valid_points_mask]
    y = y[valid_points_mask]
    c = c[valid_points_mask]

   
    #raise NotImplementedError('`remove_border_vals` in `HarrisNet.py` needs '
#        + 'to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, c
