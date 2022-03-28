import cv2
import torch
import numpy

#cv2.getGaussianKernel() -> Pytorch
def torch_getgaussiankernel(kernel_size: tuple, sigma: float, dim=2):
    """Function that returns Gaussian filter coefficients in a 1D tensor.
    Args:
        kernel_size (tuple): filter size of Gaussian kernel       
        sigma (float): gaussian standard deviation.
    Returns:
        Tensor: 2D tensor with gaussian filter coefficients arranged with 
        highest value in center to lowest at borders.

    """
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    filter = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float64)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        filter *= 1 / (std * math.sqrt(2 * math.pi)) * \
                    torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    filter = filter / torch.sum(filter)
    return filter


#cv2.copyMakeBorder() -> Pytorch
def torch_copymakeborder(tensor, pad, mode='replicate'):
    """Function to replace cv2.copyMakeBorder to Pytorch implementation using torch.nn.functional library
    Args:
        tensor  (tensor)     : 2D input tensor to be padded       
        pad     (int/tuple)  : amount to be padded at the borders
        mode    (str)        : Type of padding. 'replicate', 'constant', 
                               'reflect','circular'. Default: 'replicate'
    Returns:
        tensor_copy (tensor) : Tensor with borders padded with 'pad' amount 
                               on each side
    """
    tensor_copy = tensor.detach().clone()
    tensor_copy = tensor_copy.permute(2,0,1).unsqueeze(0)    
    #torch.nn.func.pad adds padding from the last dimension up. 
    #With 4-pad-tuple, last two dimensions (Height, Width) will be padded.
    tensor_copy = F.pad(tensor_copy, (pad,pad,pad,pad), mode, value=0)    
    #Swap back dimensions to change from NCHW -> CHW -> HWC 
    tensor_copy = tensor_copy.squeeze(0).permute(1,2,0)
    return tensor_copy


#cv2.resize() -> Pytorch
def torch_resize(tensor, Size=None, Scale=None, mode='nearest'):
    """Function to replace cv2.resize operation in pytorch with 
    "nearest neighbor" mode
    Args:
        tensor  (tensor)    : Input tensor to be resized
        
        Size    (tuple: int): Desired size of output tensor Width x Height. 
                              Default is None
        
        Scale   (int)       : Desired resizing scale. Default is None
        
        mode    (str)       : mode of resizing: 'nearest', 'linear', 
                              'bilinear', 'bicubic', 'area'. 
                              Default is 'nearest'
    Returns:
        resized (tensor)    : resized tensor with .size() == Size
    """
    #F.interpolate receives input in form of 5D: mini-batch x channels x [optional depth] x [optional height] x width
    #Need to convert template size from current HWC -> NCHW before interpolate and convert back
    tensor_copy = tensor.detach().clone()
    tensor_copy = tensor_copy.permute(2,0,1).unsqueeze(0)
    resized = F.interpolate(tensor_copy, Size, Scale, mode)
    #convert back from NCHW -> HWC
    resized = resized.squeeze(0).permute(1,2,0)
    return resized


#cv2.dilate() -> Pytorch 
#KEY-POINT: need to set origin as the center of kernel. If even sized kernel is used, calculate the origin by making syre padding amount is half of kernel  size in one dimension

def torch_dilate(image, strel, origin=(0, 0), border_value=0):
# first pad the image to have correct unfolding; origin is to align the center of kernel to 0,0 position
    image_pad = F.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, \
    origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = F.unfold(image_pad.unsqueeze(0).unsqueeze(0), \
    kernel_size=strel.shape)
    # Take maximum over the neighborhood
    result, _ = image_unfold.max(dim=1)
    # Reshape the image to recover initial shape
    dilated = torch.reshape(result, image.shape)
    return dilated
