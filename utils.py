import cv2
import torch
from torch.nn import functional as F
from torchvision.transforms import Lambda, ToTensor, Resize, Compose, ToPILImage, Normalize
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import numpy as np


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
    

class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        img = img.unsqueeze(0)
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        out = soft_mask * sharp + (1 - soft_mask) * img
        return out[0]
    
def ToLab():
    def __ToLab(x):
        x = rgb2lab(np.array(x) / 255)
        x_l = (x[..., [0]] / 50) - 1
        x_ab = x[..., 1:] / 128
        return np.concatenate((x_l, x_ab), -1)
    return Lambda(lambda x: __ToLab(x))


class CustomResize(torch.nn.Module):
    def __init__(self, shape, sharp=False, radius=50, sigma=0):
        super(CustomResize, self).__init__()
        if sharp:
            self.process = Compose([Resize(shape), ToTensor(), USMSharp(radius, sigma), ToPILImage()])
        else:
            self.process = Resize(shape)
            
    def forward(self, img):
        return self.process(img)
    

class Process:
    def __init__(self, pre_shape, post_shape=None, pre_sharp=False, post_sharp=False, radius=50, sigma=0):
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        self._process = Compose([CustomResize(pre_shape, pre_sharp, radius, sigma), ToLab(), ToTensor()])
        if post_shape == None:
            self._process_org = Compose([ToLab(), ToTensor()])
        else:
            self._process_org = Compose([CustomResize(post_shape, post_sharp, radius, sigma), ToLab(), ToTensor()])
        self._process_color = Compose([CustomResize(pre_shape, pre_sharp, radius, sigma), ToTensor(), 
                                       Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    
    def pre(self, img, mode, color=False):
        if mode == "tgt":
            if color:
                img = self._process_color(img)
            else:
                img = self._process(img)
                img = torch.cat([img[[0]]] * 3, dim=0)
        elif mode == "ref":
            img = self._process_color(img) if color else self._process(img)
        elif mode == "org":
            img = self._process_org(img).unsqueeze(0)
        else:
            raise(ValueError("Invalid mode"))
        return img.float()
        
    
    def post(self, target, pred):
        target_l = target[:, [0]].cpu()
        shape = target.shape[-2:] if self.post_shape == None else self.post_shape
        if self.pre_shape != self.post_shape:  # speedup
            pred = F.interpolate(pred.cpu(), shape)
        else:
            pred = pred.cpu()
        pred = torch.cat((target_l, pred), dim=1)[0].permute((1, 2, 0))
        pred[..., 0] = ((pred[..., 0] + 1) / 2) * 100
        pred[..., 1:] = pred[..., 1:] * 128
        pred_rgb = np.clip(lab2rgb(pred.numpy()), 0, 1)
        return Image.fromarray(np.uint8(pred_rgb * 255))
    
    def post_low(target, pred):
        assert target.shape == pred.shape
        pred = torch.cat((target_l, pred), dim=1)[0].permute((1, 2, 0))
        pred[..., 0] = ((pred[..., 0] + 1) / 2) * 100
        pred[..., 1:] = pred[..., 1:] * 128
        pred_rgb = np.clip(lab2rgb(pred.numpy()), 0, 1)
        return Image.fromarray(np.uint8(pred_rgb * 255))