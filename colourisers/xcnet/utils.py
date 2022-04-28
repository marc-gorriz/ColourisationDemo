import torch
from torch.nn import functional as F
from torchvision.transforms import Lambda, ToTensor, Resize, Compose, ToPILImage, Normalize
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def ToLab():
    def __ToLab(x):
        x = rgb2lab(np.array(x) / 255)
        x_l = (x[..., [0]] / 50) - 1
        x_ab = x[..., 1:] / 128
        return np.concatenate((x_l, x_ab), -1)
    return Lambda(lambda x: __ToLab(x))


class Process:
    def __init__(self, shape):
        self._process = Compose([Resize(shape), ToLab(), ToTensor()])
        self._process_org = Compose([ToLab(), ToTensor()])
        self._process_color = Compose([Resize(shape), ToTensor(), 
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
        pred = F.interpolate(pred.cpu(), target.shape[-2:])
        pred = torch.cat((target_l, pred), dim=1)[0].permute((1, 2, 0))
        pred[..., 0] = ((pred[..., 0] + 1) / 2) * 100
        pred[..., 1:] = pred[..., 1:] * 128
        pred_rgb = np.clip(lab2rgb(pred.numpy()), 0, 1)
        return Image.fromarray(np.uint8(pred_rgb * 255))
    
def show_image(tgt, ref, pred):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 4, 1); plt.title("Target"); plt.axis("off"); plt.imshow(tgt.convert("L"), cmap="gray")
    plt.subplot(1, 4, 2); plt.title("Reference"); plt.axis("off"); plt.imshow(ref)
    plt.subplot(1, 4, 3); plt.title("Prediction"); plt.axis("off"); plt.imshow(pred)
    plt.subplot(1, 4, 4); plt.title("Original"); plt.axis("off"); plt.imshow(tgt)
    plt.show()