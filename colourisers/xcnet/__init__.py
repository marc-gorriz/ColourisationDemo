import warnings

import torch
import tqdm
from PIL import Image
from torch.hub import load_state_dict_from_url

from .utils import Process
from .xcnet import XCNET
from .xcnet_color import XCNET_COLOR
from .xcnet_full import XCNET as XCNET_FULL

__all__ = ["xcnet", "xcnet_color", "Process", "ImageColorizer", "VideoColorizer", "get_model"]

warnings.filterwarnings("ignore") 

def xcnet():
    url = "https://www.dropbox.com/s/ouy9ginorn2ccuw/xcnet_model.pth?dl=1"
    model = XCNET(input_shape=224, axial=True, scales=2, dim=256, output_nc=2, pos=True)
    state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def xcnet_color():
    path = "/work/marcb/video-xcnet/experiments/imagenet_rgb/checkpoints/epoch:26-step:49915.pth"
    model = XCNET_COLOR()
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def xcnet_full():
    path = "/work/marcb/efficient-analogies/xcnet/experiments/xcnet_norm/checkpoints/epoch:41-step:99970.pth"
    model = XCNET_FULL()
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    return model

class ImageColorizer:
    def __init__(self, device, shape=(224, 224), color=False):
        self.device = device
        self.color = color
        model = xcnet_color() if color else xcnet()
        self.model = model.to(device)
        self.process = Process(shape)
        
    def predict(self, tgt_path, ref_path, show=False):
        tgt_img = Image.open(tgt_path)
        ref_img = Image.open(ref_path)
        tgt_orig = self.process.pre(tgt_img, "org")
        tgt = self.process.pre(tgt_img, "tgt", color=self.color)
        ref = self.process.pre(ref_img, "ref", color=self.color)
        tgt = tgt.unsqueeze(0).to(self.device)
        ref = ref.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tgt, ref)[0]
            pred_img = self.process.post(tgt_orig, pred)
        if show:
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 4, 1); plt.title("Target"); plt.axis("off"); plt.imshow(tgt_img.convert("L"), cmap="gray")
            plt.subplot(1, 4, 2); plt.title("Reference"); plt.axis("off"); plt.imshow(ref_img)
            plt.subplot(1, 4, 3); plt.title("Prediction"); plt.axis("off"); plt.imshow(pred_img)
            plt.subplot(1, 4, 4); plt.title("Original"); plt.axis("off"); plt.imshow(tgt_img)
        return pred_img
    
class VideoColorizer: 
    def __init__(self, device, pre_shape=(224, 224), post_shape=None, 
                 color=False, pre_sharp=False, post_sharp=False):
        self.device = device
        self.color = color
        model = xcnet_color() if color else xcnet()
        self.model = model.to(device)
        self.process = Process(pre_shape, post_shape, pre_sharp=pre_sharp, post_sharp=post_sharp)
        
    def predict(self, video, ref_path, post_shape=None):
        print(">> Colorizing video ...")
        ref = Image.open(ref_path)
        ref = self.process.pre(ref, "ref", color=self.color)
        ref = ref.unsqueeze(0).to(self.device)

        out = []
        for tgt in tqdm.tqdm(video.frames, total=len(video),
                             ascii=True, ncols=70):
            tgt_orig = self.process.pre(tgt, "org")
            tgt = self.process.pre(tgt, "tgt", color=self.color)
            tgt = tgt.unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                pred = self.model(tgt, ref)[0] 
            out.append(self.process.post(tgt_orig, pred))    
        return out