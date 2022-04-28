import warnings
import torch, tqdm
from PIL import Image
from torch.hub import load_state_dict_from_url

from .xcnet import XCNET
from .xcnet_color import XCNET_COLOR
from .utils import Process, show_image

__all__ = ["xcnet", "xcnet_color", "Process", "ImageColorizer"]

warnings.filterwarnings("ignore") 

def xcnet():
    url = "https://www.dropbox.com/s/ouy9ginorn2ccuw/xcnet_model.pth?dl=1"
    model = XCNET(input_shape=224, axial=True, scales=2, dim=256, output_nc=2, pos=True)
    state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def xcnet_color():
    url = "https://www.dropbox.com/s/9d16nso8vvl2yyj/xcnet_colour.pth?dl=1"
    model = XCNET_COLOR()
    state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
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
            pred = self.process.post(tgt_orig, pred)
        if show:
            show_image(tgt_img, ref_img, pred)
        return pred
    
class VideoColorizer:
    def __init__(self, device, shape=(224, 224), color=False):
        self.device = device
        self.color = color
        model = xcnet_color() if color else xcnet()
        self.model = model.to(device)
        self.process = Process(shape)
        
    def predict(self, video, ref_path):
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