import torch
import concurrent
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os, tqdm, cv2, ffmpeg, base64, io, glob

from IPython import display as ipythondisplay
from IPython.display import HTML

from .path import *

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 

from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms import ToTensor, ToPILImage

__all__ = ["VideoGenerator", "preview_video", "preview_frame", "save_video", "display_video", "save_frames"]


class VideoGenerator(Dataset):
    def __init__(self, video_root, video=False, start=1, fill=5, prefix="jpg", max_frames=None, gray=False):
        super(VideoGenerator).__init__()
        self.gray = gray
        self.max_frames = max_frames
        self.video_root = video_root
        self.frames = self.read_frames(video, start, fill, prefix)
        original_shape = self.frames[0].size
        self.original_shape = (original_shape[1], original_shape[0])
        
    def read_frames(self, video, start=1, fill=5, prefix="png"):
        if video:
            return self._read_frames_video()
        else:
            return self._read_frames(start, fill, prefix)
        
    def _read_frames_video(self):
        print(">> Reading video ...")
        cnt = 0; frames = []; vid_capture = cv2.VideoCapture(self.video_root)
        while(vid_capture.isOpened()):
            if self.max_frames != None:
                if cnt > self.max_frames: break
            ret, frame = vid_capture.read()
            if ret == True: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                f = Image.fromarray(frame)
                if self.gray:
                    f = f.convert("LA").convert("RGB")
                frames.append(f.convert("RGB")); cnt += 1
            else: break
        return frames
    
    def _read_frames(self, start=0, fill=5, prefix="png"):
        print(">> Reading video ...")
        max_frames = len(listdir(self.video_root))
        if self.max_frames != None:
            max_frames = np.minimum(self.max_frames, max_frames)
        def read_frame(idx):
            frame_name = "%s.%s" % (("%d" % (idx + start)).zfill(fill), prefix)
            f = Image.open("%s/%s" % (self.video_root, frame_name))
            if self.gray:
                f = f.convert("LA")
            return f
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            frames = list(executor.map(lambda x: read_frame(x).convert("RGB"), np.arange(max_frames)))
        return frames
    
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.frames[idx]
    
def preview_video(root_path, category, index, target_size=(400, 224)):
    video_root = "%s/%s/%s_%d_1280x720_25.mp4" % (root_path, category, category, index)
    vid_capture = cv2.VideoCapture(video_root)
    if vid_capture.isOpened() == True: frame_count = vid_capture.get(7)
    ret, frame = vid_capture.read(); frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = Image.fromarray(frame).convert("RGB").resize(target_size); 
    plt.imshow(f); plt.axis("off")
    return f, video_root

def preview_frame(video_root, video=False, start=1, fill=5, prefix="jpg", 
                  max_frames=-1, tgt_size=(400, 224)):
    target_size = (tgt_size[1], tgt_size[0])
    if video:
        vid_capture = cv2.VideoCapture(video_root)
        if vid_capture.isOpened() == True: 
            ret, frame = vid_capture.read(); frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB").resize(target_size)
    else:
        frame_name = "%s.%s" % (("%d" % (1)).zfill(fill), prefix)
        frame = Image.open("%s/%s" % (video_root, frame_name))
        frame = frame.convert("RGB").resize(target_size)
    plt.imshow(frame); plt.axis("off")
    return frame

def write_video(output_path, fps=24):
    output_path = Path(output_path)
    frames_path_template = str(output_path / 'frames/%5d.jpg')
    output_path = output_path / ("video_%d.mp4" % fps)
    ffmpeg.input(
        str(frames_path_template),
        format='image2',
        vcodec='mjpeg',
        framerate=fps,
    ).output(
        str(output_path), crf=17, vcodec='libx264'
    ).global_args('-loglevel', 'error').global_args('-y').run(capture_stdout=True)
    
def save_video(frames, output_path, name, start=0, fill=5, write=True):
    path = output_path + "/%s/frames" % name; mkdir(path)
    for i, f in tqdm.tqdm(enumerate(frames), total=len(frames), ascii=True, ncols=70): 
        f.save(path + "/%s.jpg" % ("%d" % (i + start)).zfill(fill))
    if write: write_video(output_path + "/%s" % name) 
    
def save_frames(frames, output_path, name, start=0, fill=5, prefix="png"):
    path = output_path + "/%s/frames" % name; mkdir(path)
    for i, f in tqdm.tqdm(enumerate(frames), total=len(frames), ascii=True, ncols=70): 
        f.save(path + "/%s.%s" % (("%d" % (i + start)).zfill(fill), prefix))

def display_video(video_path, height=224):
    video_path = glob.glob("%s/*.mp4" % video_path)[-1]
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(
        HTML(
            data='''<video alt="test" autoplay 
                loop controls style="height: {0}px;">
                <source src="data:video/mp4;base64,{1}" type="video/mp4" />
             </video>'''.format(
                str(height), encoded.decode('ascii')
            )
        )
    )
