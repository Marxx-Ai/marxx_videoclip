from typing import List
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modeling import modeling
from utils.text_encoder import text_encoder

class VideoClipXL:
    def __init__(
        self, 
        model_path="openclip_model/VideoClipXL_20102025.bin", 
        videos_paths=None
    ):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.videoclip_xl = modeling.VideoCLIP_XL()
        state_dict = torch.load(model_path, map_location="cpu")
        self.videoclip_xl.load_state_dict(state_dict)
        self.videoclip_xl = self.videoclip_xl.to(self.device).eval()

        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std

    def video_preprocessing(self, video_path, fnum=8):
        video = cv2.VideoCapture(video_path)
        frames = [x for x in self._frame_from_video(video)]
        step = max(1, len(frames) // fnum) if len(frames) > 0 else 1
        frames = frames[::step][:fnum]
        vid_tube = []
        for fr in frames:
            fr = fr[:,:,::-1]
            fr = cv2.resize(fr, (224, 224))
            fr = np.expand_dims(self.normalize(fr), axis=(0, 1))
            vid_tube.append(fr) 
        if not vid_tube:
            raise ValueError(f"No frames found in video: {video_path}")
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube

    def image_preprocessing(self, image_path):
        """
        Given an image file path, preprocess it as a single-frame video input.
        Returns a tensor with shape (1, 1, 3, 224, 224) to match video_preprocessing.
        """
        img = cv2.imread(image_path)  
        if img is None:
            raise ValueError(f"Failed to read image at: {image_path}")
        fr = img[:, :, :,]  
        fr = fr[:, :, ::-1]  
        fr = cv2.resize(fr, (224, 224))
        fr = self.normalize(fr)
        vid_tube = np.expand_dims(fr, axis=(0, 1))  
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube

    def get_video_embeds(self, videos):
        with torch.no_grad():
            video_inputs = torch.cat([self.video_preprocessing(video) for video in videos], 0).float().to(self.device)
            video_features = self.videoclip_xl.vision_model.get_vid_features(video_inputs).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            return video_features

    def get_text_embeds(self, texts):
        """
        Given a list of texts, returns a PyTorch tensor of normalized text embeddings.
        Returns:
            torch.Tensor: shape (len(texts), embedding_dim), already normalized.
        """
        text_inputs = text_encoder.tokenize(texts, truncate=True).to(self.device)
        text_features = self.videoclip_xl.text_model.encode_text(text_inputs).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features 

    def get_image_embeds(self, images):
        with torch.no_grad():
            image_inputs = torch.cat([self.image_preprocessing(image) for image in images], 0).float().to(self.device)
            image_features = self.videoclip_xl.vision_model.get_vid_features(image_inputs).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

