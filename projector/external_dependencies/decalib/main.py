import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from skimage.transform import estimate_transform, warp

from torchvision import transforms
from external_dependencies.face_parsing.model import BiSeNet

class DECAWrapper(object):
    def __init__(self, device: torch.device):
        print("Load DECA and FLAME ...")
        deca_cfg.model.use_tex = True
        self.deca = DECA(config = deca_cfg, device=device)
        self.scale = 1.25
        self.crop_size = 224
        self.resolution_inp = 224
        print("Load FAN ...")
        self.face_detector = detectors.FAN(device)
        print("Load Face-parsing ...")
        self.face2seg_ = BiSeNet(n_classes=19).to(device).eval().requires_grad_(False)
        self.face2seg_.load_state_dict(torch.load("./external_dependencies/face_parsing/79999_iter.pth", map_location=device))
        self.face2skin = lambda x: Image.fromarray((torch.argmax(self.face2seg_(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))((transforms.ToTensor()(x)[None, ...]*2-1).to(device)))[0], dim=1)[0] == 1).cpu().numpy().astype(np.uint8) * 255)
        self.grayscale = transforms.Grayscale()
        self.device = device
    @staticmethod
    def bbox2point(left, right, top, bottom, type='bbox'):
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center
    def deca_image_align(self, image):
        h, w, _ = image.shape
        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            raise "Error: 无法识别到人脸"
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size*self.scale)
        src_pts = np.array([
            [center[0]-size/2, center[1]-size/2], 
            [center[0] - size/2, center[1]+size/2], 
            [center[0]+size/2, center[1]-size/2]
        ])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/ 255.
        # skin = skin / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        
        # mask = warp(skin, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # mask = mask.transpose(2, 0, 1)
        return {
            'image': torch.tensor(dst_image).float(),
            # 'mask': torch.tensor(mask).float(), 
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2, 0, 1)).float(), 
            # 'original_mask': torch.tensor(skin.transpose(2, 0, 1)).float(), 
        }
    def get_landmarks(self, image: torch.Tensor):
        with torch.no_grad():
            np_image = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            landmarks = self.face_detector.model.get_landmarks(np_image)
        return landmarks
    def encode(self, img: Image.Image):
        # skin = self.face2skin(img)
        sample = self.deca_image_align(np.array(img))
        landmark = self.get_landmarks(F.interpolate(sample['image'].unsqueeze(0), (224, 224))[0])
        landmark = torch.from_numpy(landmark[0] / 224 * 2 - 1).to(self.device)[None, ...] # (1, 68, 2)
        image = sample['image'].to(self.device)[None, ...]
        # mask = sample['mask'].to(self.device)[None, ...]
        with torch.no_grad():
            codedicts = self.deca.encode(image)
        return {k: v.requires_grad_(False).detach() for k, v in codedicts.items()}