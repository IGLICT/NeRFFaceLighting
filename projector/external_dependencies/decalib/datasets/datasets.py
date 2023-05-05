# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
from PIL import Image
from typing import Optional, List, Tuple, Union
from math import ceil, floor
from . import detectors

UNIFIED_SIZE = 512

def convert_to_mask(img: Image.Image, black_lst: Optional[List[int]] = None, white_lst: Optional[List[int]] = None) -> Image.Image:
    assert black_lst is None or white_lst is None

    img = np.array(img)
    
    if black_lst is not None:
        mask = np.ones(img.shape, dtype=np.uint8) * 255
        for black_index in black_lst:
            mask[img == black_index] = 0
        return Image.fromarray(mask)
    elif white_lst is not None:
        mask = np.zeros(img.shape, dtype=np.uint8)
        for white_index in white_lst:
            mask[img == white_index] = 255
        return Image.fromarray(mask)
    else:
        raise NotImplementedError(f"One of `black_lst` or `white_lst` must not be `None`.")

def detect_bounding_box(facial_mask: Image.Image, face_mask: Image.Image, center: Tuple[float, float]) -> Tuple[int, int, int, int]:
    upper = np.min(np.where(np.array(face_mask) == 255)[0])
    lower = np.max(np.where(np.array(facial_mask) == 255)[0])
    left = int(center[0]) - floor((lower - upper) / 2)
    right = int(center[0]) + ceil((lower - upper) / 2)
    return (left, upper, right, lower)

def get_face_transform(img_size: int, input_tensor: bool) -> transforms.Compose:
    transform_s = []
    if not input_tensor:
        transform_s.append(transforms.ToTensor())
    
    transform_s = transform_s + [
        transforms.Resize((img_size, img_size), interpolation=0)
    ]

    return transforms.Compose(transform_s)

HEAD_REGION_BLACK_LST = [0]
SKIN_REGION_WHITE_LST = [1]
HAIR_REGION_WHITE_LST = [17, 18]
FACE_REGION_BLACK_LST = [0, 14, 15, 16, 17, 18]

from torch import nn
import face_alignment
from torch.nn import functional as F
from torchvision.transforms import functional as Fv

def clever_crop(img: torch.Tensor, top: int, left: int, height: int, width: int, padding_mode: str) -> torch.Tensor:
    H, W = img.shape[-2:]

    bottom = top + height
    right = left + width

    border = max(-top, -left, bottom - H, right - W, 0)
    
    if border == 0:
        return Fv.crop(img, top, left, height, width)
    else:
        return Fv.crop(Fv.pad(img, border, padding_mode=padding_mode), top + border, left + border, height, width)

class FAN(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    def bbox2point(self, left, right, top, bottom):
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        return old_size, center
    def get_landmarks(self, image: torch.Tensor):
        with torch.no_grad():
            np_image = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            landmarks = self.model.get_landmarks(np_image)
        return landmarks
    def get_center(self, image: torch.Tensor):
            landmarks = self.get_landmarks(image)
            if landmarks is None:
                return None
            kpt = landmarks[0].squeeze()
            left = np.min(kpt[:,0])
            right = np.max(kpt[:,0])
            top = np.min(kpt[:,1])
            bottom = np.max(kpt[:,1])
            _, center = self.bbox2point(left, right, top, bottom)
            return center
    def forward(self, image: torch.Tensor, out_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
        '''
        Args:
            image: The tensor of shape (3, H, W) in range [0, 1]
            out_size: The size of output tensor
        Returns:
            The cropped images
        '''
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return None, None
        else:
            H, W = image.shape[-2:]

            kpt = landmarks[0].squeeze()
            left = np.min(kpt[:,0])
            right = np.max(kpt[:,0])
            top = np.min(kpt[:,1])
            bottom = np.max(kpt[:,1])

            old_size, center = self.bbox2point(left, right, top, bottom)
            size = int(old_size * 1.25)

            top = int(center[1] - size/2)
            left = int(center[0] - size/2)

            image = clever_crop(image, top, left, size, size, "constant")

            return F.interpolate(image.unsqueeze(0), out_size)[0], landmarks

class CelebA(Dataset):
    def __init__(
        self, 
        owd: str, 
        face_dir_path: str, 
        mask_dir_path: str, 
        parse_dir_path: str, 
        dataset_size: int = -1,  # Used to only expose part of the model
        device: torch.device = "cpu", 
        **kwargs, 
    ):
        self.face_dir_path = os.path.join(owd, face_dir_path)
        self.mask_dir_path = os.path.join(owd, mask_dir_path)
        self.parse_dir_path = os.path.join(owd, parse_dir_path)
        self.dataset_size = dataset_size
        self.detector = FAN(device)

        self.fn_s = list(filter(lambda fn: fn.split('.')[1] in ['jpg', 'png'], os.listdir(self.mask_dir_path)))
    def getitem(self, index: int) -> torch.Tensor:

        face = Image.open(os.path.join(self.face_dir_path, self.fn_s[index])).resize((UNIFIED_SIZE, UNIFIED_SIZE))
        facial_mask = Image.open(os.path.join(self.mask_dir_path, self.fn_s[index])).resize((UNIFIED_SIZE, UNIFIED_SIZE))
        parsed_face = Image.open(os.path.join(self.parse_dir_path, self.fn_s[index])).resize((UNIFIED_SIZE, UNIFIED_SIZE))
        head_mask = convert_to_mask(parsed_face, HEAD_REGION_BLACK_LST)

        center = self.detector.get_center(transforms.ToTensor()(face))
        if center is None:
            return None

        bounding_box = detect_bounding_box(facial_mask, head_mask, center)
        face = transforms.ToTensor()(face)
        face = clever_crop(face, bounding_box[1], bounding_box[0], bounding_box[3] - bounding_box[1], bounding_box[2] - bounding_box[0], "reflect")

        return face
    def __len__(self) -> int:
        if self.dataset_size >= 0:
            return min(self.dataset_size, len(self.fn_s))
        else:
            return len(self.fn_s)
    def __getitem__(self, index: int) -> torch.Tensor:
        original_image = self.getitem(index)
        image, _ = self.detector(original_image, (224, 224))
        imagename = self.fn_s[index]

        original_image = transforms.Resize((224, 224))(original_image)
        landmarks = self.detector.get_landmarks(original_image)
        return {
            'original_image': original_image, 
            'image': image, 
            'imagename': imagename, 
            'landmarks': landmarks, 
        }

def video2sequence(video_path):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', device = "cuda"):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN(device)
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
            kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }