from .utils.utils_SH import *
from .model.defineHourglass_512_gray_skip import *
import os
import numpy as np
import torch
from kornia.color import rgb_to_lab, lab_to_rgb

from torch_utils import misc
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

class DPR(object):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1])
        
        # Define 'normal'
        img_size = 256
        x = np.linspace(-1, 1, img_size)
        z = np.linspace(1, -1, img_size)
        x, z = np.meshgrid(x, z)

        mag = np.sqrt(x**2 + z**2)
        valid = mag <=1
        y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
        x = x * valid
        y = y * valid
        z = z * valid
        normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
        normal = np.reshape(normal, (-1, 3))
        self.normal = normal
        
        network = HourglassNet().eval().to(device)
        network.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "trained_model", 'trained_model_03.t7')))
        self.network = network.requires_grad_(False)
        
        sh = np.loadtxt(os.path.join(os.path.dirname(__file__), "./data/example_light/", 'rotate_light_00.txt'))
        sh = sh[0:9]
        sh = sh * 0.7

        #--------------------------------------------------
        # rendering half-sphere
        sh = np.squeeze(sh)
        
        #  rendering images using the network
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        self.sh = torch.from_numpy(sh).to(device).requires_grad_(False)
    def extract_lighting(self, img: torch.Tensor):
        img = filtered_resizing(img, size=512, f=self.resample_filter)
        img = rgb_to_lab(img.clamp(-1, 1) / 2. + .5)
        L = img[:, :1, :, :] / 100.
        _, SH  = self.network(L, self.sh.expand(img.size(0), -1, -1, -1), 0)
        return SH
    def exert_lighting(self, img: torch.Tensor, sh: torch.Tensor):
        shape = img.shape[-1]
        img = filtered_resizing(img, size=512, f=self.resample_filter)
        img = rgb_to_lab(img.clamp(-1, 1) / 2. + .5)
        L = img[:, :1, :, :] / 100.
        L_p, _ = self.network(L, sh, 0)
        img = torch.cat((L_p * 100., img[:, 1:2], img[:, 2:3]), dim=1)
        return filtered_resizing(lab_to_rgb(img) * 2 - 1, size=shape, f=self.resample_filter)