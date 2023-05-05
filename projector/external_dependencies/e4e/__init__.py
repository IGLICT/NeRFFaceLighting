import torch
from .encoders.psp_encoders import Encoder4Editing

def load_e4e_standalone(checkpoint_path, device='cuda', in_channel=3):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    e4e = Encoder4Editing(50, 'ir_se', 14, True, in_channel)
    # e4e_lit = Encoder4Editing(50, 'ir_se', 7, False)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    # e4e_lit_dict = {k.replace('encoder_lit.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder_lit.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    # e4e_lit.load_state_dict(e4e_lit_dict)
    # e4e_lit.eval()
    # e4e_lit = e4e_lit.to(device)
    
    latent_avg = ckpt['latent_avg'].to(device)
    # latent_lit_avg = ckpt['latent_avg_lit'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)
    
    # def add_latent_lit_avg(model, inputs, outputs):
    #     return outputs + latent_lit_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    # e4e_lit.register_forward_hook(add_latent_lit_avg)
    return e4e #, e4e_lit

#----------------------------------------------------------------------------
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from external_dependencies.facemesh.facemesh import FaceMesh

from torch.nn import functional as F
from torchvision.transforms import functional as Fv

facemesh = FaceMesh().to("cuda")
facemesh.load_weights(os.path.join(os.path.dirname(__file__), "../facemesh/facemesh.pth"))
flip_indices = np.array([0, 1, 2, 248, 4, 5, 6, 390, 8, 9, 10, 11, 12, 12, 14, 15, 16, 17, 18, 125, 354, 251, 252, 253, 254, 339, 341, 257, 258, 259, 260, 448, 262, 249, 264, 265, 266, 267, 312, 303, 304, 311, 271, 273, 274, 275, 276, 277, 439, 279, 280, 281, 282, 283, 284, 285, 286, 291, 288, 305, 328, 308, 407, 293, 439, 295, 296, 297, 298, 299, 300, 301, 302, 311, 310, 290, 407, 324, 415, 459, 271, 268, 312, 313, 314, 315, 317, 317, 402, 318, 318, 320, 322, 323, 141, 318, 319, 326, 460, 326, 329, 330, 278, 332, 333, 334, 335, 336, 337, 338, 254, 340, 463, 467, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 19, 420, 264, 357, 278, 263, 360, 433, 463, 363, 364, 365, 366, 367, 368, 369, 94, 371, 372, 373, 374, 324, 376, 377, 378, 379, 151, 152, 380, 382, 362, 353, 384, 385, 386, 387, 387, 368, 373, 164, 391, 309, 393, 168, 394, 395, 396, 397, 414, 399, 175, 400, 401, 317, 402, 404, 405, 406, 272, 415, 407, 410, 411, 412, 413, 413, 272, 416, 417, 418, 195, 419, 197, 363, 199, 200, 421, 422, 423, 424, 425, 426, 427, 428, 360, 430, 431, 432, 433, 434, 435, 436, 437, 309, 392, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 357, 454, 289, 456, 457, 461, 459, 305, 354, 370, 464, 465, 465, 388, 466, 3, 33, 238, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 130, 127, 35, 36, 37, 81, 39, 185, 80, 191, 43, 44, 45, 46, 47, 129, 49, 50, 51, 52, 53, 54, 55, 56, 212, 58, 235, 75, 57, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 40, 240, 61, 146, 61, 218, 74, 73, 38, 83, 84, 85, 86, 178, 95, 95, 146, 91, 92, 93, 77, 146, 99, 98, 60, 100, 101, 129, 103, 104, 105, 106, 107, 108, 109, 25, 111, 155, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 156, 241, 126, 127, 128, 129, 130, 209, 132, 155, 198, 135, 136, 137, 138, 162, 140, 242, 142, 143, 163, 145, 146, 147, 148, 149, 150, 153, 154, 154, 156, 157, 158, 159, 161, 246, 162, 7, 165, 219, 167, 169, 170, 171, 172, 173, 174, 176, 177, 88, 88, 180, 181, 182, 76, 61, 61, 186, 187, 188, 190, 173, 184, 192, 193, 194, 196, 126, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 132, 214, 215, 216, 217, 218, 64, 220, 221, 222, 223, 224, 225, 226, 227, 31, 229, 230, 231, 232, 233, 234, 64, 236, 79, 238, 79, 98, 238, 20, 133, 243, 244, 33, 113])
flip_indices = torch.from_numpy(flip_indices).to("cuda")
X, Y = torch.meshgrid(torch.arange(256, dtype=int, device="cuda"), torch.arange(256, dtype=int, device="cuda"), indexing='ij')
indices = torch.stack((X, Y), axis=-1).to(torch.float)/256

@torch.no_grad()
def get_flip_face(img: torch.Tensor, return_mask: bool=True):
    assert img.size(0) == 1
    detections = facemesh.predict_on_image((F.interpolate(img.clamp(-1, 1), (192, 192), mode='bilinear', align_corners=False, antialias=True)[0]/2+.5)*255.)[:, :2]*(1/192)
    detect_matrix = torch.topk(torch.sum((detections[:, None, :] - detections[None, :, :])**2, dim=-1), k=5, dim=-1, largest=False, sorted=True).values[:, -1]
    flip_detections = detections[flip_indices]
    dist_matrix = torch.sum((indices[:, :, None, :] - detections[None, None, :, :])**2, dim=-1)
    dist_matrix = (-dist_matrix) / detect_matrix[None, None, :]
    dist_weight = F.softmax(dist_matrix, dim=-1)
    dist_weight = torch.nan_to_num(dist_weight)
    warped_indices = torch.matmul(dist_weight, flip_detections)*2-1
    warped = torch.flip(Fv.rotate(F.grid_sample(img, warped_indices[None, ...], mode='bilinear', align_corners=False), 90), dims=(-2, ))
    
    if return_mask:
        hull = ConvexHull(detections.cpu().numpy())
        polygons = np.round(detections.cpu().numpy()[hull.vertices]*256).astype(np.int64).tolist()
        polygons = [tuple(e) for e in polygons]
        mask = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygons, fill=255, outline=None)
        mask = torch.from_numpy(np.array(mask).astype(np.float32)/255.)
        mask = mask[None, None, ...].to(img.device)
    else:
        mask = None
    
    return mask, warped

@torch.no_grad()
def get_flip_face_on_batch(img_s: torch.Tensor):
    size = img_s.size(0)
    mask_s = []
    warped_s = []
    for _ in range(size):
        mask, warped = get_flip_face(img_s[_][None, ...], True)
        mask_s.append(mask)
        warped_s.append(warped)
    return torch.cat(mask_s, dim=0), torch.cat(warped_s, dim=0)

#----------------------------------------------------------------------------