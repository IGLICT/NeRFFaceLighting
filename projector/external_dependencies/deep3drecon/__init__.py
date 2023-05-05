import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from PIL import Image
import torch
import numpy as np
from mtcnn import MTCNN
from .options.test_options import TestOptions
from .models.facerecon_model import facereconmodel
from .util.load_mats import load_lm3d

#------------------------------------------------------------------------------------------------------------

# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

from PIL import ImageFilter

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=1024., mask=None, blur_sigma=3):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size
    img = img.resize((w, h), resample=Image.LANCZOS)
    
    # print(f"Cropping by ({left}, {up}, {right}, {below})")
    
    pad = int(max(-left, -up, right - img.size[0], below - img.size[1], 0))
    if pad > 0:
        print(f"Padding with {pad} ...")
        _img = np.array(img)
        _img = np.pad(_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        _img = Image.fromarray(_img)
        if blur_sigma > 0:
            _img = _img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        _img.paste(img, (pad, pad))
        img = _img
        
        left += pad
        up += pad
        right += pad
        below += pad
    
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.LANCZOS)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])
    return img, lm, mask


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=1024., rescale_factor=466.285):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    # img.save("/home/koki/Projects/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/iphone/epoch_20_000000/img_new.jpg")    
    trans_params = np.array([w0, h0, s, t[0], t[1]])
    lm_new *= 224/1024.0
    img_new_low = img_new.resize((224, 224), resample=Image.LANCZOS)

    return trans_params, img_new_low, lm_new, mask_new, img_new

#------------------------------------------------------------------------------------------------------------

def eg3d_detect_keypoints(detector, img: Image.Image):
    image = np.array(img)
    res = detector.detect_faces(image)
    
    if len(res) > 0:
        index = 0
        # Select the largest face
        if len(res) > 1:
            lowest_dist = float('Inf')
            for r in range(len(res)):
                face_pos = np.array(res[r]["box"][:2]) + np.array(res[r]["box"][2:])/2

                dist_from_center = np.linalg.norm(face_pos - np.array([1500./2, 1500./2]))
                if dist_from_center < lowest_dist:
                    lowest_dist = dist_from_center
                    index=r
        
        keypoints = res[index]['keypoints']
        if res[index]["confidence"] > 0.9:
            keypoints = [
                float(keypoints['left_eye'][0]), float(keypoints['left_eye'][1]), 
                float(keypoints['right_eye'][0]), float(keypoints['right_eye'][1]), 
                float(keypoints['nose'][0]), float(keypoints['nose'][1]), 
                float(keypoints['mouth_left'][0]), float(keypoints['mouth_left'][1]), 
                float(keypoints['mouth_right'][0]), float(keypoints['mouth_right'][1])
            ]
            return keypoints
        else:
            raise ValueError(f"Cannot find valid face in the image.")
    else:
        raise ValueError(f"Cannot find valid face in the image.")

#------------------------------------------------------------------------------------------------------------

class Deep3DWrapper(object):
    def __init__(self, device: torch.device):
        super().__init__()
        self.opt = TestOptions("--name=pretrained --epoch=20 --gpu_ids=0").parse()
        self.opt.bfm_folder = os.path.join(os.path.dirname(__file__), "BFM")
        self.opt.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        self.lm3d_std = load_lm3d(self.opt.bfm_folder)
        self.rescale_factor = (300, 466.285)
        self.center_crop_size = 700
        self.output_size = 512
        
        print("Load Deep3D ...")
        self.model = facereconmodel(self.opt)
        self.model.setup(self.opt)
        self.model.device = device
        self.model.eval()
        self.model.net_recon.to(device)
        
        print("Load MTCNN Detector ...")
        self.detector = MTCNN()
    def __call__(self, im: Image.Image):
        keypoints = eg3d_detect_keypoints(self.detector, im)
        _, H = im.size
        lm = np.array(keypoints).reshape(-1, 2)
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im_pil, lm, _, im_high = align_img(im, lm, self.lm3d_std, rescale_factor=self.rescale_factor[1])
        
        im = torch.from_numpy(np.array(im, dtype=np.float32)/255.).permute(2, 0, 1)[None, ...]
        lm = torch.from_numpy(lm)[None, ...]
        self.model.set_input({'imgs': im, 'lms': lm})
        self.model.test()
        
        pred_coeffs = {key:self.model.pred_coeffs_dict[key].cpu().numpy() for key in self.model.pred_coeffs_dict}
        pred_lm = self.model.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0], self.model.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        
        # Adjust the lighting coeffs
        a, c = self.model.facemodel.SH.a, self.model.facemodel.SH.c
        pred_coeffs['gamma'] = pred_coeffs['gamma'].reshape(-1, 3, 9)
        pred_coeffs['gamma'] = pred_coeffs['gamma'] + self.model.facemodel.init_lit.cpu().numpy()
        pred_coeffs['gamma'] = pred_coeffs['gamma'].transpose(0, 2, 1) # (B, 9, 3)
        
        pred_coeffs['gamma'][:, 0] *= a[0] * c[0] / (0.5 * np.sqrt(1 / np.pi))
        pred_coeffs['gamma'][:, 1] *= -a[1] * c[1] / (np.sqrt(3 / (4 * np.pi)))
        pred_coeffs['gamma'][:, 2] *= a[1] * c[1] / (np.sqrt(3 / (4 * np.pi)))
        pred_coeffs['gamma'][:, 3] *= -a[1] * c[1] / (np.sqrt(3 / (4 * np.pi)))
        pred_coeffs['gamma'][:, 4] *= a[2] * c[2] / (0.5 * np.sqrt(15 / np.pi))
        pred_coeffs['gamma'][:, 5] *= -a[2] * c[2] / (0.5 * np.sqrt(15 / np.pi))
        pred_coeffs['gamma'][:, 6] *= 0.5 * a[2] * c[2] / np.sqrt(3.) / (0.25 * np.sqrt(5 / np.pi))
        pred_coeffs['gamma'][:, 7] *= -a[2] * c[2] / (0.5 * np.sqrt(15 / np.pi))
        pred_coeffs['gamma'][:, 8] *= 0.5 * a[2] * c[2] / (0.25 * np.sqrt(15 / np.pi))
        return pred_coeffs