import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../external_dependencies/face_evolve/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../external_dependencies/face_evolve/applications/align"))

import cv2
import copy
import lpips
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as Fv
import pickle
import dnnlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from torchvision import transforms
from training.networks_stylegan_lit import TriPlaneGenerator

from external_dependencies.e4e import load_e4e_standalone, get_flip_face_on_batch
from external_dependencies.facemesh.facemesh import FaceMesh
from external_dependencies.face_parsing.model import BiSeNet
from face_evolve import BACKBONE as ArcFaceBackbone

class Projector(object):
    def __init__(self, device: torch.device, ckpt_path: str, encoder_path: str = None):
        self.device = device
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.synthesis_kwargs = {'noise_mode': 'const'}
        self.encoder_path = encoder_path
        
        print("Load Generator ...")
        with open(os.path.join(ckpt_path), 'rb') as f:
            _G = pickle.load(f)['G_ema'].to(device).eval()
        self.G = TriPlaneGenerator(*_G.init_args, **_G.init_kwargs).requires_grad_(False).eval().to(device)
        misc.copy_params_and_buffers(_G, self.G)
        self.G.neural_rendering_resolution = _G.neural_rendering_resolution
        self.G.rendering_kwargs['depth_resolution'] = 96
        self.G.rendering_kwargs['depth_resolution_importance'] = 96
        
        self.fov_deg = 18.837
        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=device)
        self.cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        self.cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=device)
        self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)
        
        if encoder_path != None:
            print("Load Encoders ...")
            e4e = load_e4e_standalone(encoder_path, device, 6 if "ag" in encoder_path else 3)
            self.e4e = e4e
            self.e4e_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        print("Load VGG16 ...")
        vgg16_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/vgg16.pt'))
        with dnnlib.util.open_url(vgg16_path) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)
        
        print("Load ArcFace ...")
        self.arcface = ArcFaceBackbone.to(device).eval()
        
        print("Load Face-parsing ...")
        self.face2seg_ = BiSeNet(n_classes=19).to(device).eval()
        self.face2seg_.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../external_dependencies/face_parsing/79999_iter.pth"), map_location=device))
        self.face2seg = lambda x: self.face2seg_(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x))[0]
        
        print("Load FaceMesh ...")
        self.facemesh = FaceMesh().to(device)
        self.facemesh.load_weights(os.path.join(os.path.dirname(__file__), "../external_dependencies/facemesh/facemesh.pth"))
        self.flip_indices = np.array([0, 1, 2, 248, 4, 5, 6, 390, 8, 9, 10, 11, 12, 12, 14, 15, 16, 17, 18, 125, 354, 251, 252, 253, 254, 339, 341, 257, 258, 259, 260, 448, 262, 249, 264, 265, 266, 267, 312, 303, 304, 311, 271, 273, 274, 275, 276, 277, 439, 279, 280, 281, 282, 283, 284, 285, 286, 291, 288, 305, 328, 308, 407, 293, 439, 295, 296, 297, 298, 299, 300, 301, 302, 311, 310, 290, 407, 324, 415, 459, 271, 268, 312, 313, 314, 315, 317, 317, 402, 318, 318, 320, 322, 323, 141, 318, 319, 326, 460, 326, 329, 330, 278, 332, 333, 334, 335, 336, 337, 338, 254, 340, 463, 467, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 19, 420, 264, 357, 278, 263, 360, 433, 463, 363, 364, 365, 366, 367, 368, 369, 94, 371, 372, 373, 374, 324, 376, 377, 378, 379, 151, 152, 380, 382, 362, 353, 384, 385, 386, 387, 387, 368, 373, 164, 391, 309, 393, 168, 394, 395, 396, 397, 414, 399, 175, 400, 401, 317, 402, 404, 405, 406, 272, 415, 407, 410, 411, 412, 413, 413, 272, 416, 417, 418, 195, 419, 197, 363, 199, 200, 421, 422, 423, 424, 425, 426, 427, 428, 360, 430, 431, 432, 433, 434, 435, 436, 437, 309, 392, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 357, 454, 289, 456, 457, 461, 459, 305, 354, 370, 464, 465, 465, 388, 466, 3, 33, 238, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 130, 127, 35, 36, 37, 81, 39, 185, 80, 191, 43, 44, 45, 46, 47, 129, 49, 50, 51, 52, 53, 54, 55, 56, 212, 58, 235, 75, 57, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 40, 240, 61, 146, 61, 218, 74, 73, 38, 83, 84, 85, 86, 178, 95, 95, 146, 91, 92, 93, 77, 146, 99, 98, 60, 100, 101, 129, 103, 104, 105, 106, 107, 108, 109, 25, 111, 155, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 156, 241, 126, 127, 128, 129, 130, 209, 132, 155, 198, 135, 136, 137, 138, 162, 140, 242, 142, 143, 163, 145, 146, 147, 148, 149, 150, 153, 154, 154, 156, 157, 158, 159, 161, 246, 162, 7, 165, 219, 167, 169, 170, 171, 172, 173, 174, 176, 177, 88, 88, 180, 181, 182, 76, 61, 61, 186, 187, 188, 190, 173, 184, 192, 193, 194, 196, 126, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 132, 214, 215, 216, 217, 218, 64, 220, 221, 222, 223, 224, 225, 226, 227, 31, 229, 230, 231, 232, 233, 234, 64, 236, 79, 238, 79, 98, 238, 20, 133, 243, 244, 33, 113])
        self.flip_indices = torch.from_numpy(self.flip_indices).to(device)
        X, Y = torch.meshgrid(torch.arange(512, dtype=int, device=device), torch.arange(512, dtype=int, device=device))
        self.indices = torch.stack((X, Y), axis=-1).to(torch.float)/512
        
        print("Init Inversion ...")
        self.w_avg_samples              = 10000
        self.num_steps                  = 500
        self.initial_learning_rate      = 0.01
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5

        with torch.no_grad():
            # Compute w stats.
            self.z_samples = np.random.RandomState(123).randn(self.w_avg_samples, self.G.z_dim)
            self.w_samples = self.mapping(self.G, torch.from_numpy(self.z_samples).to(device), self.conditioning_params.expand(self.w_avg_samples, -1), truncation_psi=1.)
            self.w_samples = self.w_samples[:, :1, :].cpu().numpy().astype(np.float32)
            self.w_avg = np.mean(self.w_samples, axis=0, keepdims=True)
            self.w_std = (np.sum((self.w_samples - self.w_avg) ** 2) / self.w_avg_samples) ** 0.5
        
        self.steps = 500
    @staticmethod
    def convert_tensor_to_numpy_image(image):
        return ((image/2+.5).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    @torch.no_grad()
    def parse_face(self, img: torch.Tensor):
        seg = torch.argmax(self.face2seg(img), dim=1, keepdim=True)
        return {
            "eyes": torch.logical_or(seg == 4, seg == 5).float(), 
            "skin": (seg == 1).float(), 
            "hair": (seg == 17).float(), 
        }
    def get_flip_face(self, img: torch.Tensor):
        assert img.shape[0] == 1
        detections = torch.from_numpy(self.facemesh.predict_on_image(cv2.resize(self.convert_tensor_to_numpy_image(img.clamp(-1, 1)[0]), (192, 192))).cpu().numpy()[:, :2]*(1/192)).to(img.device)
        detect_matrix = torch.topk(torch.sum((detections[:, None, :] - detections[None, :, :])**2, dim=-1), k=5, dim=-1, largest=False, sorted=True).values[:, -1]
        flip_detections = detections[self.flip_indices]
        indices = self.indices.clone()
        dist_matrix = torch.sum((indices[:, :, None, :] - detections[None, None, :, :])**2, dim=-1)
        dist_matrix = (-dist_matrix) / detect_matrix[None, None, :]
        dist_weight = F.softmax(dist_matrix, dim=-1)
        dist_weight = torch.nan_to_num(dist_weight)
        warped_indices = torch.matmul(dist_weight, flip_detections)*2-1
        warped = torch.flip(Fv.rotate(F.grid_sample(img, warped_indices[None, ...], mode='bilinear', align_corners=False), 90), dims=(-2, ))
        
        hull = ConvexHull(detections.cpu().numpy())
        polygons = np.round(detections.cpu().numpy()[hull.vertices]*512).astype(np.int64).tolist()
        polygons = [tuple(e) for e in polygons]
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygons, fill=255, outline=None)
        mask = torch.from_numpy(np.array(mask).astype(np.float32)/255.)
        mask = mask[None, None, ...].to(img.device)
        
        return mask, warped
    def get_face_feat(self, img: torch.Tensor):
        img = F.interpolate(img, (256, 256), mode='bilinear')
        img = img[:, :, 35:223, 32:220]
        img = torch.nn.AdaptiveAvgPool2d((112, 112))(img)
        feat = self.arcface(img)
        return feat / (torch.norm(feat, p=2, dim=-1, keepdim=True) + 1e-8)
    def get_dist_of_face_feats(self, feat1: torch.Tensor, feat2: torch.Tensor):
        return 1 - (feat1 * feat2).sum(dim=-1).mean(dim=0)
    @staticmethod
    def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1.):
        return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=14, update_emas=False)
    @staticmethod
    def mapping_lit(G, l: torch.Tensor):
        return G.backbone.mapping_lit(None, l, update_emas=False)
    @staticmethod
    def encode(G, ws, w_lit, **synthesis_kwargs):
        x, planes = G.backbone.synthesis(ws, update_emas=False, **synthesis_kwargs)
        planes_lit = G.backbone.synthesis_lit(x, planes, w_lit, update_emas=False, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes_lit = planes_lit.view(len(planes_lit), 3, 32, planes_lit.shape[-2], planes_lit.shape[-1])
        return planes.detach().clone(), planes_lit.detach().clone()
    @staticmethod
    def decode(G, 
        ws: torch.Tensor, 
        cam: torch.Tensor, 
        planes: torch.Tensor, 
        **synthesis_kwargs
    ):
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)
        neural_rendering_resolution = G.neural_rendering_resolution
        
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        N, M, _ = ray_origins.shape
        
        feature_samples, _, albedo_samples, _, _, _ = G.renderer(planes[0], planes[1], G.decoder, G.shading_decoder, ray_origins, ray_directions, G.rendering_kwargs)
        
        # Reshape into 'raw' neural-rendered image
        H = W = G.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        
        albedo_image = albedo_samples.permute(0, 2, 1).reshape(N, albedo_samples.shape[-1], H, W).contiguous()
        
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = G.superresolution(
            rgb_image, 
            feature_image, 
            ws, 
            noise_mode=G.rendering_kwargs['superresolution_noise_mode'], 
            **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'}
        )
        
        albedo_image_raw = albedo_image[:, :3]
        albedo_image_sr = G.superresolution(albedo_image_raw, albedo_image, ws, noise_mode=G.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        
        return {
            'image_raw': rgb_image, 
            'image': sr_image, 
            'image_albedo': albedo_image_sr if getattr(G, "shading_decoder", None) != None else None, 
        }
    def mask_loss(self, loss_fn, x, y, mask):
        return (torch.sum(loss_fn(x * mask, y * mask), dim=(-1, -2)) / (torch.sum(mask, dim=(-1, -2)) + 1e-5)).mean()
    def mask_out(self, x, mask):
        return x * mask + x.detach() * (1 - mask)
    def project(self, 
        img: Image.Image, 
        camera_params: torch.Tensor, 
        sh: torch.Tensor, 
        # Projection Options
        optimize_sh: bool = True, 
        skip_pti: bool = False, 
        lambda_id: float = 0.1, 
        lambda_flip: float = 0.25, 
        lambda_sim: float = 0.5, 
        lambda_detail: float = 0.1, 
        finetune_shading: bool = True, 
    ):
        assert img.size == (512, 512)
        w_lit = self.mapping_lit(self.G, sh)

        w_opt_loss = []
        sh_opt_loss = []

        # Compute w pivot
        G = copy.deepcopy(self.G).eval().requires_grad_(False)
        
        if self.encoder_path != None:
            input_img = self.e4e_transform(img).to(self.device)[None, ...]
            with torch.no_grad():
                if self.e4e.in_channel == 3:
                    w_opt = self.e4e(input_img)
                else:
                    mask, flip_img = get_flip_face_on_batch(input_img)
                    w_opt = self.e4e(torch.cat((input_img, flip_img * mask + input_img * (1 - mask)), dim=1))
        else:
            start_w = self.w_avg
            noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}
            
            target_image = torch.from_numpy(np.array(img, dtype=np.uint8).transpose([2, 0, 1])).to(self.device).unsqueeze(0).to(torch.float32)
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')
            target_features = self.vgg16(target_image, resize_images=False, return_lpips=True)
            
            w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=self.initial_learning_rate)
            
            for step in range(self.steps * 2):
                # Learning rate schedule.
                t = step / (self.steps * 2)
                w_noise_scale = self.w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
                lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
                lr = self.initial_learning_rate * lr_ramp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Synth images from opt_w.
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
                
                synth_image = G.synthesis(ws, G.backbone.mapping_lit.w_avg[None, None, :].expand_as(w_lit).to(w_lit.device), camera_params, None, **self.synthesis_kwargs)["image"]
                
                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                synth_image = (synth_image + 1) * (255/2)
                synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')

                # Features for synth images.
                synth_features = self.vgg16(synth_image, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum()
                
                # Noise regularization.
                reg_loss = 0.0
                for v in noise_bufs.values():
                    noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                    while True:
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                        if noise.shape[2] <= 8:
                            break
                        noise = F.avg_pool2d(noise, kernel_size=2)

                loss = dist + reg_loss * self.regularize_noise_weight

                w_opt_loss.append(float(loss))
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            w_opt = w_opt.detach().repeat([1, 14, 1]).requires_grad_(False)
        
        if optimize_sh:
            target_image = torch.from_numpy(np.array(img, dtype=np.uint8).transpose([2, 0, 1])).to(self.device).unsqueeze(0).to(torch.float32)
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')
            target_features = self.vgg16(target_image, resize_images=False, return_lpips=True)
            
            sh_opt = sh.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([sh_opt], betas=(0.9, 0.999), lr=self.initial_learning_rate)
            for step in range(100):
                synth_image = G.synthesis(w_opt, self.mapping_lit(self.G, sh_opt), camera_params, None, **self.synthesis_kwargs)["image"]
                
                synth_image = (synth_image + 1) * (255/2)
                synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')

                # Features for synth images.
                synth_features = self.vgg16(synth_image, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum()
                
                loss = dist
                sh_opt_loss.append(float(loss))
                
                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            sh_opt = sh_opt.detach().clone().requires_grad_(False)
            w_lit = self.mapping_lit(self.G, sh_opt)
        else:
            sh_opt = sh
        
        pti_loss = []

        if not skip_pti:
            G.train().requires_grad_(True)
            G.superresolution.requires_grad_(False)
            
            if not finetune_shading:
                G.shading_decoder.requires_grad_(False)
                G.backbone.synthesis_lit.requires_grad_(False)
            
            optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)

            with torch.no_grad():
                target_images = torch.from_numpy(
                    np.array(img, dtype=np.uint8).transpose([2, 0, 1])
                ).to(self.device).unsqueeze(0).to(torch.float32) / 255. * 2 - 1
                target_image_mask, target_flip_image = self.get_flip_face(target_images)
                target_flip_image = target_image_mask * target_flip_image + (1 - target_image_mask) * target_images
                target_image = target_image_mask * target_images + (1 - target_image_mask) * target_images
                
                target_images_masks = self.parse_face(target_images)
            
            for step in range(self.steps):
                out = G.synthesis(w_opt, w_lit, camera_params, None, **self.synthesis_kwargs)
                synth_image = out["image"]
                albedo_image = out["image_albedo"]
                
                loss_l1 = torch.nn.L1Loss()(synth_image, target_images)
                loss_lpips = torch.squeeze(self.lpips_loss(synth_image, target_images))
                
                albedo_face_mask, albedo_flip_image = self.get_flip_face(albedo_image)
                albedo_flip_image = albedo_face_mask * albedo_flip_image + (1 - albedo_face_mask) * albedo_image.detach()
                albedo_image = albedo_face_mask * albedo_image + (1 - albedo_face_mask) * albedo_image.detach()
                
                loss_id = \
                        self.get_dist_of_face_feats(self.get_face_feat(albedo_image), self.get_face_feat(target_image)) + \
                        self.get_dist_of_face_feats(self.get_face_feat(albedo_image), self.get_face_feat(target_flip_image))
                
                loss_flip = torch.squeeze(self.lpips_loss(albedo_image, albedo_flip_image))
                loss_sim = torch.squeeze(self.lpips_loss(albedo_image, target_image))
                            
                loss_detail = \
                                torch.squeeze(self.lpips_loss(self.mask_out(albedo_image, target_images_masks["eyes"]), self.mask_out(target_image, target_images_masks["eyes"]))) + \
                                self.mask_loss(lambda x,y: (x-y).abs(), synth_image, target_images, target_images_masks["eyes"])
                
                loss = loss_l1 + loss_lpips + loss_id * 0.5 * lambda_id + loss_flip * lambda_flip + loss_sim * lambda_sim + loss_detail * lambda_detail

                pti_loss.append({
                    "loss": float(loss), 
                    "l1": float(loss_l1), 
                    "lpips": float(loss_lpips), 
                    "id": float(loss_id), 
                    "flip": float(loss_flip), 
                    "sim": float(loss_sim), 
                    "detail": float(loss_detail), 
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return G.requires_grad_(False), w_opt.detach().clone(), sh_opt.detach().clone(), \
                w_opt_loss, sh_opt_loss, pti_loss