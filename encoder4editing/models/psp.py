import torch
import numpy as np
from torch import nn
from models.encoders import psp_encoders
from models.eg3d.networks_stylegan_lit import TriPlaneGenerator as Generator
from models.eg3d.camera_utils import *
from configs.paths_config import model_paths

from torch_utils import misc

init_kwargs = {'c_dim': 25, 'l_dim': 9, 'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'mapping_lit_kwargs': {'num_layers': 1}, 'synthesis_lit_kwargs': {'num_blocks': 3, 'num_fp16_blk': 0}, 'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only', 'rendering_kwargs': {'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 'superresolution_module': 'models.eg3d.superresolution.SuperresolutionHybrid8XDC', 'c_gen_conditioning_zero': False, 'gpc_reg_prob': 0.5, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True, 'lighting_reg': 25.0, 'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2]}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4, 'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'img_resolution': 512, 'img_channels': 3}


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.decoder = Generator(**init_kwargs).to(self.opts.device)
        self.encoder = self.set_encoder(self.decoder.backbone.num_ws, opts.disable_mixing, in_channel=3*(2 if opts.flip_input else 1))
        self.encoder_lit = self.set_encoder(self.decoder.backbone.num_ws_lit, True, in_channel=3*(2 if opts.flip_input else 1))
        
        # Define camera
        self.fov_deg = 18.837
        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=opts.device)
        self.cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=opts.device)
        self.cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
        self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=opts.device)
        self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self, style_count, disable_mixing=False, in_channel=3):
        return psp_encoders.Encoder4Editing(50, 'ir_se', style_count, disable_mixing, in_channel=in_channel)

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.eval().requires_grad_(False)
            misc.copy_params_and_buffers(encoder_ckpt, self.encoder)
            self.encoder.train().requires_grad_(True)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt, strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count, repeat_lit=self.encoder_lit.style_count)

    def forward(self, x, c, l, resize=True, randomize_noise=True, return_latents=False, mask_light=False):
        codes = self.encoder(x)
        codes_lit = self.encoder_lit(x)
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
            if not mask_light:
                codes_lit = codes_lit + self.latent_avg_lit.repeat(codes_lit.shape[0], 1, 1)
            else:
                codes_lit = codes_lit * 0. + self.latent_avg_lit.repeat(codes_lit.shape[0], 1, 1)
        
        images = self.decoder(codes, c, codes_lit, l, input_is_latent=True, noise_mode='const' if not randomize_noise else 'random')

        if resize:
            images["image"] = self.face_pool(images["image"])
            images["image_albedo"] = self.face_pool(images["image_albedo"])

        if return_latents:
            return images, (codes, codes_lit)
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None, repeat_lit=None):
        if 'latent_avg' in ckpt and 'latent_avg_lit' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            self.latent_avg_lit = ckpt['latent_avg_lit'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                z_samples = np.random.RandomState(123).randn(10000, self.decoder.z_dim)
                w_samples = self.decoder.backbone.mapping(torch.from_numpy(z_samples).to(self.opts.device), self.conditioning_params.expand(10000, -1), truncation_psi=1., truncation_cutoff=14, update_emas=False)
                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
                w_avg = np.mean(w_samples, axis=0)
                self.latent_avg = torch.from_numpy(w_avg).to(self.opts.device)
                self.latent_avg_lit = self.decoder.backbone.mapping_lit.w_avg.detach().clone().reshape(1, -1).to(self.opts.device)
        else:
            self.latent_avg = None
            self.latent_avg_lit = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
        if repeat_lit is not None and self.latent_avg_lit is not None:
            self.latent_avg_lit = self.latent_avg_lit.repeat(repeat_lit, 1)