import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import dnnlib
from torch_utils import misc
from torch_utils import persistence

from training.triplane import OSGDecoder
from training.volumetric_rendering import math_utils
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes

from training.networks_stylegan2 import FullyConnectedLayer, SynthesisBlock, MappingNetwork

#----------------------------------------------------------------------------

class ShadingDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'], activation='softexp', alpha=options['alpha'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)
        x = self.net(x)
        x = x.view(N, M, -1)
        
        return {'shading': x}

#----------------------------------------------------------------------------

class LitImportanceRenderer(ImportanceRenderer):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
    def run_model(
        self, 
        planes, 
        lit_planes, 
        decoder, 
        decoder_shading, 
        sample_coordinates, 
        sample_directions, 
        options, 
    ):
        if options.get('return_normal', False) == True:
            with torch.enable_grad():
                sample_coordinates = sample_coordinates.requires_grad_(True)
                sampled_features = sample_from_planes(
                    self.plane_axes, 
                    planes, 
                    sample_coordinates, 
                    padding_mode='zeros', 
                    box_warp=options['box_warp']
                )
                
                out = decoder(sampled_features, sample_directions)
                out['normal'] = torch.autograd.grad(torch.sum(out['sigma']), sample_coordinates, create_graph=True)[0].detach().to(torch.float32)
                
            sample_coordinates = sample_coordinates.requires_grad_(False)
        else:
            sampled_features = sample_from_planes(
                self.plane_axes, 
                planes, 
                sample_coordinates, 
                padding_mode='zeros', 
                box_warp=options['box_warp']
            )
            
            out = decoder(sampled_features, sample_directions)
        
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        
        if not options.get('density_only', False) and lit_planes is not None:
            sampled_shading_features = sample_from_planes(
                self.plane_axes, 
                lit_planes, 
                sample_coordinates, 
                padding_mode='zeros', 
                box_warp=options['box_warp']
            )
            
            # Merge `shading` information
            out_shading = decoder_shading(sampled_shading_features, sample_directions)
            for name in out_shading:
                out[name] = out_shading[name]
        return out
    def unify_samples(self, 
        depths1, colors1, densities1, albedo1, normal1, shading1, 
        depths2, colors2, densities2, albedo2, normal2, shading2, 
        rendering_options):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        
        all_shading = torch.cat([shading1, shading2], dim = -2)
        all_shading = torch.gather(all_shading, -2, indices.expand(-1, -1, -1, all_shading.shape[-1]))
        
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        
        if not rendering_options.get('shading_only', False):
            all_colors = torch.cat([colors1, colors2], dim = -2)
            all_albedo = torch.cat([albedo1, albedo2], dim = -2)
            
            all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
            all_albedo = torch.gather(all_albedo, -2, indices.expand(-1, -1, -1, all_albedo.shape[-1]))
        else:
            all_colors = None
            all_albedo = None
        
        with torch.no_grad():
            if rendering_options.get('return_normal', False):
                all_normal = torch.cat([normal1, normal2], dim = -2)
                all_normal = torch.gather(all_normal, -2, indices.expand(-1, -1, -1, 3))
            else:
                all_normal = None

        return all_depths, all_colors, all_densities, all_albedo, all_normal, all_shading
    def forward(
        self, 
        planes, 
        lit_planes, 
        decoder, 
        decoder_shading, 
        ray_origins, 
        ray_directions, 
        rendering_options, 
    ):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        
        out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
        shading_coarse = out['shading']
        densities_coarse = out['sigma']
        
        if not rendering_options.get('shading_only', False):
            # Apply Shading
            albedo_coarse = out['rgb']
            colors_coarse = albedo_coarse * shading_coarse
            albedo_coarse = albedo_coarse.reshape(batch_size, num_rays, samples_per_ray, albedo_coarse.shape[-1])
            colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
            
            if rendering_options.get('return_normal', False):
                normal_coarse = out['normal']
                normal_coarse = normal_coarse.reshape(batch_size, num_rays, samples_per_ray, 3)
            else:
                normal_coarse = None
        else:
            albedo_coarse = None
            colors_coarse = None
            normal_coarse = None
        
        shading_coarse = shading_coarse.reshape(batch_size, num_rays, samples_per_ray, shading_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        assert N_importance > 0
        
        _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

        out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
        shading_fine = out['shading']
        densities_fine = out['sigma']
        
        if not rendering_options.get('shading_only', False):
            # Apply Shading
            albedo_fine = out['rgb']
            colors_fine = albedo_fine * shading_fine
            albedo_fine = albedo_fine.reshape(batch_size, num_rays, N_importance, albedo_fine.shape[-1])
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            
            if rendering_options.get('return_normal', False):
                normal_fine = out['normal']
                normal_fine = normal_fine.reshape(batch_size, num_rays, N_importance, 3)
            else:
                normal_fine = None
        else:
            albedo_fine = None
            colors_fine = None
            normal_fine = None
        
        shading_fine = shading_fine.reshape(batch_size, num_rays, N_importance, shading_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
        
        all_depths, all_colors, all_densities, all_albedos, all_normals, all_shadings = self.unify_samples(
            depths_coarse, colors_coarse, densities_coarse, albedo_coarse, normal_coarse, shading_coarse, 
            depths_fine, colors_fine, densities_fine, albedo_fine, normal_fine, shading_fine, 
            rendering_options = rendering_options, 
        )

        # Aggregate
        rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        if not rendering_options.get('shading_only', False):
            albedo_final = torch.sum(weights * ((all_albedos[:, :, :-1] + all_albedos[:, :, 1:]) / 2), dim=-2) * 2 - 1 # Scale to (-1, 1)
            if rendering_options.get('return_normal', False):
                with torch.no_grad():
                    normals = (all_normals[:, :, :-1] + all_normals[:, :, 1:]) / 2
                    normals = -normals
                    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7) # (B, H*W, N, 3)
                    normal_final = torch.sum(weights * normals, -2)
                    normal_final = normal_final / (torch.norm(normal_final, dim=-1, keepdim=True) + 1e-7)
            else:
                normal_final = None
        else:
            albedo_final = None
            normal_final = None
        
        shading_final = torch.sum(weights * ((all_shadings[:, :, :-1] + all_shadings[:, :, 1:]) / 2), dim=-2) # (0, 1)
        
        return rgb_final, depth_final, albedo_final, normal_final, shading_final, weights.sum(2)

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = self.channels_dict[res // 2] if res > 4 else 0
            out_channels = self.channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        
        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        # Output `x` at the same time
        return x, img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLighting(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channels,                   # Fixed Number of channels.
        num_blocks,                 # Number of synthesis blocks.
        num_fp16_blk    = 4,        # Use FP16 for the N highest blocks.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert num_fp16_blk <= num_blocks
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_fp16_blk = num_fp16_blk
        
        self.num_ws = 0
        for num_block in range(num_blocks):
            use_fp16 = (num_block >= num_blocks - num_fp16_blk)
            is_last = (num_block == num_blocks - 1)
            block = SynthesisBlock(channels, channels, w_dim=w_dim, resolution=img_resolution,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, disable_upsample=True, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{num_block}', block)

    def forward(self, x, img, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for num_block in range(self.num_blocks):
                block = getattr(self, f'b{num_block}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        misc.assert_shape(x, [None, self.channels, self.img_resolution, self.img_resolution])
        misc.assert_shape(img, [None, self.img_channels, self.img_resolution, self.img_resolution])
        for num_block, cur_ws in zip(range(self.num_blocks), block_ws):
            block = getattr(self, f'b{num_block}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_blk={self.num_fp16_blk:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        l_dim,                      # Conditioning lighting (L) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs       = {},  # Arguments for MappingNetwork.
        mapping_lit_kwargs   = {},  # Arguments for MappingNetwork of Lighting.
        synthesis_lit_kwargs = {},  # Arguments for SynthesisLighting.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.l_dim = l_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        # Lighting Modules
        self.synthesis_lit = SynthesisLighting(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, channels=self.synthesis.channels_dict[img_resolution], **synthesis_lit_kwargs)
        self.num_ws_lit = self.synthesis_lit.num_ws
        self.mapping_lit = MappingNetwork(z_dim=0, c_dim=l_dim, w_dim=w_dim, num_ws=self.num_ws_lit, **mapping_lit_kwargs)

    def forward(self, z, c, l, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        x, img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        ws_lit = self.mapping_lit(None, l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        lit_img = self.synthesis_lit(x, img, ws_lit, update_emas=update_emas, **synthesis_kwargs)
        return img, lit_img

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        l_dim,                      # Conditioning lighting (L) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        mapping_lit_kwargs  = {},   # Arguments for MappingNetwork of Lighting.
        synthesis_lit_kwargs= {},   # Arguments for SynthesisLighting.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.l_dim=l_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = LitImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = Generator(z_dim, c_dim, l_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, mapping_lit_kwargs=mapping_lit_kwargs, synthesis_lit_kwargs=synthesis_lit_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.shading_decoder = ShadingDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 1, 'alpha': 0.1})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, l, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas), self.backbone.mapping_lit(None, l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, ws_lit, c, l, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, rendering_kwargs={}, return_planes=False, _planes_lit=None, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        rendering_options = self.rendering_kwargs
        for name in rendering_kwargs:
            rendering_options[name] = rendering_kwargs[name]
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes, planes_lit = self._last_planes
        else:
            x, planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            if _planes_lit == None:
                planes_lit = self.backbone.synthesis_lit(x, planes, ws_lit, update_emas=update_emas, **synthesis_kwargs)
            else:
                planes_lit = _planes_lit
        if cache_backbone:
            self._last_planes = planes, planes_lit
        
        if return_planes:
            return {'planes': planes, 'planes_lit': planes_lit}

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes_lit = planes_lit.view(len(planes_lit), 3, 32, planes_lit.shape[-2], planes_lit.shape[-1])
        
        # Perform volume rendering
        feature_samples, depth_samples, albedo_samples, normal_samples, shading_samples, weights_samples = self.renderer(planes, planes_lit, self.decoder, self.shading_decoder, ray_origins, ray_directions, rendering_options) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        if not rendering_options.get('shading_only', False):
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            albedo_image = albedo_samples.permute(0, 2, 1).reshape(N, albedo_samples.shape[-1], H, W)
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            if rendering_options.get('return_normal', False):
                normal_image = normal_samples.permute(0, 2, 1).reshape(N, normal_samples.shape[-1], H, W)
            else:
                normal_image = None
            
            # Run superresolution to get final image
            rgb_image = feature_image[:, :3]
            sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=rendering_options['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            
            albedo_image_raw = albedo_image[:, :3]
            albedo_image_sr = self.superresolution(albedo_image_raw, albedo_image, ws, noise_mode=rendering_options['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = None
            rgb_image = None
            albedo_image_raw = None
            albedo_image_sr = None
            normal_image = None
            depth_image = None
        shading_image = shading_samples.permute(0, 2, 1).reshape(N, shading_samples.shape[-1], H, W)
        
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_albedo': albedo_image_sr, 'image_albedo_raw': albedo_image_raw, 'image_normal': normal_image, 'image_shading': shading_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        _, planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, None, self.decoder, None, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        x, planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, None, self.decoder, None, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, l, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws, ws_lit = self.mapping(z, c, l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, ws_lit, c, l, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

#----------------------------------------------------------------------------
