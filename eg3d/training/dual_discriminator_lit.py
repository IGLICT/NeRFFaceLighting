# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered
        
    return ada_filtered_64

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        l_dim,                          # Conditioning light (L) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.l_dim = l_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0
        
        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        
        # Pose Branch.
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        
        # Light & Pose Branch.
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'l{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim + l_dim > 0:
            self.mapping_l = MappingNetwork(z_dim=0, c_dim=c_dim+l_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.l4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise

    def forward(self, img, c, l, update_emas=False, **block_kwargs):
        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
        img_o = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        
        # Pose Branch.
        x, img = None, img_o
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        
        # Light Branch.
        xl, imgl = None, img_o
        for res in self.block_resolutions:
            block = getattr(self, f'l{res}')
            xl, imgl = block(xl, imgl, **block_kwargs)
        
        cmap = None
        if self.c_dim + self.l_dim > 0:
            cmap = self.mapping_l(None, torch.cat((c, l), dim=1))
        xl = self.l4(xl, imgl, cmap)
        
        misc.assert_shape(x, [None, 1])
        misc.assert_shape(xl, [None, 1])
        
        return torch.cat((x, xl), dim=1)

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, l_dim={self.l_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
