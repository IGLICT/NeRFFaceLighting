import torch
import argparse
from models.psp import pSp
from models.encoders.psp_encoders import Encoder4Editing


def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts


def load_e4e_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    e4e = Encoder4Editing(50, 'ir_se', 14, False)
    e4e_lit = Encoder4Editing(50, 'ir_se', 7, True)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    e4e_lit_dict = {k.replace('encoder_lit.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder_lit.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    e4e_lit.load_state_dict(e4e_lit_dict)
    e4e_lit.eval()
    e4e_lit = e4e_lit.to(device)
    
    latent_avg = ckpt['latent_avg'].to(device)
    latent_lit_avg = ckpt['latent_lit_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)
    
    def add_latent_lit_avg(model, inputs, outputs):
        return outputs + latent_lit_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    e4e_lit.register_forward_hook(add_latent_lit_avg)
    return e4e, e4e_lit
