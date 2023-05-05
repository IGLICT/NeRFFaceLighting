import os
import json
import random
import torch
import numpy as np
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import make_grid

from criteria import id_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from lpips import LPIPS
from models.psp import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger

from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from configs.paths_config import dataset_paths
from external_dependencies.facemesh.facemesh import FaceMesh

random.seed(0)
torch.manual_seed(0)

def load_labels():
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    _path = dataset_paths['ffhq']
    _all_fnames = os.listdir(_path)
    _image_fnames = sorted(fname for fname in _all_fnames if _file_ext(fname) in ['.png'])
    
    fname = os.path.join(dataset_paths['ffhq'], 'dataset.json')
    with open(fname, "r") as f:
        data = json.load(f)
        labels = data['labels']
        shs = data['sh']
    
    labels = dict(labels)
    labels = [labels[fname] for fname in _image_fnames]
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    
    shs = dict(shs)
    shs = [shs[fname] for fname in _image_fnames]
    shs = np.array(shs)
    shs = shs.astype({1: np.int64, 2: np.float32}[shs.ndim])
    
    assert labels.shape[0] == shs.shape[0]
    length = labels.shape[0]
    
    assert length > 0
    
    def iterate_random_labels(batch_size, device):
        while True:
            c = [labels[np.random.randint(length)] for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(device)
            
            l = [shs[np.random.randint(length)] for _i in range(batch_size)]
            l = torch.from_numpy(np.stack(l)).pin_memory().to(device)
            yield c, l
    return iterate_random_labels

#----------------------------------------------------------------------------

from torchvision.transforms import functional as Fv

facemesh = FaceMesh().to("cuda")
facemesh.load_weights("./external_dependencies/facemesh/facemesh.pth")
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

class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device
        # Initialize network
        self.net = pSp(self.opts).to(self.device)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # Initialize sampler
        self.label_iter = load_labels()(opts.batch_size, self.device)

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net=self.opts.lpips_type).to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.scaler = torch.cuda.amp.GradScaler(1)

        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        # Initialize dataset
        self.train_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.scaler.get_scale() < 1.:
                    self.scaler.update(1.)
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                x, y, y_hat, latent = self.forward(batch, mask_light=self.opts.mask_light)
                y_hat_albedo = y_hat["image_albedo"]
                if self.opts.mask_light:
                    y_hat = y_hat["image_albedo"]
                else:
                    y_hat = y_hat["image"]
                loss, encoder_loss_dict, _ = self.calc_loss(x, y, y_hat, y_hat_albedo, latent[0], latent[1])
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 100 == 0):
                    self.parse_and_log_images(x, y, y_hat, title='images/train/faces')
                    # self.parse_and_log_images_gen(samples_x, samples_y, samples_a, gen_image, gen_albedo, title='images/train_gen/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(self.scaler.get_scale(), loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                    
                    # self.print_metrics(loss_dict_a, prefix='train_gen')
                    # self.log_metrics(loss_dict_a, prefix='train_gen')

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
                self.net.encoder_lit.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
                self.net.encoder_lit.set_progressive_stage(ProgressiveStage(i))
    
    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters()) + list(self.net.encoder_lit.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      label_path=dataset_args['label_path'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        print("Number of training samples: {}".format(len(train_dataset)))
        return train_dataset

    def calc_loss(self, x, y, y_hat, y_hat_albedo, latent, latent_lit):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            assert self.is_progressive_training()
            dims_to_discriminate = self.get_dims_to_discriminate()

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 13:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.net.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y).mean()
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.sym_lambda > 0:
            face_mask, flip_albedo = get_flip_face_on_batch(y_hat_albedo)
            loss_sym = self.lpips_loss(y_hat_albedo, flip_albedo * face_mask + y_hat_albedo * (1 - face_mask)).mean()
            loss_dict['loss_sym'] = float(loss_sym)
            loss += loss_sym * self.opts.sym_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch, mask_light=False):
        x, y, labels = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        cam, sh = labels['cam'].to(self.device).float(), labels['sh'].to(self.device).float()
        if self.opts.flip_input:
            mask, flip_x = get_flip_face_on_batch(x)
            y_hat, latents = self.net.forward(torch.cat((x, flip_x * mask + x * (1 - mask)), dim=1), cam, sh, return_latents=True, mask_light=mask_light)
        else:
            y_hat, latents = self.net.forward(x, cam, sh, return_latents=True, mask_light=mask_light)
        return x, y, y_hat, latents

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, scale, metrics_dict, prefix):
        print('Metrics for {}, step {}, scale {}'.format(prefix, self.global_step, scale))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=2):
        self.log_images(title, im_data={
            'input_face': x, 
            'target_face': y, 
            'output_face': y_hat, 
        })
    
    def parse_and_log_images_gen(self, x, y, a, y_hat, a_hat, title):
        step = self.global_step
        self.logger.add_image(title, make_grid(torch.cat([
            x, 
            y, 
            a, 
            y_hat, 
            a_hat, 
        ], dim=0), x.size(0), normalize=True), step)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        step = self.global_step
        self.logger.add_image(name, make_grid(torch.cat([
            im_data['input_face'], 
            im_data['target_face'], 
            im_data['output_face']
        ], dim=0), im_data['input_face'].size(0), normalize=True), step)
        
    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
            save_dict['latent_avg_lit'] = self.net.latent_avg_lit

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None
    @torch.no_grad()
    def sample_fake(self):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        sample_c, sample_l = next(self.label_iter)
        real_w = self.net.decoder.backbone.mapping(sample_z, sample_c)
        real_w_lit = self.net.decoder.backbone.mapping_lit(sample_z, sample_l)
        out = self.net.decoder(real_w, sample_c, real_w_lit, sample_l, input_is_latent=True)
        out["image"] = self.face_pool(out["image"])
        out["image_albedo"] = self.face_pool(out["image_albedo"])
        return out, sample_c, sample_l

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):
        loss_dict = {}
        x, _, _ = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.net.decoder.backbone.mapping(sample_z, self.net.conditioning_params.expand(sample_z.size(0), -1))
        if self.opts.flip_input:
            mask, flip_x = get_flip_face_on_batch(x)
            x = torch.cat((x, flip_x * mask + x * (1 - mask)), dim=1)
        fake_w = self.net.encoder(x)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
