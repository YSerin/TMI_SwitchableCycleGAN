import itertools
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_ssim
import numpy as np

from utils.image_pool import ImagePool
from networks.discriminator import Discriminator2 as Discriminator
from networks.adain import half_PolyPhase_resUnet_Adain as Generator


import ipdb

class Model():
    def __init__(self, opt, current_i):
        self.opt = opt
        if self.opt.gpu_parallel:
            self.device = torch.device('cuda:{}'.format(opt.multiple_ids[0]))
        else:
            self.device = torch.device('cuda:{}'.format(opt.gpu_ids))
        self.max_Val_Gan_loss = 0

        self.visual_names = ['real_S', 'real_H', 'fake_S', 'fake_H', 'rec_S', 'rec_H']
        self.loss_names = ['cycle_S_loss', 'cycle_H_loss', 'cycle_total_loss',
                           'identity_S_loss', 'identity_H_loss', 'identity_total_loss',
                           'Gan_S_loss', 'Gan_H_loss', 'Gan_total_loss',
                           'total_D_S_loss','total_D_H_loss']
        self.loss_val_names = ['Val_cycle_S_loss', 'Val_cycle_H_loss', 'Val_cycle_total_loss',
                           'Val_identity_S_loss', 'Val_identity_H_loss', 'Val_identity_total_loss',
                           'Val_Gan_S_loss', 'Val_Gan_H_loss', 'Val_Gan_total_loss']

        self.model_names = ['netG', 'netD_H', 'netD_S']

        self.netG = Generator()

        self.netD_S = Discriminator()
        self.netD_H = Discriminator()

        if torch.cuda.device_count() > 1 and self.opt.gpu_parallel:
            self.netG = nn.DataParallel(self.netG, device_ids = self.opt.multiple_ids, output_device=self.opt.multiple_ids[0])
            self.netD_S = nn.DataParallel(self.netD_S, device_ids = self.opt.multiple_ids, output_device=self.opt.multiple_ids[0])
            self.netD_H = nn.DataParallel(self.netD_H, device_ids = self.opt.multiple_ids, output_device=self.opt.multiple_ids[0])
            self.netG.to(self.device)
            self.netD_S.to(self.device)
            self.netD_H.to(self.device)
        else:
            self.netG.to(self.device)
            self.netD_S.to(self.device)
            self.netD_H.to(self.device)

        if self.opt.isTrain:
            self.fake_S_pool = ImagePool(self.opt.pool_size, self.opt.pool_prob)
            self.fake_H_pool = ImagePool(self.opt.pool_size, self.opt.pool_prob)

            self.criterion_GAN = nn.MSELoss()
            self.criterion_Cycle = nn.L1Loss()
            self.criterion_Identity = nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_S.parameters(),self.netD_H.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            if self.opt.lr_schedule and not self.opt.continue_train:
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, milestones=opt.milestones, gamma=0.5)

        if not self.opt.isTrain or (opt.continue_train and current_i):
            load_suffix = '%d' % opt.load_epoch
            self.load_networks(load_suffix)
        if self.opt.isTrain and opt.continue_train and current_i:
            load_suffix = '%d' % opt.load_epoch
            self.load_optimizer(load_suffix)
        self.print_networks()

    def set_input(self, data):
        self.real_S = data['real_S'].to(self.device)
        self.real_H = data['real_H'].to(self.device)
        self.id = data['id']
        self.path_H = data['path_H']
        self.path_S = data['path_S']

    def forward(self):
        self.fake_H = self.netG(self.real_S, alpha=0.0)
        self.rec_S = self.netG(self.fake_H, alpha=1.0)
        self.fake_S = self.netG(self.real_H, alpha=1.0)
        self.rec_H = self.netG(self.fake_S, alpha=0.0)

    def backward_G(self):

        # Identity loss
        self.identity_S = self.netG(self.real_H, alpha=0.0)
        self.identity_H = self.netG(self.real_S, alpha=1.0)
        self.identity_S_loss = self.criterion_Identity(self.identity_S, self.real_H)
        self.identity_H_loss = self.criterion_Identity(self.identity_H, self.real_S)
        self.identity_total_loss= self.identity_S_loss + self.identity_H_loss

        # Gan Loss
        fake_H_logits = self.netD_S(self.fake_H)
        fake_S_logits = self.netD_H(self.fake_S)
        self.Gan_S_loss = self.criterion_GAN(fake_H_logits, torch.ones_like(fake_H_logits))
        self.Gan_H_loss = self.criterion_GAN(fake_S_logits, torch.ones_like(fake_S_logits))
        self.Gan_total_loss = (self.Gan_S_loss + self.Gan_H_loss)

        # Cycle Loss
        self.cycle_S_loss = self.criterion_Cycle(self.rec_S, self.real_S)
        self.cycle_H_loss = self.criterion_Cycle(self.rec_H, self.real_H)
        self.cycle_total_loss = (self.cycle_S_loss + self.cycle_H_loss)

        #total loss
        self.total_g_loss = (self.opt.lambda_gan * self.Gan_total_loss +
                        self.opt.lambda_identity * self.identity_total_loss +
                        self.opt.lambda_cycle * self.cycle_total_loss )

        self.total_g_loss.backward()


    def backward_D_S(self):

        # real
        pred_real = self.netD_S(self.real_H)
        D_real_loss = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # fake
        fake_H = self.fake_H_pool.query(self.fake_H)
        pred_fake = self.netD_S(fake_H.detach())
        D_fake_loss = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        self.total_D_S_loss = (D_real_loss + D_fake_loss)/2
        self.total_D_S_loss.backward()

    def backward_D_H(self):

        # real
        pred_real = self.netD_H(self.real_S)
        D_real_loss = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # fake
        fake_S = self.fake_S_pool.query(self.fake_S)
        pred_fake = self.netD_H(fake_S.detach())
        D_fake_loss = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        self.total_D_H_loss = (D_real_loss + D_fake_loss) / 2
        self.total_D_H_loss.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    # main loop
    def optimize_parameters(self):

        self.forward()
        self.set_requires_grad([self.netD_S, self.netD_H], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_S, self.netD_H], True)
        self.optimizer_D.zero_grad()
        self.backward_D_S()
        self.backward_D_H()
        self.optimizer_D.step()
        if self.opt.lr_schedule and not self.opt.continue_train:
            self.lr_scheduler.step()

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, validation=False):
        errors_ret = OrderedDict()
        if validation:
            for name in self.loss_val_names:
                if isinstance(name, str):
                    errors_ret[name] = getattr(self,name)

        else:
            for name in self.loss_names:
                if isinstance(name, str):
                    errors_ret[name] = getattr(self, name)
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pth' % (epoch, name)
                save_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                         .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], save_filename))
                net = getattr(self, name)
                torch.save(net.state_dict(),save_path)
        save_filename = '%s_optimizer.pth' % (epoch)
        save_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                 .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], save_filename))
        torch.save({
            'optimizerG_state_dict': self.optimizer_G.state_dict(),
            'optimizerD_state_dict': self.optimizer_D.state_dict(),
            'max_Val_Gan_loss': self.max_Val_Gan_loss}, save_path)

    def save_best_metrics(self, epoch, PSNR_H_max,PSNR_S_max,SSIM_H_max,SSIM_S_max):
        save_filename = 'best_metrics.pth'
        save_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                 .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], save_filename))
        torch.save({
            'epoch': epoch,
            'PSNR_H_max': PSNR_H_max,
            'PSNR_S_max': PSNR_S_max,
            'SSIM_H_max': SSIM_H_max,
            'SSIM_S_max': SSIM_S_max}, save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if self.opt.phase == 'test' and name in ['netD_H', 'netD_S']:
                pass
            else:
                if isinstance(name, str):
                    load_filename = '%s_%s.pth' % (epoch, name)
                    if self.opt.phase =='test':
                        load_path = os.path.join(self.opt.save_dir, '{}/{}'.format(self.opt.name, load_filename))
                    else:
                        load_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                             .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], load_filename))
                    net = getattr(self, name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)

                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    if (self.opt.phase == 'test' and self.opt.gpu_parallel_train)\
                            or (self.opt.phase == 'train' and self.opt.continue_train and self.opt.gpu_parallel):
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:]
                            new_state_dict[name] = v
                        net.load_state_dict(new_state_dict)
                    else:
                        net.load_state_dict(state_dict)


    def load_optimizer(self, epoch):
        load_filename = '%s_optimizer.pth' % (epoch)
        load_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                 .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], load_filename))
        checkpoint = torch.load(load_path,map_location=self.device)
        self.optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.max_Val_Gan_loss = checkpoint['max_Val_Gan_loss']

    def load_best_metrics(self):
        load_filename = 'best_metrics.pth'
        load_path = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/{}'
                                 .format(self.opt.name, self.opt.val_id[0], self.opt.val_id[1], load_filename))
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch']
        p_H = checkpoint['PSNR_H_max']
        p_S = checkpoint['PSNR_S_max']
        s_H = checkpoint['SSIM_H_max']
        s_S = checkpoint['SSIM_S_max']
        return epoch, p_H, p_S, s_H, s_S

    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def test(self):
        with torch.no_grad():
            self.forward()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def eval_metrics(self):
        # unpreprocessing
        id_list = sorted(os.listdir(self.opt.data_dir + '{}_total'.format(self.opt.data_type)))
        dir_stat = self.opt.data_dir + 'statistic_{}'.format(self.opt.data_type)
        if self.opt.data_dir.split('/')[-2] == 'dataset_M':
            mean_H = np.load(dir_stat + '/mean_Hr68.npz')['meanlist']
            mean_S = np.load(dir_stat + '/mean_Hr40.npz')['meanlist']
            std_H = np.load(dir_stat + '/std_Hr68.npz')['stdlist']
            std_S = np.load(dir_stat + '/std_Hr40.npz')['stdlist']
        elif self.opt.data_dir.split('/')[-2] == 'dataset':
            mean_H = np.load(dir_stat + '/mean_H.npz')['meanlist']
            mean_S = np.load(dir_stat + '/mean_S.npz')['meanlist']
            std_H = np.load(dir_stat + '/std_H.npz')['stdlist']
            std_S = np.load(dir_stat + '/std_S.npz')['stdlist']
        else:
            raise ValueError

        img_H = self.real_H * std_H[id_list.index(self.id[0][0])] + mean_H[id_list.index(self.id[0][0])]
        img_S = self.real_S * std_S[id_list.index(self.id[1][0])] + mean_S[id_list.index(self.id[1][0])]
        img_HfromS = self.fake_H * std_S[id_list.index(self.id[1][0])] + mean_S[id_list.index(self.id[1][0])]
        img_SfromH = self.fake_S * std_H[id_list.index(self.id[0][0])] + mean_H[id_list.index(self.id[0][0])]

        # H window max: 400 + 1500/2 = 1150
        # S window max: 50 + 120/2 = 110
        img_H = (torch.clamp(self._unpreprocessing(img_H), 400 - 1500 / 2, 400 + 1500 / 2) / 1150).cpu()
        img_HfromS = (torch.clamp(self._unpreprocessing(img_HfromS), 400 - 1500 / 2, 400 + 1500 / 2) / 1150).cpu()
        img_S = (torch.clamp(self._unpreprocessing(img_S), 50 - 120 / 2, 50 + 120 / 2) / 110).cpu()
        img_SfromH = (torch.clamp(self._unpreprocessing(img_SfromH), 50 - 120 / 2, 50 + 120 / 2) / 110).cpu()

        mse_H = torch.mean((img_H - img_HfromS) ** 2, dim=[1, 2, 3])
        mse_S = torch.mean((img_S - img_SfromH) ** 2, dim=[1, 2, 3])
        score_H = -10 * torch.log10(mse_H)
        score_S = -10 * torch.log10(mse_S)

        # SSIM
        SSIM_fake_H = pytorch_ssim.ssim(img_H, img_HfromS)
        SSIM_fake_S = pytorch_ssim.ssim(img_S, img_SfromH)

        return (SSIM_fake_H, SSIM_fake_S, score_H, score_S)


    def _unpreprocessing(self, image):
        output = image
        mu_h2o = 0.0192
        output = (output - mu_h2o) * 1000 / mu_h2o  ### TODO: output [output<0] = 0
        return output

    def calc_loss(self):
        with torch.no_grad():
            identity_S = self.netG(self.real_H, alpha=0.0)
            identity_H = self.netG(self.real_S, alpha=1.0)
            self.Val_identity_S_loss = self.criterion_Identity(identity_S, self.real_H)
            self.Val_identity_H_loss = self.criterion_Identity(identity_H, self.real_S)
            self.Val_identity_total_loss = self.Val_identity_S_loss + self.Val_identity_H_loss

            fake_H_logits = self.netD_S(self.fake_H)
            fake_S_logits = self.netD_H(self.fake_S)
            self.Val_Gan_S_loss = self.criterion_GAN(fake_H_logits, torch.ones_like(fake_H_logits))
            self.Val_Gan_H_loss = self.criterion_GAN(fake_S_logits, torch.ones_like(fake_S_logits))
            self.Val_Gan_total_loss = (self.Val_Gan_S_loss + self.Val_Gan_H_loss)

            # Cycle Loss
            self.Val_cycle_S_loss = self.criterion_Cycle(self.rec_S, self.real_S)
            self.Val_cycle_H_loss = self.criterion_Cycle(self.rec_H, self.real_H)
            self.Val_cycle_total_loss = (self.Val_cycle_S_loss + self.Val_cycle_H_loss)

            if self.Val_Gan_total_loss > self.max_Val_Gan_loss:
                self.max_Val_Gan_loss = self.Val_Gan_total_loss
                updated = True
                return updated
            else:
                updated = False
                return updated
