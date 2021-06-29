import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.val_id = opt.val_id

        if self.opt.isTrain:
            summary_path = os.path.join(self.opt.save_dir, self.opt.name, 'validation_id_{}_{}'.format(self.val_id[0], self.val_id[1]), 'runs')
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            self.tboard = SummaryWriter(summary_path)
            self.log_name = os.path.join(opt.save_dir, self.opt.name, 'validation_id_{}_{}'.format(self.val_id[0], self.val_id[1]), 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        self.log_name_test = os.path.join(opt.save_dir, opt.name, 'validation_id_{}_{}'.format(self.val_id[0], self.val_id[1]), 'loss_inference_{}.txt'.format(self.val_id[0], self.val_id[1]))

        dir_stat_HS = self.opt.data_dir[:-3] + '/statistic_{}'.format(self.opt.data_type)
        dir_stat = self.opt.data_dir + 'statistic_{}'.format(self.opt.data_type)
        if opt.data_dir.split('/')[-2] == 'dataset_M':
            self.mean_H = np.load(dir_stat + '/mean_Hr68.npz')['meanlist']
            self.mean_M = np.load(dir_stat + '/mean_Hr49.npz')['meanlist']
            self.mean_S = np.load(dir_stat + '/mean_Hr40.npz')['meanlist']
            self.std_H = np.load(dir_stat + '/std_Hr68.npz')['stdlist']
            self.std_M = np.load(dir_stat + '/std_Hr49.npz')['stdlist']
            self.std_S = np.load(dir_stat + '/std_Hr40.npz')['stdlist']
            self.id_list_HS = sorted(os.listdir(self.opt.data_dir + '{}_total'.format(self.opt.data_type)))
            self.id_list_M = sorted(os.listdir(self.opt.data_dir + '{}_total'.format(self.opt.data_type)))
            self.id_list = self.id_list_HS
        elif opt.data_dir.split('/')[-2] == 'dataset_KEY':
            self.mean_H = np.load(dir_stat + '/mean_H.npz')['meanlist']
            self.mean_S = np.load(dir_stat + '/mean_S.npz')['meanlist']
            self.std_H = np.load(dir_stat + '/std_H.npz')['stdlist']
            self.std_S = np.load(dir_stat + '/std_S.npz')['stdlist']
            self.id_list = sorted(os.listdir(self.opt.data_dir + '{}_total'.format(self.opt.data_type)))
        else:
            raise ValueError


    def display_images(self, visuals, epoch, batch=None):
        if self.opt.phase == 'train':
            for label, image in visuals.items():
                image = image[0:1,...]
                image = torch.add(image, torch.mul(-1, torch.min(image)))
                if torch.max(image) != 0: image = torch.div(image, torch.max(image))
                self.tboard.add_images(label, image)


    def save_images(self, visuals, epoch, step_train):
        image_path = os.path.join(self.opt.save_dir,'{}/validation_id_{}_{}/images/epoch_{}'.format(self.opt.name, self.val_id[0],self.val_id[1], epoch))
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        fig = plt.figure(figsize=(30, 20))
        i = 0
        for label, image in visuals.items():
            image_arr = image.cpu().detach().numpy()[0, 0, :, :]
            ax = fig.add_subplot(3,3,i+1)
            ax.set_title(label=label, loc='center')
            ax.imshow(image_arr, cmap='gray')
            ax.title.set_text('{}'.format(label))
            i += 1

        plt.savefig(os.path.join(image_path,'step_{}_images.png'.format(step_train)))
        plt.close(fig)

    def display_losses(self, losses, epoch, data_cnt):
        for key, val in losses.items():
            self.tboard.add_scalar(key, val, epoch + data_cnt)

    def print_losses(self, losses, epoch, iters):
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)
        print(message)  # print the message
        if self.opt.isTrain:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  
        else:
            with open(self.log_name_test, "a") as log_file:
                log_file.write('%s\n' % message)  


    def save_losses(self, epoch, subject_num, subject_dice, subject_loss):
        message = '(epoch: %d, subject_number: %s) ' % (epoch, subject_num)
        message += 'dice_score: %.6f ' % (subject_dice)
        message += 'loss: %.6f ' % (subject_loss)
        print(message)  
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  

    def save_image_summary_train(self, visuals, epoch, step, id):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/train'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        id_H = id[0][0]
        id_S = id[1][0]
        real_S = visuals['real_S'].cpu().detach().numpy()
        real_H = visuals['real_H'].cpu().detach().numpy()

        fake_S = visuals['fake_S'].cpu().detach().numpy()
        fake_H = visuals['fake_H'].cpu().detach().numpy()

        rec_S = visuals['rec_S'].cpu().detach().numpy()
        rec_H = visuals['rec_H'].cpu().detach().numpy()

        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        clip_low_H = (H_low_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_up_H = (H_high_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_low_S = (S_low_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]
        clip_up_S = (S_high_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]


        real_S = np.clip(real_S, clip_low_S, clip_up_S)
        rec_S = np.clip(rec_S, clip_low_S, clip_up_S)
        real_H = np.clip(real_H, clip_low_H, clip_up_H)
        rec_H = np.clip(rec_H, clip_low_H, clip_up_H)
        fake_S = np.clip(fake_S, clip_low_S, clip_up_S)
        fake_H = np.clip(fake_H, clip_low_H, clip_up_H)

        display_list = [real_S[0, 0, ...], fake_H[0, 0, ...], rec_S[0,0,...],
                        real_H[0, 0, ...], fake_S[0, 0, ...], rec_H[0,0,...],]
        title = ['real S', 'fake H', 'Rec S',
                     'real H', 'fake S', 'rec H']

        # Results
        fig = plt.figure(figsize=(40, 30))
        for i in range(len(display_list)):
            axs = fig.add_subplot(2, 3, i + 1)
            axs.set_title(label=title[i], loc='center')
            im = axs.imshow(np.squeeze(display_list[i]), cmap='gray')
            fig.colorbar(im, ax=axs)

        fig.savefig(save_dir + '/Summary_epoch_{}_step_{}.png'.format(epoch, step))
        fig.clear()
        plt.clf()
        plt.close('all')
    
    def save_image_summary_val(self, model, epoch, step, id):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/val'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        id_H = id[0][0]
        id_S = id[1][0]
        real_S = model.real_S.cpu().detach().numpy()
        real_H = model.real_H.cpu().detach().numpy()

        with torch.no_grad():
            fake_S = model.netG(model.real_H, alpha_s=0.0, alpha_t=1.0).cpu().detach().numpy()
            fake_H = model.netG(model.real_S, alpha_s=1.0, alpha_t=0.0).cpu().detach().numpy()

            identity_S = model.netG(model.real_H, alpha_s=0.0, alpha_t=0.0).cpu().detach().numpy()
            identity_H = model.netG(model.real_S, alpha_s=1.0, alpha_t=1.0).cpu().detach().numpy()

        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        clip_low_H = (H_low_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_up_H = (H_high_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_low_S = (S_low_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]
        clip_up_S = (S_high_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]

        real_S_Swin = np.clip(real_S, clip_low_S, clip_up_S)
        real_S_Hwin = np.clip(real_S, clip_low_H, clip_up_H)
        real_H_Hwin = np.clip(real_H, clip_low_H, clip_up_H)
        real_H_Swin = np.clip(real_H, clip_low_S, clip_up_S)
        fake_S_Swin = np.clip(fake_S, clip_low_S, clip_up_S)
        fake_S_Hwin = np.clip(fake_S, clip_low_H, clip_up_H)
        fake_H_Hwin = np.clip(fake_H, clip_low_H, clip_up_H)
        fake_H_Swin = np.clip(fake_H, clip_low_S, clip_up_S)
        identity_H_Swin = np.clip(identity_H, clip_low_S, clip_up_S)
        identity_H_Hwin = np.clip(identity_H, clip_low_H, clip_up_H)
        identity_S_Hwin = np.clip(identity_S, clip_low_H, clip_up_H)
        identity_S_Swin = np.clip(identity_S, clip_low_S, clip_up_S)

        display_list = [real_H_Hwin[0, 0, ...], fake_H_Hwin[0, 0, ...], identity_H_Hwin[0,0,...], real_S_Hwin[0,0,...],
                        real_H_Hwin[0, 0, ...], identity_S_Hwin[0, 0, ...], fake_S_Hwin[0, 0, ...], real_S_Hwin[0, 0, ...],
                        real_H_Swin[0, 0, ...], fake_H_Swin[0, 0, ...], identity_H_Swin[0, 0, ...], real_S_Swin[0, 0, ...],
                        real_H_Swin[0, 0, ...], identity_S_Swin[0, 0, ...], fake_S_Swin[0,0,...], real_S_Swin[0,0,...]]
        title = ['real H', 'fake H (a=0)', 'idt H (a=1)', 'real S (input)',
                 'real H (input)', 'idt S (a=0)', 'fake S (a=1)', 'real S',
                 'real H', 'fake H (a=0)', 'idt H (a=1)', 'real S (input)',
                 'real H (input)', 'idt S (a=0)', 'fake S (a=1)', 'real S']

        # Results
        fig = plt.figure(figsize=(50, 50))
        for i in range(len(display_list)):
            axs = fig.add_subplot(4, 4, i + 1)
            axs.set_title(label=title[i], loc='center', size=50)
            axs.imshow(np.squeeze(display_list[i]), cmap='gray')

        fig.savefig(save_dir + '/Summary_epoch_{}_step_{}.png'.format(epoch, step))
        fig.clear()
        plt.clf()
        plt.close('all')

    def save_image_summary_test_parallel(self, model, dataset_val, index, epoch):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/val'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.opt.gpu_parallel:
            self.device = torch.device('cuda:{}'.format(self.opt.multiple_ids[0]))
        else:
            self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids))

        real_S = torch.zeros((len(index),1,512,512))
        real_H = torch.zeros((len(index),1,512,512))
        id = []
        for j, idx in enumerate(index):
            real_S[j] = dataset_val[idx]['real_S'].to(self.device)
            real_H[j] = dataset_val[idx]['real_H'].to(self.device)
            id.append(dataset_val[idx]['id'][0])

        real_S = real_S.to(self.device)
        real_H = real_H.to(self.device)
        with torch.no_grad():
            fake_S = model.netG(real_H, alpha_s=0.0, alpha_t=1.0).cpu().detach().numpy()
            fake_H = model.netG(real_S, alpha_s=1.0, alpha_t=0.0).cpu().detach().numpy()

            identity_S = model.netG(real_H, alpha_s=0.0, alpha_t=0.0).cpu().detach().numpy()
            identity_H = model.netG(real_S, alpha_s=1.0, alpha_t=1.0).cpu().detach().numpy()

            real_S = real_S.cpu().detach().numpy()
            real_H = real_H.cpu().detach().numpy()

        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        for j, idx in enumerate(index):
            clip_low_H = (H_low_mu - self.mean_H[self.id_list.index(str(id[j]))])/self.std_H[self.id_list.index(str(id[j]))]
            clip_up_H = (H_high_mu - self.mean_H[self.id_list.index(str(id[j]))])/self.std_H[self.id_list.index(str(id[j]))]
            clip_low_S = (S_low_mu - self.mean_S[self.id_list.index(str(id[j]))]) / self.std_S[self.id_list.index(str(id[j]))]
            clip_up_S = (S_high_mu - self.mean_S[self.id_list.index(str(id[j]))]) / self.std_S[self.id_list.index(str(id[j]))]

            real_S_Swin = np.clip(real_S, clip_low_S, clip_up_S)
            real_S_Hwin = np.clip(real_S, clip_low_H, clip_up_H)
            real_H_Hwin = np.clip(real_H, clip_low_H, clip_up_H)
            real_H_Swin = np.clip(real_H, clip_low_S, clip_up_S)
            fake_S_Swin = np.clip(fake_S, clip_low_S, clip_up_S)
            fake_S_Hwin = np.clip(fake_S, clip_low_H, clip_up_H)
            fake_H_Hwin = np.clip(fake_H, clip_low_H, clip_up_H)
            fake_H_Swin = np.clip(fake_H, clip_low_S, clip_up_S)
            identity_H_Swin = np.clip(identity_H, clip_low_S, clip_up_S)
            identity_H_Hwin = np.clip(identity_H, clip_low_H, clip_up_H)
            identity_S_Hwin = np.clip(identity_S, clip_low_H, clip_up_H)
            identity_S_Swin = np.clip(identity_S, clip_low_S, clip_up_S)

            display_list = [real_H_Hwin[j, 0, ...], fake_H_Hwin[j, 0, ...], identity_H_Hwin[j,0,...], real_S_Hwin[j,0,...],
                            real_H_Hwin[j, 0, ...], identity_S_Hwin[j, 0, ...], fake_S_Hwin[j, 0, ...], real_S_Hwin[j, 0, ...],
                            real_H_Swin[j, 0, ...], fake_H_Swin[j, 0, ...], identity_H_Swin[j, 0, ...], real_S_Swin[j, 0, ...],
                            real_H_Swin[j, 0, ...], identity_S_Swin[j, 0, ...], fake_S_Swin[j,0,...], real_S_Swin[j,0,...]]
            title = ['real H', 'fake H (a=0)', 'idt H (a=1)', 'real S (input)',
                     'real H (input)', 'idt S (a=0)', 'fake S (a=1)', 'real S',
                     'real H', 'fake H (a=0)', 'idt H (a=1)', 'real S (input)',
                     'real H (input)', 'idt S (a=0)', 'fake S (a=1)', 'real S']

            # Results
            fig = plt.figure(figsize=(50, 50))
            for i in range(len(display_list)):
                axs = fig.add_subplot(4, 4, i + 1)
                axs.set_title(label=title[i], loc='center', size=50)
                axs.imshow(np.squeeze(display_list[i]), cmap='gray')

            fig.savefig(save_dir + '/Summary_epoch_{}_step_{}.png'.format(epoch, idx))
            fig.clear()
            plt.clf()
            plt.close('all')

    def save_image_summary_train_M(self, visuals, epoch, step, id):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/train'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        id_H = id[0][0]
        id_S = id[1][0]
        real_S = visuals['real_S'].cpu().detach().numpy()
        real_H = visuals['real_H'].cpu().detach().numpy()

        fake_S = visuals['fake_SfromH'].cpu().detach().numpy()
        fake_H = visuals['fake_HfromS'].cpu().detach().numpy()

        rec_S = visuals['rec_SfromH'].cpu().detach().numpy()
        rec_H = visuals['rec_HfromS'].cpu().detach().numpy()

        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        clip_low_H = (H_low_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_up_H = (H_high_mu - self.mean_H[self.id_list.index(str(id_H))])/self.std_H[self.id_list.index(str(id_H))]
        clip_low_S = (S_low_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]
        clip_up_S = (S_high_mu - self.mean_S[self.id_list.index(str(id_S))]) / self.std_S[self.id_list.index(str(id_S))]


        real_S = np.clip(real_S, clip_low_S, clip_up_S)
        rec_S = np.clip(rec_S, clip_low_S, clip_up_S)
        real_H = np.clip(real_H, clip_low_H, clip_up_H)
        rec_H = np.clip(rec_H, clip_low_H, clip_up_H)
        fake_S = np.clip(fake_S, clip_low_S, clip_up_S)
        fake_H = np.clip(fake_H, clip_low_H, clip_up_H)

        display_list = [real_S[0, 0, ...], fake_H[0, 0, ...], rec_S[0,0,...],
                        real_H[0, 0, ...], fake_S[0, 0, ...], rec_H[0,0,...],]
        title = ['real S', 'fake H', 'Rec S',
                     'real H', 'fake S', 'rec H']

        # Results
        fig = plt.figure(figsize=(40, 30))
        for i in range(len(display_list)):
            axs = fig.add_subplot(2, 3, i + 1)
            axs.set_title(label=title[i], loc='center')
            im = axs.imshow(np.squeeze(display_list[i]), cmap='gray')
            fig.colorbar(im, ax=axs)

        fig.savefig(save_dir + '/Summary_epoch_{}_step_{}.png'.format(epoch, step))
        fig.clear()
        plt.clf()
        plt.close('all')

    def save_image_summary_val_M(self, model, epoch, step, id):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/val'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        id_H = id[0][0]
        id_M = id[2][0]
        id_S = id[1][0]
        real_S = model.real_S.cpu().detach().numpy()
        real_M = model.real_M.cpu().detach().numpy()
        real_H = model.real_H.cpu().detach().numpy()

        with torch.no_grad():
            SfromH = model.netG(model.real_H, alpha_s=0.0, alpha_t=1.0).cpu().detach().numpy()
            MfromH = model.netG(model.real_H, alpha_s=0.0, alpha_t=0.5).cpu().detach().numpy()
            HfromH = model.netG(model.real_H, alpha_s=0.0, alpha_t=0.0).cpu().detach().numpy()

            SfromS = model.netG(model.real_S, alpha_s=1.0, alpha_t=1.0).cpu().detach().numpy()
            MfromS = model.netG(model.real_S, alpha_s=1.0, alpha_t=0.5).cpu().detach().numpy()
            HfromS = model.netG(model.real_S, alpha_s=1.0, alpha_t=0.0).cpu().detach().numpy()

            SfromM = model.netG(model.real_M, alpha_s=0.5, alpha_t=1.0).cpu().detach().numpy()
            MfromM = model.netG(model.real_M, alpha_s=0.5, alpha_t=0.5).cpu().detach().numpy()
            HfromM = model.netG(model.real_M, alpha_s=0.5, alpha_t=0.0).cpu().detach().numpy()

        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        clip_low_H = (H_low_mu - self.mean_H[self.id_list_HS.index(str(id_H))])/self.std_H[self.id_list_HS.index(str(id_H))]
        clip_up_H = (H_high_mu - self.mean_H[self.id_list_HS.index(str(id_H))])/self.std_H[self.id_list_HS.index(str(id_H))]
        clip_low_S = (S_low_mu - self.mean_S[self.id_list_HS.index(str(id_S))]) / self.std_S[self.id_list_HS.index(str(id_S))]
        clip_up_S = (S_high_mu - self.mean_S[self.id_list_HS.index(str(id_S))]) / self.std_S[self.id_list_HS.index(str(id_S))]
        clip_low_H_M = (H_low_mu - self.mean_M[self.id_list_M.index(str(id_M))]) / self.std_M[self.id_list_M.index(str(id_M))]
        clip_up_H_M = (H_high_mu - self.mean_M[self.id_list_M.index(str(id_M))]) / self.std_M[self.id_list_M.index(str(id_M))]
        clip_low_S_M = (S_low_mu - self.mean_M[self.id_list_M.index(str(id_M))]) / self.std_M[self.id_list_M.index(str(id_M))]
        clip_up_S_M = (S_high_mu - self.mean_M[self.id_list_M.index(str(id_M))]) / self.std_M[self.id_list_M.index(str(id_M))]

        diff_SfromS = np.clip(real_S-SfromS, -100,100)
        diff_SfromM = np.clip(real_S-SfromM, -100,100)
        diff_SfromH = np.clip(real_S-SfromH, -100,100)
        diff_MfromS = np.clip(real_M-MfromS, -100,100)
        diff_MfromM = np.clip(real_M-MfromM, -100,100)
        diff_MfromH = np.clip(real_M-MfromH, -100,100)
        diff_HfromS = np.clip(real_H-HfromS, -100,100)
        diff_HfromM = np.clip(real_H-HfromM, -100,100)
        diff_HfromH = np.clip(real_H-HfromH, -100,100)

        real_S = np.clip(real_S, clip_low_S, clip_up_S)
        real_M_Swin = np.clip(real_M, clip_low_S_M, clip_up_S_M)
        real_M_Hwin = np.clip(real_M, clip_low_H_M, clip_up_H_M)
        real_H = np.clip(real_H, clip_low_H, clip_up_H)

        SfromS = np.clip(SfromS, clip_low_S, clip_up_S)
        SfromM = np.clip(SfromM, clip_low_S, clip_up_S)
        SfromH = np.clip(SfromH, clip_low_S, clip_up_S)
        MfromS_Swin = np.clip(MfromS, clip_low_S_M, clip_up_S_M)
        MfromM_Swin = np.clip(MfromM, clip_low_S_M, clip_up_S_M)
        MfromH_Swin = np.clip(MfromH, clip_low_S_M, clip_up_S_M)
        MfromS_Hwin = np.clip(MfromS, clip_low_H_M, clip_up_H_M)
        MfromM_Hwin = np.clip(MfromM, clip_low_H_M, clip_up_H_M)
        MfromH_Hwin = np.clip(MfromH, clip_low_H_M, clip_up_H_M)
        HfromS = np.clip(HfromS, clip_low_H, clip_up_H)
        HfromM = np.clip(HfromM, clip_low_H, clip_up_H)
        HfromH = np.clip(HfromH, clip_low_H, clip_up_H)

        display_list = [SfromS[0,0,...], MfromS_Swin[0,0,...], MfromS_Hwin[0,0,...], HfromS[0,0,...],
                        SfromM[0,0,...], MfromM_Swin[0,0,...], MfromM_Hwin[0,0,...], HfromM[0,0,...],
                        SfromH[0,0,...], MfromH_Swin[0,0,...], MfromH_Hwin[0,0,...], HfromH[0,0,...],
                        real_S[0,0,...], real_M_Swin[0,0,...], real_M_Hwin[0,0,...], real_H[0,0,...]]
        title = ['SfromS', 'MfromS_Swin', 'MfromS_Hwin', 'HfromS',
                 'SfromM', 'MfromM_Swin', 'MfromM_Hwin', 'HfromM',
                 'SfromH', 'MfromH_Swin', 'MfromH_Hwin', 'HfromH',
                 'real_S', 'real_M_Swin', 'real_M_Hwin', 'real_H']

        # Results
        fig = plt.figure(figsize=(50, 50))
        for i in range(len(display_list)):
            axs = fig.add_subplot(4, 4, i + 1)
            axs.set_title(label=title[i], loc='center', size=50)
            im = axs.imshow(np.squeeze(display_list[i] * 1024), cmap='gray')
            fig.colorbar(im, ax=axs)
            if i in [0,4,8,12]:
                im.set_clim(vmin=clip_low_S , vmax=clip_up_S)
            elif i in [3,7,11,15]:
                im.set_clim(vmin=clip_low_H , vmax=clip_up_H)
            elif i in [2,6,10,14]:
                im.set_clim(vmin=clip_low_H_M , vmax=clip_up_H_M)
            elif i in [1,5,9,13]:
                im.set_clim(vmin=clip_low_S_M , vmax=clip_up_S_M)

        fig.savefig(save_dir + '/Summary_epoch_{}_step_{}_test.png'.format(epoch, step))
        fig.clear()
        plt.clf()
        plt.close('all')

        display_list_diff = [diff_SfromS[0, 0, ...], diff_MfromS[0, 0, ...], diff_HfromS[0, 0, ...],
                             diff_SfromM[0, 0, ...], diff_MfromM[0, 0, ...], diff_HfromM[0, 0, ...],
                             diff_SfromH[0, 0, ...], diff_MfromH[0, 0, ...], diff_HfromH[0, 0, ...]]
        title_diff = ['SfromS', 'MfromS', 'HfromS',
                      'SfromM', 'MfromM', 'HfromM',
                      'SfromH', 'MfromH', 'HfromH']

        # Results
        fig = plt.figure(figsize=(50, 50))
        for i in range(len(display_list_diff)):
            axs = fig.add_subplot(3,3, i + 1)
            axs.set_title(label=title_diff[i], loc='center', size=50)
            im = axs.imshow(np.squeeze(display_list_diff[i] * 1024), cmap='gray')
            fig.colorbar(im, ax=axs)
            im.set_clim(vmin=-100, vmax=100)


        fig.savefig(save_dir + '/Summary_epoch_{}_step_{}_test_diff.png'.format(epoch, step))
        fig.clear()
        plt.clf()
        plt.close('all')

    def save_image_summary_test_M_parallel(self, model, dataset_val, index, epoch):
        save_dir = os.path.join(self.opt.save_dir, '{}/validation_id_{}_{}/images/val'.format(self.opt.name, self.val_id[0], self.val_id[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.opt.gpu_parallel:
            self.device = torch.device('cuda:{}'.format(self.opt.multiple_ids[0]))
        else:
            self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids))

        real_S = torch.zeros((len(index), 1, 512, 512))
        real_M = torch.zeros((len(index), 1, 512, 512))
        real_H = torch.zeros((len(index), 1, 512, 512))
        id_HS = []
        id_M = []
        for j, idx in enumerate(index):
            if idx > len(dataset_val.path_M):
                index[j] = len(dataset_val.path_M)-10
            else:
                pass

        for j, idx in enumerate(index):
            real_S[j] = dataset_val[idx]['real_S'].to(self.device)
            real_M[j] = dataset_val[idx]['real_M'].to(self.device)
            real_H[j] = dataset_val[idx]['real_H'].to(self.device)
            id_HS.append(dataset_val[idx]['id'][0])
            id_M.append(dataset_val[idx]['id'][2])

        real_S = real_S.to(self.device)
        real_M = real_M.to(self.device)
        real_H = real_H.to(self.device)
        with torch.no_grad():
            SfromH = model.netG(real_H, alpha_s=0.0, alpha_t=1.0).cpu().detach().numpy()
            MfromH = model.netG(real_H, alpha_s=0.0, alpha_t=0.5).cpu().detach().numpy()
            HfromH = model.netG(real_H, alpha_s=0.0, alpha_t=0.0).cpu().detach().numpy()

            SfromS = model.netG(real_S, alpha_s=1.0, alpha_t=1.0).cpu().detach().numpy()
            MfromS = model.netG(real_S, alpha_s=1.0, alpha_t=0.5).cpu().detach().numpy()
            HfromS = model.netG(real_S, alpha_s=1.0, alpha_t=0.0).cpu().detach().numpy()

            SfromM = model.netG(real_M, alpha_s=0.5, alpha_t=1.0).cpu().detach().numpy()
            MfromM = model.netG(real_M, alpha_s=0.5, alpha_t=0.5).cpu().detach().numpy()
            HfromM = model.netG(real_M, alpha_s=0.5, alpha_t=0.0).cpu().detach().numpy()

            real_S = real_S.cpu().detach().numpy()
            real_M = real_M.cpu().detach().numpy()
            real_H = real_H.cpu().detach().numpy()


        S_win_level, S_win_width = 50, 120
        H_win_level, H_win_width = 400, 1500

        S_low = S_win_level - S_win_width/2
        S_high = S_win_level + S_win_width/2
        H_low = H_win_level - H_win_width/2
        H_high = H_win_level + H_win_width/2
        S_low_mu = (S_low*0.0192/1000) + 0.0192
        S_high_mu = (S_high*0.0192/1000) + 0.0192
        H_low_mu = (H_low * 0.0192 / 1000) + 0.0192
        H_high_mu = (H_high * 0.0192 / 1000) + 0.0192

        diff_SfromS = np.clip(real_S - SfromS, -100, 100)
        diff_SfromM = np.clip(real_S - SfromM, -100, 100)
        diff_SfromH = np.clip(real_S - SfromH, -100, 100)
        diff_MfromS = np.clip(real_M - MfromS, -100, 100)
        diff_MfromM = np.clip(real_M - MfromM, -100, 100)
        diff_MfromH = np.clip(real_M - MfromH, -100, 100)
        diff_HfromS = np.clip(real_H - HfromS, -100, 100)
        diff_HfromM = np.clip(real_H - HfromM, -100, 100)
        diff_HfromH = np.clip(real_H - HfromH, -100, 100)

        for j, idx in enumerate(index):
            clip_low_H = (H_low_mu - self.mean_H[self.id_list_HS.index(str(id_HS[j]))])/self.std_H[self.id_list_HS.index(str(id_HS[j]))]
            clip_up_H = (H_high_mu - self.mean_H[self.id_list_HS.index(str(id_HS[j]))])/self.std_H[self.id_list_HS.index(str(id_HS[j]))]
            clip_low_S = (S_low_mu - self.mean_S[self.id_list_HS.index(str(id_HS[j]))]) / self.std_S[self.id_list_HS.index(str(id_HS[j]))]
            clip_up_S = (S_high_mu - self.mean_S[self.id_list_HS.index(str(id_HS[j]))]) / self.std_S[self.id_list_HS.index(str(id_HS[j]))]
            clip_low_H_M = (H_low_mu - self.mean_M[self.id_list_M.index(str(id_M[j]))]) / self.std_M[self.id_list_M.index(str(id_M[j]))]
            clip_up_H_M = (H_high_mu - self.mean_M[self.id_list_M.index(str(id_M[j]))]) / self.std_M[self.id_list_M.index(str(id_M[j]))]
            clip_low_S_M = (S_low_mu - self.mean_M[self.id_list_M.index(str(id_M[j]))]) / self.std_M[self.id_list_M.index(str(id_M[j]))]
            clip_up_S_M = (S_high_mu - self.mean_M[self.id_list_M.index(str(id_M[j]))]) / self.std_M[self.id_list_M.index(str(id_M[j]))]

            real_S = np.clip(real_S, clip_low_S, clip_up_S)
            real_M_Swin = np.clip(real_M, clip_low_S_M, clip_up_S_M)
            real_M_Hwin = np.clip(real_M, clip_low_H_M, clip_up_H_M)
            real_H = np.clip(real_H, clip_low_H, clip_up_H)

            SfromS = np.clip(SfromS, clip_low_S, clip_up_S)
            SfromM = np.clip(SfromM, clip_low_S, clip_up_S)
            SfromH = np.clip(SfromH, clip_low_S, clip_up_S)
            MfromS_Swin = np.clip(MfromS, clip_low_S_M, clip_up_S_M)
            MfromM_Swin = np.clip(MfromM, clip_low_S_M, clip_up_S_M)
            MfromH_Swin = np.clip(MfromH, clip_low_S_M, clip_up_S_M)
            MfromS_Hwin = np.clip(MfromS, clip_low_H_M, clip_up_H_M)
            MfromM_Hwin = np.clip(MfromM, clip_low_H_M, clip_up_H_M)
            MfromH_Hwin = np.clip(MfromH, clip_low_H_M, clip_up_H_M)
            HfromS = np.clip(HfromS, clip_low_H, clip_up_H)
            HfromM = np.clip(HfromM, clip_low_H, clip_up_H)
            HfromH = np.clip(HfromH, clip_low_H, clip_up_H)

            display_list = [SfromS[j,0,...], MfromS_Swin[j,0,...], MfromS_Hwin[j,0,...], HfromS[j,0,...],
                            SfromM[j,0,...], MfromM_Swin[j,0,...], MfromM_Hwin[j,0,...], HfromM[j,0,...],
                            SfromH[j,0,...], MfromH_Swin[j,0,...], MfromH_Hwin[j,0,...], HfromH[j,0,...],
                            real_S[j,0,...], real_M_Swin[j,0,...], real_M_Hwin[j,0,...], real_H[j,0,...]]
            title = ['SfromS', 'MfromS_Swin', 'MfromS_Hwin', 'HfromS',
                     'SfromM', 'MfromM_Swin', 'MfromM_Hwin', 'HfromM',
                     'SfromH', 'MfromH_Swin', 'MfromH_Hwin', 'HfromH',
                     'real_S', 'real_M_Swin', 'real_M_Hwin', 'real_H']

            # Results
            fig = plt.figure(figsize=(50, 50))
            for i in range(len(display_list)):
                axs = fig.add_subplot(4, 4, i + 1)
                axs.set_title(label=title[i], loc='center', size=50)
                im = axs.imshow(np.squeeze(display_list[i]), cmap='gray')
                fig.colorbar(im, ax=axs)
                if i in [0,4,8,12]:
                    im.set_clim(vmin=clip_low_S, vmax=clip_up_S)
                elif i in [3,7,11,15]:
                    im.set_clim(vmin=clip_low_H, vmax=clip_up_H)
                elif i in [2,6,10,14]:
                    im.set_clim(vmin=clip_low_H_M, vmax=clip_up_H_M)
                elif i in [1,5,9,13]:
                    im.set_clim(vmin=clip_low_S_M, vmax=clip_up_S_M)

            fig.savefig(save_dir + '/Summary_epoch_{}_step_{}_test.png'.format(epoch, idx))
            fig.clear()
            plt.clf()
            plt.close('all')

            display_list_diff = [diff_SfromS[j, 0, ...], diff_MfromS[j, 0, ...], diff_HfromS[j, 0, ...],
                                 diff_SfromM[j, 0, ...], diff_MfromM[j, 0, ...], diff_HfromM[j, 0, ...],
                                 diff_SfromH[j, 0, ...], diff_MfromH[j, 0, ...], diff_HfromH[j, 0, ...]]
            title_diff = ['SfromS', 'MfromS', 'HfromS',
                          'SfromM', 'MfromM', 'HfromM',
                          'SfromH', 'MfromH', 'HfromH']

            # Results
            fig = plt.figure(figsize=(50, 50))
            for i in range(len(display_list_diff)):
                axs = fig.add_subplot(3,3, i + 1)
                axs.set_title(label=title_diff[i], loc='center', size=50)
                im = axs.imshow(np.squeeze(display_list_diff[i] * 1024), cmap='gray')
                fig.colorbar(im, ax=axs)
                im.set_clim(vmin=-100, vmax=100)


            fig.savefig(save_dir + '/Summary_epoch_{}_step_{}_test_diff.png'.format(epoch, idx))
            fig.clear()
            plt.clf()
            plt.close('all')