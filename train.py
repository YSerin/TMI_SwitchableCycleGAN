import argparse
import logging
import time

from torch.utils.data import DataLoader
import torch

from Dataset_size import Dataset_fold_best as Dataset
from Dataset_size import Dataset_3fold_best_M as Dataset_M
from Model import Model
from utils.print_options import print_options
from utils.visualizer import Visualizer
import ipdb
#######################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser()

#settings
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

parser.add_argument('--data_type', type=str, default='Facial_bone', help='Facial_bone or Head or Head_v2')
parser.add_argument('--Mkernel', type=str, default='None', help='Hr49 or Hr56')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--continue_train', type=int, default=0, help='continue training: load the latest model')
parser.add_argument('--load_val_id', type=str, default='0,1,2,3,4', help='training set')
parser.add_argument('--load_val_TF', type=str, default='False,False,False,False,False', help='continue or not for each training set')
parser.add_argument('--load_epoch_list', type=str, default='0,0,0,0,0', help='num_epoch to load when opt.load_val_TF is True')
parser.add_argument('--save_best_model', type=int, default=1, help='True: save only best model, False: save model at each best epoch')
parser.add_argument('--load_best_model', type=int, default=1, help='True: save only best model, False: save model at each best epoch')
parser.add_argument('--data_size', type=str, default='whole', help='whole, half, quarter')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')

#parameters
parser.add_argument('--isTrain', action='store_false', help='Train or Test')
parser.add_argument('--img_crop_size', type = int, default=128, help='img_crop_size')
parser.add_argument('--batch_size', type = int, default=16, help='batch size; 24 for parallel-gpus, 8 for single-gpu')
parser.add_argument('--shuffle', type=int, default=1, help='shuffle')
parser.add_argument('--lambda_gan', type=float, default = 1, help = 'weight for gan loss')
parser.add_argument('--lambda_cycle', type = float, default=10, help='weight cycle loss')
parser.add_argument('--lambda_identity', type = float, default=5, help='weight identity loss')
parser.add_argument('--lr', type = float, default=1e-5, help='learning rate')
parser.add_argument('--lr_schedule', type = int, default=0, help='learning rate')
parser.add_argument('--milestones', type = str, default='5,10', help='milestone for learning rate schedule')
parser.add_argument('--beta1', type = int, default=0.9, help='beta for optimizer Adam')
parser.add_argument('--pool_size', type = int, default=50, help='size of buffer which stores previously generated images (fake_c)')
parser.add_argument('--pool_prob', type = float, default=0.5, help='probability of choosing matched image as an input of discriminator')

#intervals
parser.add_argument('--num_display_img_interval', type = int, default=100, help='frequency of displaying images on tensorboard')    #100
parser.add_argument('--num_display_loss_interval', type = int, default=240, help='frequency of displaying images on tensorboard')    #72(Head), 240(FB)
parser.add_argument('--num_save_interval', type = int, default=480, help='frequency of saving images')                                #240(Head), 480(FB)
parser.add_argument('--num_print_interval', type = int, default=480, help='frequency of printing losses on console & saving as txt file')   #240(Head), 480(FB)
parser.add_argument('--model_save_interval', type = int, default=50, help='frequency of saving models')

parser.add_argument('--gpu_ids', type=str, default='4', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU for NOT Parallel learning')
parser.add_argument('--gpu_parallel', type=int, default=0, help='Parallel learning or Not')
parser.add_argument('--multiple_ids', type=str, default='4,5', help='device_ids for Parallel learning')
parser.add_argument('--save_dir', type=str, default='./result/checkpoints', help='directory for model save')
parser.add_argument('--data_dir', type=str, default='./dataset/', help='directory for data')

opt = parser.parse_args()

temp = []
for i in range(len(opt.load_val_id.split(','))):
    temp.append(int(opt.load_val_id.split(',')[i]))
opt.load_val_id = temp

temp = []
for i in range(len(opt.multiple_ids.split(','))):
    temp.append(int(opt.multiple_ids.split(',')[i]))
opt.multiple_ids = temp

temp = []
for i in range(len(opt.load_epoch_list.split(','))):
    temp.append(int(opt.load_epoch_list.split(',')[i]))
opt.load_epoch_list = temp

temp = []
for i in range(len(opt.load_val_TF.split(','))):
    if opt.load_val_TF.split(',')[i] == 'True':
        temp.append(True)
    elif opt.load_val_TF.split(',')[i] == 'False':
        temp.append(False)
opt.load_val_TF = temp


print_options(parser, opt)


for val_i in opt.load_val_id:
    if opt.data_dir.split('/')[-2] == 'dataset_KEY_M':
        dataset = Dataset_M(opt, phase='train', M=opt.Mkernel, test_index=val_i, size=opt.data_size)
        dataset_val = Dataset_M(opt, phase='validation', M=opt.Mkernel, test_index=val_i, size=opt.data_size)
        opt.val_id = [dataset_val.dir[0][-8:], dataset_val.dir[1][-8:]]
    elif opt.data_dir.split('/')[-2] == 'dataset_KEY':
        dataset = Dataset(opt, phase='train', test_index=val_i, size = opt.data_size)
        dataset_val = Dataset(opt, phase='validation', test_index=val_i,  size = opt.data_size)
        opt.val_id = [dataset_val.dir[0][-8:], dataset_val.dir[1][-8:]]
    else:
        raise ValueError

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=2)
    if opt.gpu_parallel:
        batch_size_val = len(opt.multiple_ids)
    else:
        batch_size_val = 1
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=1)
    dataset_size = len(dataset)
    print("Train Dataset size: ", dataset_size)
    print("Validation Dataset size: {} \t Validation ID: {}, {}".format(len(dataset_val), opt.val_id[0], opt.val_id[1]))


    current_i = opt.load_val_TF[val_i]
    if current_i:
        current_epoch = opt.load_epoch_list[val_i]
        opt.load_epoch = opt.load_epoch_list[val_i]
    else:
        current_epoch = 0


    model = Model(opt,current_i)
    visualizer = Visualizer(opt)

    total_iters = 0
    PSNR_H_max, PSNR_S_max = 0, 0
    SSIM_H_max, SSIM_S_max = 0, 0

    if opt.continue_train and current_i:
        epoch, PSNR_H_max, PSNR_S_max, SSIM_H_max, SSIM_S_max = model.load_best_metrics()
        print("Last Best Metrics on validation set \t (epoch: %3d) PSNR_H: %2.4f / PSNR_S: %2.4f / SSIM_H: %1.4f / SSIM_S: %1.4f"
              % (epoch, PSNR_H_max, PSNR_S_max, SSIM_H_max, SSIM_S_max))

    # main loop
    for epoch in range(current_epoch, opt.epochs):
        current_epoch += 1
        epoch_start_time = time.time()
        epoch_iters = 0
        step_train = 0

        model.train()
        # train for an epoch
        for i, data in enumerate(dataloader):
            if opt.gpu_parallel and len(data['real_S']) == 1:
               pass
            else:
                model.set_input(data)
                model.optimize_parameters()

                total_iters += opt.batch_size
                epoch_iters += opt.batch_size
                step_train += 1

                if epoch_iters % opt.num_display_loss_interval == 0:
                    visualizer.display_losses(model.get_current_losses(), current_epoch, epoch_iters / dataset_size)
                if epoch_iters % opt.num_save_interval == 0 and current_epoch % 5 == 0:
                    visualizer.save_image_summary_train(model.get_current_visuals(), current_epoch, epoch_iters, data['id'])
                if epoch_iters % opt.num_print_interval == 0:
                    visualizer.print_losses(model.get_current_losses(), current_epoch, epoch_iters)

        if current_epoch % opt.model_save_interval == 0 :
            print('saving the model at the end of epoch %d \t Time Taken: %d sec for 1 epoch' % (current_epoch, time.time()-epoch_start_time))
            model.save_networks(current_epoch)

        # validation
        model.eval()
        PSNR_H, PSNR_S = 0, 0
        SSIM_H, SSIM_S = 0, 0
        flag_save_images = False
        for i, data in enumerate(dataloader_val):
            if opt.gpu_parallel and data['real_S'].shape[0] % len(opt.multiple_ids) != 0:
                denom = len(dataset_val) - data['real_S'].shape[0]
            else:
                model.set_input(data)
                model.test()

                visualizer.display_losses(model.get_current_losses(validation=True), current_epoch, epoch_iters / dataset_size)
                s_H, s_S, p_H, p_S = model.eval_metrics()
                PSNR_H += p_H
                PSNR_S += p_S
                SSIM_H += s_H
                SSIM_S += s_S

                if (current_epoch % opt.model_save_interval == 0) and (i == 0 or i == 170):
                    visualizer.save_image_summary_val(model, current_epoch, i, data['id'])
                    flag_save_images = True

        if opt.gpu_parallel:
            PSNR_H = torch.sum(PSNR_H)
            PSNR_S = torch.sum(PSNR_S)
            SSIM_H *= len(opt.multiple_ids)
            SSIM_S *= len(opt.multiple_ids)

            if data['real_S'].shape[0] % len(opt.multiple_ids) != 0:
                PSNR_H /= denom
                PSNR_S /= denom
                SSIM_H /= denom
                SSIM_S /= denom
            else:
                PSNR_H /= len(dataset_val)
                PSNR_S /= len(dataset_val)
                SSIM_H /= len(dataset_val)
                SSIM_S /= len(dataset_val)
        else:
            PSNR_H /= len(dataset_val)
            PSNR_S /= len(dataset_val)
            SSIM_H /= len(dataset_val)
            SSIM_S /= len(dataset_val)

        print('epoch %d \t PSNR_H: %2.2f / PSNR_S: %2.2f' % (current_epoch, PSNR_H, PSNR_S))
        print('epoch %d \t SSIM_H: %2.4f / SSIM_S: %2.4f' % (current_epoch, SSIM_H, SSIM_S))

        # save model with highest PSNR
        if (PSNR_H > PSNR_H_max and PSNR_S > PSNR_S_max and SSIM_H > SSIM_H_max and SSIM_S > SSIM_S_max) \
                or (PSNR_H > 30.2 and PSNR_S > 20.5 and SSIM_H > 0.85 and SSIM_S > 0.83):
            PSNR_H_max = PSNR_H
            PSNR_S_max = PSNR_S
            SSIM_H_max = SSIM_H
            SSIM_S_max = SSIM_S

            if current_epoch > 10:
                model.save_networks(current_epoch)
                if not flag_save_images:
                    visualizer.save_image_summary_test_parallel(model, dataset_val, [0, 80], current_epoch)

            model.save_best_metrics(current_epoch, PSNR_H_max, PSNR_S_max, SSIM_H_max, SSIM_S_max)
            message = '(epoch: %d, PSNR_H: %2.4f, PSNR_S: %2.4f, SSIM_H: %2.4f, SSIM_S: %2.4f) ' % (current_epoch,PSNR_H_max,PSNR_S_max,SSIM_H_max,SSIM_S_max)
            val_metric_name = opt.save_dir + '/' + opt.name + '/validation_id_{}_{}/'.format(opt.val_id[0], opt.val_id[1]) + 'validation_metrics.txt'
            with open(val_metric_name, "a") as f:
                f.write('%s\n' % message)

        print('End of epoch %d \t Time Taken: %d sec for 1 epoch' % (current_epoch, time.time() - epoch_start_time))

