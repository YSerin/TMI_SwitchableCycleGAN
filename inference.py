import argparse
import os

import numpy as np
import pydicom
import torch
import pytorch_ssim

from Model import Model
from Dataset_size import Dataset_3fold_best_M as Dataset_M
from Dataset_size import Dataset_fold_best as Dataset
from torch.utils.data import DataLoader
from inference_M import _write_dicom_fakeHr68, _write_dicom_fakeHr40


def _eval_metrics(img_H, img_S, img_HfromS, img_SfromH):
    img_H /= 1150
    img_HfromS /= 1150
    img_S /= 110
    img_SfromH /= 110

    mse_H = torch.mean((img_H - img_HfromS) ** 2, dim=[1, 2, 3])
    mse_S = torch.mean((img_S - img_SfromH) ** 2, dim=[1, 2, 3])
    score_H = -10 * torch.log10(mse_H)
    score_S = -10 * torch.log10(mse_S)

    ssim_H = pytorch_ssim.ssim(img_H, img_HfromS)
    ssim_S = pytorch_ssim.ssim(img_S, img_SfromH)

    return ssim_H, ssim_S, torch.mean(score_H), torch.mean(score_S)

def _windowing(input, H=True):
    if H :
        output = torch.clamp(input, 400-1500/2, 400+1500/2)
    else:
        output = torch.clamp(input, 50-120/2, 50+120/2)
    return output

def _write_dicom_fakeS(path_H, fake_S, alpha, opt):
    H_new = pydicom.dcmread(path_H)
    fake_S = ((fake_S - H_new.RescaleIntercept) / H_new.RescaleSlope).cpu().detach().numpy()
    H_new.PixelData = np.array(np.clip(fake_S, a_min=0, a_max=np.inf), dtype=np.uint16)      
    H_new.SeriesDescription = H_new.SeriesDescription.replace('J70h', 'S from H')

    id = path_H.split('/H/')[1]
    save_path = opt.save_dir + '/' + opt.name + '/validation_id_{}_{}/testset/{}ep/{}/S from H/'.format(opt.val_id[0],opt.val_id[1], opt.load_epoch, alpha) + id
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H_new.save_as(save_path, True)
    del H_new, fake_S


def _write_dicom_fakeH(path_S, fake_H, alpha, opt):
    S_new = pydicom.dcmread(path_S)
    fake_H = ((fake_H - S_new.RescaleIntercept) / S_new.RescaleSlope).cpu().detach().numpy()
    S_new.PixelData = np.array(np.clip(fake_H, a_min=0, a_max=np.inf), dtype=np.uint16)  
    if opt.data_type == 'Head':
        S_new.SeriesDescription = S_new.SeriesDescription.replace('J30s', 'H from S')
    elif opt.data_type == 'Facial_bone':
        S_new.SeriesDescription = S_new.SeriesDescription.replace('J40s', 'H from S')
    else:
        raise ValueError

    id = path_S.split('/S/')[1]
    save_path = opt.save_dir + '/' + opt.name + '/validation_id_{}_{}/testset/{}ep/{}/H from S/'.format(opt.val_id[0],opt.val_id[1], opt.load_epoch, alpha) + id
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    S_new.save_as(save_path, True)
    del S_new, fake_H

def _unpreprocessing(image):
    output = image
    mu_h2o = 0.0192
    output = (output - mu_h2o) * 1000 / mu_h2o     
    return output

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--data_type', type=str, default='Facial_bone', help='Facial_bone or Head')
    parser.add_argument('--name', type=str, default='last', help='name of the experiment')
    parser.add_argument('--load_val_id', type=str, default='4', help='if specified, print more debugging information')
    parser.add_argument('--load_epoch_list', type=str, default='0,0,0,0,191', help='if specified, print more debugging information')
    parser.add_argument('--load_best_model', type=int, default=0, help='True: save only best model, False: save model at each best epoch')
    parser.add_argument('--alphas', type=str, default='0.0,0.5,1.0', help='alpha values to be tested')
    parser.add_argument('--data_size', type=str, default='whole', help='whole, half, quarter')
    parser.add_argument('--write_dicom', type=str, default='False,False,False,False,False', help='if specified, print more debugging information')

    parser.add_argument('--isTrain', action='store_true', help='Train or Test')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--shuffle', type=int, default=0, help='shuffle')

    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--gpu_parallel', type=int, default=0, help='Parallel or Not for Inference')  
    parser.add_argument('--gpu_parallel_train', type=int, default=0, help='Parallel learning or Not for Training')  
    parser.add_argument('--save_dir', type=str, default='./result/checkpoints', help='directory for model save')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='directory for model save')

    return parser.parse_args()

def main():
    opt = parse_arguments()
    temp = []
    for i in range(len(opt.load_val_id.split(','))):
        temp.append(int(opt.load_val_id.split(',')[i]))
    opt.load_val_id = temp

    temp = []
    for i in range(len(opt.load_epoch_list.split(','))):
        temp.append(int(opt.load_epoch_list.split(',')[i]))
    opt.load_epoch_list = temp

    temp = []
    for i in range(len(opt.alphas.split(','))):
        temp.append(opt.alphas.split(',')[i])
    opt.alphas = temp
    print(opt)

    id_list = sorted(os.listdir(opt.data_dir + '{}_total'.format(opt.data_type)))
    dir_stat = opt.data_dir + 'statistic_{}'.format(opt.data_type)
    if opt.data_dir.split('/')[-2] == 'dataset_KEY_M':
        mean_H = np.load(dir_stat + '/mean_Hr68.npz')['meanlist']
        mean_M = np.load(dir_stat + '/mean_Hr49.npz')['meanlist']
        mean_S = np.load(dir_stat + '/mean_Hr40.npz')['meanlist']
        std_H = np.load(dir_stat + '/std_Hr68.npz')['stdlist']
        std_M = np.load(dir_stat + '/std_Hr49.npz')['stdlist']
        std_S = np.load(dir_stat + '/std_Hr40.npz')['stdlist']
    elif opt.data_dir.split('/')[-2] == 'dataset_KEY':
        mean_H = np.load(dir_stat + '/mean_H.npz')['meanlist']
        mean_S = np.load(dir_stat + '/mean_S.npz')['meanlist']
        std_H = np.load(dir_stat + '/std_H.npz')['stdlist']
        std_S = np.load(dir_stat + '/std_S.npz')['stdlist']
    else:
        raise ValueError

    for idx in opt.load_val_id:


        if opt.data_dir.split('/')[-2] == 'dataset_M':
            dataset = Dataset_M(opt, phase='test', size=opt.data_size)
            opt.val_id = [dataset.id_list_train[2*idx],dataset.id_list_train[2*idx+1]]
        elif opt.data_dir.split('/')[-2] == 'dataset':
            dataset = Dataset(opt, phase='test')
            opt.val_id = [dataset.id_list_train[idx], dataset.id_list_train[idx + 5]]
        else:
            raise ValueError
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=1)
        dataset_size = len(dataset) 
        print(dataset_size)

        opt.load_epoch = opt.load_epoch_list[idx]
        if opt.load_epoch:

            model = Model(opt, current_i = False)
            model.eval()
            for alpha in opt.alphas:
                print("Alpha: {}".format(alpha))
                img_H = np.zeros((dataset_size, 1, 512, 512))
                img_S = np.zeros((dataset_size, 1, 512, 512))
                img_HfromS = np.zeros((dataset_size, 1, 512, 512))
                img_SfromH = np.zeros((dataset_size, 1, 512, 512))

                for i, data in enumerate(dataloader):

                    model.set_input(data)
                    id = model.id[0]
                    real_H = model.real_H
                    real_S = model.real_S
                    # forward
                    with torch.no_grad():
                        fake_H = model.netG(real_S,alpha_s=1.0, alpha_t= float(alpha)).cpu()
                        fake_S = model.netG(real_H,alpha_s=0.0, alpha_t= float(alpha)).cpu()

                    fake_H = fake_H.squeeze()
                    fake_S = fake_S.squeeze()

                    # un-normalize for each volume
                    fake_H = fake_H * std_S[id_list.index(id[0])] + mean_S[id_list.index(id[0])]
                    fake_S = fake_S * std_H[id_list.index(id[0])] + mean_H[id_list.index(id[0])]
                    real_H = real_H * std_H[id_list.index(id[0])] + mean_H[id_list.index(id[0])]
                    real_S = real_S * std_S[id_list.index(id[0])] + mean_S[id_list.index(id[0])]

                    # un-preprocessing
                    fake_H = _unpreprocessing(fake_H)
                    fake_S = _unpreprocessing(fake_S)
                    real_H = _unpreprocessing(real_H)
                    real_S = _unpreprocessing(real_S)

                    # write dicom
                    if opt.write_dicom[idx] and opt.data_dir.split('/')[-2] == 'dataset':
                        _write_dicom_fakeH(model.path_S[0], fake_H, alpha, opt)
                        _write_dicom_fakeS(model.path_H[0], fake_S, alpha, opt)
                    elif opt.write_dicom[idx] and opt.data_dir.split('/')[-2] == 'dataset_M':
                        _write_dicom_fakeHr68(model.path_S[0], fake_H, alpha, opt)
                        _write_dicom_fakeHr40(model.path_H[0], fake_S, alpha, opt)

                    img_H[i,0] = real_H.cpu().detach().numpy()
                    img_S[i,0] = real_S.cpu().detach().numpy()
                    img_HfromS[i,0] = fake_H.detach().numpy()
                    img_SfromH[i,0] = fake_S.detach().numpy()

                if (alpha == '0.0' or alpha == '1.0'):
                    real_H = _windowing(torch.tensor(img_H, dtype=torch.float32), H=True)
                    real_S = _windowing(torch.tensor(img_S, dtype=torch.float32), H=False)
                    fake_H = _windowing(torch.tensor(img_HfromS, dtype=torch.float32), H=True)
                    fake_S = _windowing(torch.tensor(img_SfromH, dtype=torch.float32), H=False)
                    s_H, s_S, p_H, p_S = _eval_metrics(real_H, real_S, fake_H, fake_S)

                    print('Val ID: {}, {}'.format(opt.val_id[0], opt.val_id[1]))
                    print('alpha %s \t PSNR_H: %2.4f / PSNR_S: %2.4f / SSIM_H: %2.4f / SSIM_S: %2.4f' % (alpha, p_H, p_S, s_H, s_S))


if __name__ == "__main__":
    main()








